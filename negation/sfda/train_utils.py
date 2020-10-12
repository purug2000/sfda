from transformers.trainer_utils import nested_concat,nested_numpify,Any
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import torch.nn as nn
import torch
from transformers.file_utils import is_torch_tpu_available
from tqdm import tqdm as tqdm

from typing import NamedTuple, Union,Tuple,Optional,Dict
import numpy as np
import logging
from transformers import Trainer


logger = logging.getLogger(__name__)
class sfdaPredictionOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[np.ndarray]
    metrics: Optional[Dict[str, float]]
    feat_matrix: Optional[np.ndarray]

class sfdaTrainer(Trainer):
        def __init__(
        self,
        **kwargs,
    ):
            super(sfdaTrainer,self).__init__(**kwargs)
        def predict(self, test_dataset: Dataset, ret_feats: Optional[bool] = None) -> sfdaPredictionOutput:
            """
            Run prediction and returns predictions and potential metrics.

            Depending on the dataset and your use case, your test dataset may contain labels.
            In that case, this method will also return metrics, like in :obj:`evaluate()`.

            Args:
                test_dataset (:obj:`Dataset`):
                    Dataset to run the predictions on. If it is an :obj:`datasets.Dataset`, columns not accepted by the
                    ``model.forward()`` method are automatically removed.

            Returns:
                `NamedTuple`:
                predictions (:obj:`np.ndarray`):
                    The predictions on :obj:`test_dataset`.
                label_ids (:obj:`np.ndarray`, `optional`):
                    The labels (if the dataset contained some).
                metrics (:obj:`Dict[str, float]`, `optional`):
                    The potential dictionary of metrics (if the dataset contained labels).
            """
            test_dataloader = self.get_test_dataloader(test_dataset)

            return self.prediction_loop(test_dataloader, description="Prediction",ret_feats = ret_feats)


        def prediction_loop(
        self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None , ret_feats: Optional[bool] = None,
        ) -> sfdaPredictionOutput:
            """
            Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

            Works both with or without labels.
            """
            if hasattr(self, "_prediction_loop"):
                warnings.warn(
                    "The `_prediction_loop` method is deprecated and won't be called in a future version, define `prediction_loop` in your subclass.",
                    FutureWarning,
                )
                return self._prediction_loop(dataloader, description, prediction_loss_only=prediction_loss_only)

            prediction_loss_only = (
                prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
            )

            assert not getattr(
                self.model.config, "output_attentions", False
            ), "The prediction loop does not work with `output_attentions=True`."
            assert not getattr(
                self.model.config, "output_hidden_states", False
            ), "The prediction loop does not work with `output_hidden_states=True`."

            model = self.model
            # multi-gpu eval
            if self.args.n_gpu > 1:
                model = torch.nn.DataParallel(model)
            else:
                model = self.model
            # Note: in torch.distributed mode, there's no point in wrapping the model
            # inside a DistributedDataParallel as we'll be under `no_grad` anyways.

            batch_size = dataloader.batch_size
            logger.info("***** Running %s *****", description)
            logger.info("  Num examples = %d", self.num_examples(dataloader))
            logger.info("  Batch size = %d", batch_size)
            eval_losses: List[float] = []
            preds: torch.Tensor = None
            label_ids: torch.Tensor = None
            feat_mat: torch.Tensor = None
            model.eval()

            if is_torch_tpu_available():
                dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

            if self.args.past_index >= 0:
                self._past = None

            disable_tqdm = not self.is_local_process_zero() or self.args.disable_tqdm
            for inputs in tqdm(dataloader, desc=description, disable=disable_tqdm):
                loss, logits, labels,feats = self.prediction_step(model, inputs, prediction_loss_only,ret_feats = ret_feats)
                batch_size = inputs[list(inputs.keys())[0]].shape[0]
                if loss is not None:
                    eval_losses.extend([loss] * batch_size)
                if logits is not None:
                    preds = logits if preds is None else nested_concat(preds, logits, dim=0)
                if labels is not None:
                    label_ids = labels if label_ids is None else nested_concat(label_ids, labels, dim=0)
                if feats is not None:
                    feat_mat = feats if feat_mat is None else nested_concat(feat_mat,feats)

            if self.args.past_index and hasattr(self, "_past"):
                # Clean the state at the end of the evaluation loop
                delattr(self, "_past")

            if self.args.local_rank != -1:
                # In distributed mode, concatenate all results from all nodes:
                if preds is not None:
                    preds = distributed_concat(preds, num_total_examples=self.num_examples(dataloader))
                if label_ids is not None:
                    label_ids = distributed_concat(label_ids, num_total_examples=self.num_examples(dataloader))
                if feat_mat is not None:
                    feat_mat = distributed_concat(feat_mat, num_total_examples=self.num_examples(dataloader))
            
            elif is_torch_tpu_available():
                # tpu-comment: Get all predictions and labels from all worker shards of eval dataset
                if preds is not None:
                    preds = nested_xla_mesh_reduce(preds, "eval_preds")
                if label_ids is not None:
                    label_ids = nested_xla_mesh_reduce(label_ids, "eval_label_ids")
                if feat_mat is not None:
                    feat_mat = nested_xla_mesh_reduce(feat_mat, "eval_feat_mat")
                if eval_losses is not None:
                    eval_losses = xm.mesh_reduce("eval_losses", torch.tensor(eval_losses), torch.cat).tolist()

            # Finally, turn the aggregated tensors into numpy arrays.
            if preds is not None:
                preds = nested_numpify(preds)
            if label_ids is not None:
                label_ids = nested_numpify(label_ids)
            if feat_mat is not None:
                feat_mat = nested_numpify(feat_mat)

            if self.compute_metrics is not None and preds is not None and label_ids is not None:
                metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
            else:
                metrics = {}
            if len(eval_losses) > 0:
                if self.args.local_rank != -1:
                    metrics["eval_loss"] = (
                        distributed_broadcast_scalars(eval_losses, num_total_examples=self.num_examples(dataloader))
                        .mean()
                        .item()
                    )
                else:
                    metrics["eval_loss"] = np.mean(eval_losses)

            # Prefix all keys with eval_
            for key in list(metrics.keys()):
                if not key.startswith("eval_"):
                    metrics[f"eval_{key}"] = metrics.pop(key)

            return sfdaPredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics,feat_matrix = feat_mat)


        def prediction_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], prediction_loss_only: bool, ret_feats: bool,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor],Optional[torch.Tensor]]:
            """
            Perform an evaluation step on :obj:`model` using obj:`inputs`.

            Subclass and override to inject custom behavior.

            Args:
                model (:obj:`nn.Module`):
                    The model to evaluate.
                inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                    The inputs and targets of the model.

                    The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                    argument :obj:`labels`. Check your model's documentation for all accepted arguments.
                prediction_loss_only (:obj:`bool`):
                    Whether or not to return the loss only.

            Return:
                Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
                A tuple with the loss, logits and labels (each being optional).
            """
            has_labels = all(inputs.get(k) is not None for k in self.args.label_names)
            inputs = self._prepare_inputs(inputs)

            with torch.no_grad():
                if ret_feats is True:
                    outputs = model(**inputs,return_dict = True)
#                     print(outputs)
                    loss = outputs.loss
                    logits = outputs.logits
#                     print(outputs.last_hidden_state.shape)
                    feats = outputs.last_hidden_state[:,0,:].detach()
                    if has_labels:
                        # The .mean() is to reduce in case of distributed training
                        loss = loss.mean().item()
                else:
                    feats = None
                    outputs = model(**inputs)
                    if has_labels:
                        # The .mean() is to reduce in case of distributed training
                        loss = outputs[0].mean().item()
                        logits = outputs[1:]
                    else:
                        loss = None
                        # Slicing so we get a tuple even if `outputs` is a `ModelOutput`.
                        logits = outputs[:]
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index if has_labels else self.args.past_index - 1]

            if prediction_loss_only:
                return (loss, None, None,feats)

            logits = tuple(logit.detach() for logit in logits)
            if len(logits) == 1:
                logits = logits[0]

            if has_labels:
                labels = tuple(inputs.get(name).detach() for name in self.args.label_names)
                if len(labels) == 1:
                    labels = labels[0]
            else:
                labels = None

            return (loss, logits, labels,feats)