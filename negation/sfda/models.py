from transformers.modeling_outputs import ModelOutput,MaskedLMOutput
from transformers.modeling_roberta import RobertaPreTrainedModel,RobertaModel,RobertaClassificationHead, RobertaLMHead
from typing import NamedTuple, Union,Tuple,Optional,Dict
import torch

from torch.nn import CrossEntropyLoss


from transformers.configuration_utils import PretrainedConfig
from transformers.file_utils import (
    DUMMY_INPUTS,
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    WEIGHTS_NAME,
    ModelOutput,
    cached_path,
    hf_bucket_url,
    is_remote_url,
    is_torch_tpu_available,
    replace_return_docstrings,
)
from torch import nn
import re
import os
import logging
from .utils import compute_plabel_and_conf


logger = logging.getLogger(__name__)

class sfdaNegationClassifierOutput(ModelOutput):
    loss: Optional[Union[torch.FloatTensor,list]] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    last_hidden_state: Optional[Tuple[torch.FloatTensor]] = None
        
        
class sfdaSourceRobertaNegation(RobertaPreTrainedModel):
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return sfdaNegationClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            last_hidden_state = outputs[0]
        )
       
        
class sfdaTargetRobertaNegation(RobertaPreTrainedModel):
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)
        self.classifier_t = RobertaClassificationHead(config)
        self.classifier_s2t = RobertaClassificationHead(config)

        self.init_weights()
    @classmethod
    def from_pretrained_source(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        '''
        Adapted from transformers.PreTrainedModel.from_pretrained() 
        '''
        config = kwargs.pop("config", None)
        state_dict = kwargs.pop("state_dict", None)
        cache_dir = kwargs.pop("cache_dir", None)
        from_tf = kwargs.pop("from_tf", False)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        local_files_only = kwargs.pop("local_files_only", False)
        use_cdn = kwargs.pop("use_cdn", True)
        mirror = kwargs.pop("mirror", None)

        # Load config if we don't provide a configuration
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                *model_args,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                **kwargs,
            )
        else:
            model_kwargs = kwargs

        # Load model
        if pretrained_model_name_or_path is not None:
            if os.path.isdir(pretrained_model_name_or_path):
                if from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")):
                    # Load from a TF 1.0 checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")
                elif from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)):
                    # Load from a TF 2.0 checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)
                elif os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)):
                    # Load from a PyTorch checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
                else:
                    raise EnvironmentError(
                        "Error no file named {} found in directory {} or `from_tf` set to False".format(
                            [WEIGHTS_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME + ".index"],
                            pretrained_model_name_or_path,
                        )
                    )
            elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
                archive_file = pretrained_model_name_or_path
            elif os.path.isfile(pretrained_model_name_or_path + ".index"):
                assert (
                    from_tf
                ), "We found a TensorFlow checkpoint at {}, please set from_tf to True to load from this checkpoint".format(
                    pretrained_model_name_or_path + ".index"
                )
                archive_file = pretrained_model_name_or_path + ".index"
            else:
                archive_file = hf_bucket_url(
                    pretrained_model_name_or_path,
                    filename=(TF2_WEIGHTS_NAME if from_tf else WEIGHTS_NAME),
                    use_cdn=use_cdn,
                    mirror=mirror,
                )

            try:
                # Load from URL or cache if already cached
                resolved_archive_file = cached_path(
                    archive_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                )
                if resolved_archive_file is None:
                    raise EnvironmentError
            except EnvironmentError:
                msg = (
                    f"Can't load weights for '{pretrained_model_name_or_path}'. Make sure that:\n\n"
                    f"- '{pretrained_model_name_or_path}' is a correct model identifier listed on 'https://huggingface.co/models'\n\n"
                    f"- or '{pretrained_model_name_or_path}' is the correct path to a directory containing a file named one of {WEIGHTS_NAME}, {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME}.\n\n"
                )
                raise EnvironmentError(msg)

            if resolved_archive_file == archive_file:
                logger.info("loading weights file {}".format(archive_file))
            else:
                logger.info("loading weights file {} from cache at {}".format(archive_file, resolved_archive_file))
        else:
            resolved_archive_file = None

        # Instantiate model.
        model = cls(config, *model_args, **model_kwargs)

        if state_dict is None and not from_tf:
            try:
                state_dict = torch.load(resolved_archive_file, map_location="cpu")
            except Exception:
                raise OSError(
                    "Unable to load weights from pytorch checkpoint file. "
                    "If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True. "
                )

        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        if from_tf:
            if resolved_archive_file.endswith(".index"):
                # Load from a TensorFlow 1.X checkpoint - provided by original authors
                model = cls.load_tf_weights(model, config, resolved_archive_file[:-6])  # Remove the '.index'
            else:
                # Load from our TensorFlow 2.0 checkpoints
                try:
                    from .modeling_tf_pytorch_utils import load_tf2_checkpoint_in_pytorch_model

                    model = load_tf2_checkpoint_in_pytorch_model(model, resolved_archive_file, allow_missing_keys=True)
                except ImportError:
                    logger.error(
                        "Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed. Please see "
                        "https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions."
                    )
                    raise
        else:
            # Convert old format to new format if needed from a PyTorch state_dict
            old_keys = []
            new_keys = []
            for key in state_dict.keys():
                new_key = None
                if "gamma" in key:
                    new_key = key.replace("gamma", "weight")
                if "beta" in key:
                    new_key = key.replace("beta", "bias")
                if new_key:
                    old_keys.append(key)
                    new_keys.append(new_key)
            for old_key, new_key in zip(old_keys, new_keys):
                state_dict[new_key] = state_dict.pop(old_key)

            # copy state_dict so _load_from_state_dict can modify it
            metadata = getattr(state_dict, "_metadata", None)
            state_dict = state_dict.copy()
            if metadata is not None:
                state_dict._metadata = metadata

            # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
            # so we need to apply the function recursively.
#             print(F'{state_dict}')
            def load(module: nn.Module, prefix=""):
#                 print(F"{module} from {prefix}")
                local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
                module._load_from_state_dict(
                    state_dict,
                    prefix,
                    local_metadata,
                    True,
                    missing_keys,
                    unexpected_keys,
                    error_msgs,
                )
                for name, child in module._modules.items():
                    if child is not None:
                        load(child, prefix + name + ".")
       
            # Make sure we are able to load base models as well as derived models (with heads)
            start_prefix = ""
            model_to_load = model
            ###########################################################################################
            ###########################################################################################
            
            start_prefix = cls.base_model_prefix + "."
            load(model_to_load.classifier_t,prefix = "classifier.")
            load(model_to_load.classifier_s2t,prefix = "classifier.")
            load(model_to_load.roberta, prefix=start_prefix)
            
            ###########################################################################################
            ###########################################################################################
            if model.__class__.__name__ != model_to_load.__class__.__name__:
                base_model_state_dict = model_to_load.state_dict().keys()
                head_model_state_dict_without_base_prefix = [
                    key.split(cls.base_model_prefix + ".")[-1] for key in model.state_dict().keys()
                ]
                missing_keys.extend(head_model_state_dict_without_base_prefix - base_model_state_dict)

                # Some models may have keys that are not in the state by design, removing them before needlessly warning
            # the user.
            if cls.authorized_missing_keys is not None:
                for pat in cls.authorized_missing_keys:
                    missing_keys = [k for k in missing_keys if re.search(pat, k) is None]

            if cls.authorized_unexpected_keys is not None:
                for pat in cls.authorized_unexpected_keys:
                    unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

            if len(unexpected_keys) > 0:
                logger.warning(
                    f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when "
                    f"initializing {model.__class__.__name__}: {unexpected_keys}\n")
            else:
                logger.info(f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n")
            
            if len(error_msgs) > 0:
                raise RuntimeError(
                    "Error(s) in loading state_dict for {}:\n\t{}".format(
                        model.__class__.__name__, "\n\t".join(error_msgs)
                    )
                )
        # make sure token embedding weights are still tied if needed
        model.tie_weights()

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()

        if output_loading_info:
            loading_info = {
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "error_msgs": error_msgs,
            }
            return model, loading_info

        if hasattr(config, "xla_device") and config.xla_device and is_torch_tpu_available():
            import torch_xla.core.xla_model as xm

            model = xm.send_cpu_data_to_device(model, xm.xla_device())
            model.to(xm.xla_device())

        return model
   
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels = None,
        prototype_p =None,
        prototype_f =None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        cf_ratio = 1.0,
        train_mode = "mlm"
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
            
            -->> Written for DeBugging only, will change later <<-- 
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        
        ###############################################################################
        ##############################################################################
        
        
        sequence_output = outputs[0]
        if train_mode == "mlm":
            prediction_scores = self.lm_head(sequence_output)

            masked_lm_loss = None
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

            if not return_dict:
                output = (prediction_scores,) + outputs[2:]
                return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

            return MaskedLMOutput(
                loss=masked_lm_loss,
                logits=prediction_scores,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        elif train_mode == "fine_tune":
            logits_t = self.classifier_t(sequence_output) #logits of the target classifier
            logits_s2t = self.classifier_s2t(sequence_output) #logits of s2t classifier for maintaining sanity

            loss = None
                        
            if labels is not None:
                if self.num_labels == 1:
                    #  I fWe are doing regression
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), s_labels.view(-1))
                else:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits_s2t.view(-1, self.num_labels), labels.view(-1)) + loss_fct(logits_t.view(-1, self.num_labels), labels.view(-1))
        else:
            logits_t = self.classifier_t(sequence_output) #logits of the target classifier
            logits_s2t = self.classifier_s2t(sequence_output) #logits of s2t classifier for maintaining sanity

            loss = None
                        
            if prototype_p is not None: 
                t_labels,conf_mask = compute_plabel_and_conf(prototype_p,prototype_f,outputs[0][:,0,:],cf_ratio)
                if labels is not None:
                    if self.num_labels == 1:
                        #  I fWe are doing regression
                        loss_fct = MSELoss()
                        s2t_loss = loss_fct(logits.view(-1), s_labels.view(-1))
                    else:
                        loss_fct = CrossEntropyLoss()
                        s2t_loss = loss_fct(logits_s2t.view(-1, self.num_labels), labels.view(-1))
                if t_labels is not None:
    #                 if conf_mask is None:
                    if self.num_labels == 1:
                        # If  We are doing regression
                        loss_fct = MSELoss()
                        t_loss = loss_fct(logits_t.view(-1), t_labels.view(-1))
                    else:
                        loss_fct = CrossEntropyLoss(reduction = 'none')
                        t_loss = loss_fct(logits_s2t.view(-1, self.num_labels), t_labels.view(-1))
                        t_loss =  torch.mean(t_loss*conf_mask,dim =0, keepdim = False) ## Consider only High Confidence points
                    loss = [s2t_loss,t_loss] # returs both s2tloss and t_loss
            else:
                if labels is not None:
                    if self.num_labels == 1:
                        #  We are doing regression
                        loss_fct = MSELoss()
                        t_loss = loss_fct(logits.view(-1), s_labels.view(-1))
                    else:
                        loss_fct = CrossEntropyLoss()
                        t_loss = loss_fct(logits_t.view(-1, self.num_labels), labels.view(-1))
                    loss = t_loss
      
        if return_dict:
            return sfdaNegationClassifierOutput(
                loss=loss,
                logits=logits_t,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                last_hidden_state = outputs[0]
            )
        else:
            return sfdaNegationClassifierOutput(
                loss=loss,
                logits=logits_t,
                hidden_states=None,
                attentions=outputs.attentions,
                last_hidden_state = outputs[0]
            )
    
    ########################################################################################
    ########################################################################################
