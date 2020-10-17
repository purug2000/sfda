import dataclasses
import logging

import os
import argparse
from os.path import basename, dirname
import sys
from dataclasses import dataclass, field
import numpy as np
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
    HfArgumentParser,
    EvalPrediction,
)
from typing import Callable, Dict, Optional, List, Union
logger = logging.getLogger(__name__)
from transformers.data.metrics import acc_and_f1

from enum import Enum
from sfda.models import sfdaTargetRobertaNegation
from sfda.trainer import sfdaTrainer
from sfda.DataProcessor import sfdaNegationDataset
from sfda.DataProcessor import NegationDataset


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    src_model_name_or_pth: str = field(
       default= "tmills/roberta_sfda_sharpseed", metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_save_path : str = field(
        default=None, metadata={"help": "Save path for model"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
        

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    # Only allowed task is Negation, don't need this field from Glue
    #task_name: str = field(metadata={"help": "The name of the task to train on: " + ", ".join(glue_processors.keys())})
    train_file: str = field(
        metadata={"help": "The input train file"}
    )
    train_pred: str = field(
        metadata={"help": "A file containing the generated pseudo labels for the train file "}
    )
    eval_file:str = field(
        metadata={"help": "A file to evaluate on."}
    )
    eval_pred:str = field(
        metadata={"help": "A file to evaluate on."}
    )
    update_freq: int = field(
        default = 100,
        metadata={"help": "The number of global steps after which  APM prototypes are updated "}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

def build_compute_metrics_fn() -> Callable[[EvalPrediction], Dict]:
    def compute_metrics_fn(p: EvalPrediction):
        preds = np.argmax(p.predictions, axis=1)
        return acc_and_f1(preds, p.label_ids)

    return compute_metrics_fn
        
        
def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    set_seed(training_args.seed)

    try:
        num_labels = 2
        output_mode = 'classification'
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))
    config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.src_model_name_or_pth,
            num_labels=num_labels,
            finetuning_task='negation',
            cache_dir=model_args.cache_dir,
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.src_model_name_or_pth,
        cache_dir=model_args.cache_dir
    )
    model = sfdaTargetRobertaNegation.from_pretrained_source(
        model_args.src_model_name_or_pth,
        from_tf=bool(".ckpt" in model_args.src_model_name_or_pth),
        config=config,
        cache_dir=model_args.cache_dir,
        )
    train_dataset = sfdaNegationDataset.from_tsv(data_args.train_file, data_args.train_pred,tokenizer)
    eval_dataset = sfdaNegationDataset.from_tsv(data_args.eval_file, data_args.eval_pred,tokenizer)
    trainer = sfdaTrainer(
        model=model,
        args=training_args,
        update_freq = data_args.update_freq,
        compute_metrics=build_compute_metrics_fn(),
        train_dataset = train_dataset,
        eval_dataset = eval_dataset
    )
    trainer.train(model_path=model_args.src_model_name_or_pth if os.path.isdir(model_args.src_model_name_or_pth) else None
        )
    eval_result = trainer.evaluate()
    output_eval_file = os.path.join(
        training_args.output_dir, f"eval_results.txt"
    )
    if trainer.is_world_process_zero():
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key, value in eval_result.items():
                logger.info("  %s = %s", key, value)
                writer.write("%s = %s\n" % (key, value))
    trainer.save_model()
    
def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

    