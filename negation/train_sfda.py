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
    DataCollatorForLanguageModeling,
)
from typing import Callable, Dict, Optional, List, Union
logger = logging.getLogger(__name__)
from transformers.data.metrics import acc_and_f1

from enum import Enum
from sfda.models import sfdaTargetRobertaNegation
from sfda.trainer import sfdaTrainer
from sfda.DataProcessor import sfdaNegationDataset
from sfda.DataProcessor import NegationDataset
from datasets import load_dataset
import shutil


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
        
        
@dataclass
class sfdaTrainingArguments:
    APM_Strategy: str = field(
        default="top_k", metadata={"help": "APM update strategy, use top_k for updating APM with top_k from each label and thresh for specifying it with a threshold score."}
    )
    top_k: int = field(
        default=100, metadata={"help": "[For top_k APM update strategy], the number of prototypes extracted for each label"}
    )
    cf_ratio: float = field(
        default=1.0, metadata={"help": "The minimum ratio of min similarity  of  the closest class to the max similarity point of the farthest class to be eligible for consideration as High Confidence point"}
    )
    update_freq: int = field(
        default = 100,
        metadata={"help": "The number of global steps after which  APM prototypes are updated "}
    )
    alpha_routine: str = field(
        default="exp", metadata={"help": "The alpha update startegy. Choose from \"exp\" : Exponential routine, \"sqr\" : Square routine , \"lin\": Linear routine,, \"cube\": Cube routine "}
    )
    do_mlm: bool = field(
        default=False, metadata={"help": "Choose if you want to perform MLM pretraining"}
    )
    do_fine_tune: bool = field(
        default=False, metadata={"help": "Choose if you want to perform A finetuning on targets first pretraining"}
    )
    mlm_lr: float = field(
        default=5e-6, metadata={"help": "Specify learning rate for MLM training"}
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, sfdaTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, sfda_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, sfda_args = parser.parse_args_into_dataclasses()

    
    
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
    logger.info("sfda_args %s", sfda_args)
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

    ######  ----->      MLM Pretraining      <-----  ######
    if (sfda_args.do_mlm):
        MLM_path = os.path.join(training_args.output_dir,F"MLM{sfda_args.mlm_lr}")
        training_args.output_dir = MLM_path

        if (
            os.path.exists(MLM_path)
            and os.listdir(MLM_path)
        ):
            
            if not training_args.overwrite_output_dir :
                logger.warning(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overwrite, loading model-weights from the directory for now."
                )
                model = sfdaTargetRobertaNegation.from_pretrained(
                    MLM_path,
                    from_tf=bool(".ckpt" in model_args.src_model_name_or_pth),
                    config=config,
                    cache_dir=model_args.cache_dir,
                )
            else:
                shutil.rmtree(MLM_path)
                logger.warning(F"Removed {MLM_path}")
                dataset = load_dataset("text", data_files=  data_args.train_file)
                def tokenize_function(examples):
                    return tokenizer(examples["text"], return_special_tokens_mask=True)
                tokenized_dataset = dataset.map(
                        tokenize_function,
                        batched=True,
                        num_proc=None,
                    )
                data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.3)
                trainer_mlm = Trainer(
                    model=model,
                    args=TrainingArguments(output_dir = MLM_path ,learning_rate = sfda_args.mlm_lr),
                    compute_metrics=build_compute_metrics_fn(),
                    train_dataset = tokenized_dataset["train"],
                    data_collator = data_collator
                )
                logger.info("Performing MLM pretraining")
                trainer_mlm.train()
                trainer_mlm.save_model()            
        else:
            dataset = load_dataset("text", data_files=  data_args.train_file)
            def tokenize_function(examples):
                return tokenizer(examples["text"], return_special_tokens_mask=True)
            tokenized_dataset = dataset.map(
                    tokenize_function,
                    batched=True,
                    num_proc=None,
                )
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.3)
            trainer_mlm = Trainer(
                model=model,
                args=TrainingArguments(output_dir = MLM_path ,learning_rate = sfda_args.mlm_lr),
                compute_metrics=build_compute_metrics_fn(),
                train_dataset = tokenized_dataset["train"],
                data_collator = data_collator
            )
            logger.info("Performing MLM pretraining")
            trainer_mlm.train()
            trainer_mlm.save_model()            
    
        
    ######  <------------------------------------->  ######
    ######  ------>       Fine-Tuning       <------  ######
    
    if sfda_args.do_fine_tune:
        train_dataset = sfdaNegationDataset.from_tsv(data_args.train_file, data_args.train_pred,tokenizer)
        eval_dataset = sfdaNegationDataset.from_tsv(data_args.eval_file, data_args.eval_pred,tokenizer)
        trainer = sfdaTrainer(
            model=model,
            args=training_args,
            sfda_args = sfda_args,
            compute_metrics=build_compute_metrics_fn(),
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            train_mode = "fine_tune"
        )
        trainer.train()
        trainer.save_model()
    else:
        ######  <------------------------------------->  ######

        ######  ------>      SFDA Training      <------  ######
    
        training_args.output_dir = os.path.join(training_args.output_dir,F"top-{sfda_args.top_k}-cf_ratio{sfda_args.cf_ratio}/")
        save_path = os.path.join(training_args.output_dir,F"dev_pred_sfda.tsv")
        logger.info(save_path)
        train_dataset = sfdaNegationDataset.from_tsv(data_args.train_file, data_args.train_pred,tokenizer)
        eval_dataset = sfdaNegationDataset.from_tsv(data_args.eval_file, data_args.eval_pred,tokenizer)
            
        trainer = sfdaTrainer(
            model=model,
            args=training_args,
            sfda_args = sfda_args,
            compute_metrics=build_compute_metrics_fn(),
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
        )
        
        if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
        ):

            if not training_args.overwrite_output_dir:
                logger.warning(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. Attempting to load pre-trained weights from dir. Use --overwrite_output_dir to overwrite."
                )
                model = sfdaTargetRobertaNegation.from_pretrained(
                    training_args.output_dir,
                    config=config,
                )
                trainer = sfdaTrainer(
                  model=model,
                  args=training_args,
                  sfda_args = sfda_args,
                  compute_metrics=build_compute_metrics_fn(),
                  train_dataset = train_dataset,
                  eval_dataset = eval_dataset,
              )
              
            else:
                shutil.rmtree(training_args.output_dir)
                trainer.train()
                trainer.save_model()  
        else:
            trainer.train()
            trainer.save_model()
    
    
        ######  <------------------------------------->  ######
    

    ######  ------>        Evaluation       <------  ######
    
    
    eval_result = trainer.evaluate(eval_dataset)
    output_eval_file = os.path.join(
        training_args.output_dir, f"eval_results.txt"
    )
    if trainer.is_world_process_zero():
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key, value in eval_result.items():
                logger.info("  %s = %s", key, value)
                writer.write("%s = %s\n" % (key, value))
    predictions = trainer.predict(eval_dataset).predictions
    predictions = np.argmax(predictions, axis=1)
    with open(save_path, "w") as writer:
        logger.info("***** Test results *****")
        for index, item in enumerate(predictions):
            item = train_dataset.get_labels()[item]
            writer.write("%s\n" % (item))
    
    ######  <------------------------------------->  ######
    

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
    

