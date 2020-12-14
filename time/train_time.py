import os
import argparse
import torch
from torch.utils.data import Dataset
from transformers import (
    InputFeatures,
    AutoConfig,
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments
)
from spacy.lang.en import English
import anafora
from sfda.DataProcessor import sfdaTimexDataset
from sfda.models import sfdaRobertaForTokenClassification
from sfda.trainer import sfdaTrainer
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, List, Union
import shutil

@dataclass
class sfdaTrainingArguments:
    APM_Strategy: str = field(
        default="top_k", metadata={"help": "APM update strategy, use top_k for updating APM with top_k from each label and thresh for specifying it with a threshold score."}
    )
    top_k: int = field(
        default=100, metadata={"help": "[For top_k APM update strategy], the number of prototypes extracted for each label"}
    )
    cf_ratio: float = field(
        default=0, metadata={"help": "The minimum ratio of min similarity  of  the closest class to the max similarity point of the farthest class to be eligible for consideration as High Confidence point"}
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
    mlm_lr: float = field(
        default=5e-6, metadata={"help": "Specify learning rate for MLM training"}
    )


def train(data_dir, anno_dir,anno_dir_p, save_dir):
    sfda_args = sfdaTrainingArguments()
    # load the Huggingface config, tokenizer, and model
    model_name = "clulab/roberta-timex-semeval"
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              config=config,
                                              use_fast=True)
    model = sfdaRobertaForTokenClassification.from_pretrained_source(model_name,
                                                            config=config)

    # load the spacy sentence segmenter
    nlp = English()
    nlp.add_pipe(nlp.create_pipe("sentencizer"))
    if (
        os.path.exists(save_dir)
        and os.listdir(save_dir)
    ):
        shutil.rmtree(save_dir)
        print(F"Removed {save_dir}")

    # create a torch dataset from a directory of Anafora XML annotations and text files
    dataset = sfdaTimexDataset.from_texts(data_dir, anno_dir,anno_dir_p, nlp, tokenizer, config)

    # train and save the torch model
    trainer = sfdaTrainer(
        num_labels = config.num_labels,
        sfda_args = sfda_args,
        model=model,
        args=TrainingArguments(save_dir),
        train_dataset=dataset,
        data_collator=lambda features: dict(
            input_ids=torch.stack([f.input_ids for f in features]),
            attention_mask=torch.stack([f.attention_mask for f in features]),
            labels=torch.stack([f.label_p for f in features]))
    )
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="""%(prog)s trains SEMEVAL-2021 temporal baseline.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-t", "--train", metavar="DIR", dest="train_dir",
                        help="The root of the training set directory tree containing raw text.")
    parser.add_argument("-a", "--anno", metavar="DIR", dest="anno_dir",
                        help="The root of the training set directory tree containing Anafora XML.")
    parser.add_argument("-p", "--annop", metavar="DIR", dest="anno_dir_p",
                        help="The root of the Predictions of training set directory tree containing Anafora XML.")
    parser.add_argument("-s", "--save", metavar="DIR", dest="save_dir",
                        help="The directory to save the model and the log files.")
    args = parser.parse_args()
    train(args.train_dir, args.anno_dir, args.anno_dir_p, args.save_dir)
