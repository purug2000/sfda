import os
import re
import anafora
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
from typing import List, Optional, Union
from dataclasses import dataclass
from spacy.lang.en import English



# New parent class 
@dataclass(frozen=True)
class sfdaInputFeatures:
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None
    label_p: Optional[Union[int, float]] = None


# New class for dataloader
class sfdaTimexInputFeatures(sfdaInputFeatures):

    def __init__(self, input_ids, attention_mask, offset_mapping, label, label_p):
        super().__init__(input_ids=input_ids, attention_mask=attention_mask, label=label, label_p = label_p)
        self.offset_mapping = offset_mapping

    @classmethod
    def from_sentence(cls, input_data, sent_idx, sent_offset, annotations, annotations_p, config):
        input_ids = input_data["input_ids"][sent_idx]
        attention_mask = input_data["attention_mask"][sent_idx]
        offset_mapping = input_data["offset_mapping"][sent_idx]
        labels = input_data["labels"][sent_idx]
        labels_p = input_data["labels"][sent_idx]

        start_open = None
        start_open_p  = None
        for token_idx, offset in enumerate(offset_mapping):
            start, end = offset.numpy()
            if start == end:
                continue
            start += sent_offset
            end += sent_offset
            offset_mapping[token_idx][0] = start
            offset_mapping[token_idx][1] = end

            # The annotation my have trailing spaces. Check if the current token is included in the span.
            if start_open is not None and annotations[start_open][0] <= start:
                start_open = None
            # If it is a new annotation and there is still one opened, close it
            # Otherwise, add the token to the opened annotation or open a new one
            if start_open is not None and start in annotations:
                start_open = None
            elif start_open is not None:
                annotation = annotations[start_open][1]
                labels[token_idx] = config.label2id["I-" + annotation]
            elif start in annotations:
                annotation = annotations[start][1]
                labels[token_idx] = config.label2id["B-" + annotation]
                start_open = start
            # Check if the annotation ends in this token and close it
            if start_open is not None and end == annotations[start_open][0]:
                start_open = None

            # The annotation my have trailing spaces. Check if the current token is included in the span.
            if start_open_p is not None and annotations_p[start_open_p][0] <= start:
                start_open_p = None
            # If it is a new annotation and there is still one opened, close it
            # Otherwise, add the token to the opened annotation or open a new one
            if start_open_p is not None and start in annotations_p:
                start_open_p = None
            elif start_open_p is not None:
                annotation_p = annotations_p[start_open_p][1]
                labels_p[token_idx] = config.label2id["I-" + annotation_p]
            elif start in annotations_p:
                annotation_p = annotations_p[start][1]
                labels_p[token_idx] = config.label2id["B-" + annotation_p]
                start_open_p = start
            # Check if the annotation ends in this token and close it
            if start_open_p is not None and end == annotations_p[start_open_p][0]:
                start_open_p = None
        return cls(
            input_ids,
            attention_mask,
            offset_mapping,
            labels,
            labels_p # New field added - for predicted labels
        )



# New class for dataset
class sfdaTimexDataset(Dataset):

    def __init__(self, doc_indices, features):
        self.doc_indices = doc_indices
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]

    @classmethod
    def from_texts(cls, data_dir, anno_dir, anno_dir_p, nlp, tokenizer, config):
        if not os.path.exists(data_dir):
            raise Exception("The %s directory does not exist." % data_dir)
        text_directory_files = anafora.walk(data_dir, xml_name_regex=".*((?<![.].{3})|[.]txt)$")
        features = []
        doc_indices = []
        for text_files in text_directory_files:
            doc_index = len(features)
            text_subdir_path, text_doc_name, text_file_names = text_files
            if len(text_file_names) != 1:
                raise Exception("Wrong number of text files in %s" % text_subdir_path)

            text_file_path = os.path.join(data_dir, text_subdir_path, text_file_names[0])


            # load  ground truth
            anafora_path = os.path.join(anno_dir, text_subdir_path)
            anafora_directory_files = anafora.walk(anafora_path, xml_name_regex="[.]xml$")
            anafora_directory_files = list(anafora_directory_files)
            if len(anafora_directory_files) != 1:
                raise Exception("Wrong structure in %s" % anafora_path)
            anafora_subdir_path, anafora_doc_name, anafora_file_names = anafora_directory_files[0]
            if len(anafora_file_names) != 1:
                raise Exception("Wrong number of anafora files in %s" % anafora_subdir_path)
            # Load the annotations from ground truth
            anafora_file_path = os.path.join(anafora_path, anafora_subdir_path, anafora_file_names[0])
            data = anafora.AnaforaData.from_file(anafora_file_path)
            annotations = dict()
            for annotation in data.annotations:
                label = annotation.type
                for span in annotation.spans:
                    start, end = span
                    annotations[start] = (end, label)


           # load  predictions
            anafora_path_p = os.path.join(anno_dir_p, text_subdir_path)
            anafora_directory_files_p = anafora.walk(anafora_path_p, xml_name_regex="[.]xml$")
            anafora_directory_files_p = list(anafora_directory_files_p)
            if len(anafora_directory_files_p) != 1:
                raise Exception("Wrong structure in %s" % anafora_path_p)
            anafora_subdir_path_p, anafora_doc_name_p, anafora_file_names_p = anafora_directory_files_p[0]
            if len(anafora_file_names_p) != 1:
                raise Exception("Wrong number of anafora files in %s" % anafora_subdir_path_p)
            # Load the annotations from predictions
            anafora_file_path_p = os.path.join(anafora_path_p, anafora_subdir_path_p, anafora_file_names_p[0])
            data_p = anafora.AnaforaData.from_file(anafora_file_path_p)
            annotations_p = dict()
            for annotation in data_p.annotations:
                label = annotation.type
                for span in annotation.spans:
                    start, end = span
                    annotations_p[start] = (end, label)

            # Read, segment and tokenize the raw text.
            with open(text_file_path) as txt_file:
                text = txt_file.read()
            doc = nlp(text)
            input_raw = [sent.text_with_ws for sent in doc.sents]
            input_data = tokenizer(input_raw,
                                   return_tensors="pt",
                                   padding="max_length",
                                   truncation="longest_first",
                                   return_offsets_mapping=True)

            # Initialize label sequence with 0. Use ignore index for padding tokens
            negative_attention_mask = (~input_data["attention_mask"].byte()).true_divide(255).long()
            input_data["labels"] = negative_attention_mask.mul(config.label_pad_id)
            # Assign label_pad to </s> token
            sent_indices = torch.arange(input_data["labels"].shape[0])
            last_non_padded = [sent_indices, input_data["labels"].argmax(dim=1)]
            input_data["labels"][last_non_padded] = config.label_pad_id
            # Assign label_pad to <s> token
            input_data["labels"][:, 0] = config.label_pad_id

            sent_offset = 0
            for sent_idx, _ in enumerate(input_data["input_ids"]):
                features.append(sfdaTimexInputFeatures.from_sentence(
                    input_data,
                    sent_idx,
                    sent_offset,
                    annotations,
                    annotations_p, # New field for predicted annotations
                    config))
                sent_offset += len(input_raw[sent_idx])

            doc_indices.append((text_subdir_path, doc_index, len(features)))
        return cls(doc_indices, features)
