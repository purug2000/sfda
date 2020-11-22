from torch.utils.data.dataset import Dataset
from transformers.data.processors.utils import InputExample, InputFeatures
from transformers.data.processors.glue import glue_convert_examples_to_features
from transformers.data.processors.utils import DataProcessor
import logging
labels = ["-1", "1"]
max_length = 128
logger = logging.getLogger(__name__)

class sfdaNegationDataset(Dataset):
    def __init__(self, features):
        self.features = features
        self.label_list = ["-1", "1"]
    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def get_labels(self):
        return self.label_list

    @classmethod
    def from_tsv(cls, tsv_file, pseudo_tsv, tokenizer):
        """Creates examples for the test set."""
        rev_label_list = {"-1":0, "1":1}
        lines = DataProcessor._read_tsv(tsv_file)
        lab = DataProcessor._read_tsv(pseudo_tsv)
        examples = []
        for (i, line) in enumerate(lines):
            guid = 'instance-%d' % i
            if line[0] in labels:
                text_a = '\t'.join(line[1:])
            else:
                text_a = '\t'.join(line)

            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=lab[i][0]))

        features = glue_convert_examples_to_features(
            examples,
            tokenizer,
            max_length=max_length,
            label_list=labels,
            output_mode='classification',
        )
        return cls(features)
    

class NegationDataset(Dataset):
    def __init__(self, features):
        self.features = features
        self.label_list = ["-1", "1"]
    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def get_labels(self):
        return self.label_list

    @classmethod
    def from_tsv(cls, tsv_file, tokenizer):
        """Creates examples for the test set."""
        rev_label_list = {"-1":0, "1":1}
        lines = DataProcessor._read_tsv(tsv_file)
#         lab = DataProcessor._read_tsv(pseudo_tsv)
        examples = []
        for (i, line) in enumerate(lines):
            guid = 'instance-%d' % i
            if line[0] in labels:
                label = line[0]
                text_a = '\t'.join(line[1:])
            else:
                text_a = '\t'.join(line)
                label = None

            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

        features = glue_convert_examples_to_features(
            examples,
            tokenizer,
            max_length=max_length,
            label_list=labels,
            output_mode='classification',
        )
        return cls(features)
    
class MLMDataset(Dataset):
    def __init__(self, features):
        self.features = features
        self.label_list = ["-1", "1"]
    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def get_labels(self):
        return self.label_list

    @classmethod
    def from_tsv(cls, tsv_file, tokenizer):
        """Creates examples for the test set."""
        rev_label_list = {"-1":0, "1":1}
        lines = DataProcessor._read_tsv(tsv_file)
#         lab = DataProcessor._read_tsv(pseudo_tsv)
        examples = []
        for (i, line) in enumerate(lines):
            guid = 'instance-%d' % i
            if line[0] in labels:
                label = line[0]
                text_a = '\t'.join(line[1:])
            else:
                text_a = '\t'.join(line)
                label = None

            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

        features = glue_convert_examples_to_features(
            examples,
            tokenizer,
            max_length=max_length,
            label_list=labels,
            output_mode='classification',
        )
        return cls(features)
