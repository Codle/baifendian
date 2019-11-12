"""
This file is used to create PyTorch Dataset
"""

import torch
import pandas as pd
import os
from transformers import BertTokenizer
from torch.utils.data import Dataset


class InputFeature(object):

    def __init__(self, features):
        self.input_ids = features[0]
        self.attention_mask = features[1]
        self.token_type_ids = features[2]
        if len(features) == 4:
            self.label = features[3]
        else:
            self.label = None

    def get_data(self):
        if self.label:
            return (torch.tensor(self.input_ids),
                    torch.tensor(self.attention_mask),
                    torch.tensor(self.token_type_ids),
                    torch.tensor(self.label))
        else:
            return (torch.tensor(self.input_ids),
                    torch.tensor(self.attention_mask),
                    torch.tensor(self.token_type_ids))


class BaiFengdianDataset(Dataset):
    def __init__(self, data, df=None):
        self.data = data
        self.df = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):

        feature = self.data[item]
        return feature.get_data()


def get_dataset(args):
    if args.mode == 'train':
        df = pd.read_csv(os.path.join(
            args.data_path, 'train_set.csv'), sep='\t')
    elif args.mode == 'dev':
        df = pd.read_csv(os.path.join(args.data_path, 'dev_set.csv'), sep='\t')
    else:
        raise KeyError("mode only can be train or dev or test")

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    data = []
    max_length = args.max_length

    for i in range(len(df)):
        question1 = df['question1'][i]
        question2 = df['question2'][i]

        inputs = tokenizer.encode_plus(
            question1,
            question2,
            add_special_tokens=True,
            max_length=max_length,
        )

        input_ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]

        # tokens are attended to.
        attention_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)

        input_ids = input_ids + ([0] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        if args.mode == 'train':
            label = [1] if df['label'][i] else [0]
            data.append(InputFeature(
                (input_ids, attention_mask, token_type_ids, label)))
        else:
            data.append(InputFeature(
                (input_ids, attention_mask, token_type_ids)))

    return BaiFengdianDataset(data, df)
