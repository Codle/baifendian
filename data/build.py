from typing import Dict, Iterator, Tuple

import torch

import texar.torch as tx


class DataSource(tx.data.DataSource):

    def __init__(self, df, mode='test'):
        data = []
        for idx, row in df.iterrows():
            data.append({
                'text_a': row['question1'],
                'text_b': row['question2'],
                'label': row['label'] if mode != 'test' else None
            })
        self.data = data

    def __getitem__(self, item: int) -> Dict:
        return self.data[item]

    def __iter__(self) -> Iterator:
        return self.data.__iter__()

    def __len__(self) -> int:
        return len(self.data)


class Dataset(tx.data.DatasetBase):

    def __init__(self, df, hparams, device=None):
        source = DataSource(df, hparams['name'])
        self.tokenizer = tx.data.BERTTokenizer('bert-base-chinese')
        super().__init__(source, hparams['hparams'], device)

    def process(self, raw_example) -> Tuple:
        """ 处理单条数据
        """
        encode_text = self.tokenizer.encode_text(text_a=raw_example['text_a'],
                                                 text_b=raw_example['text_a'],
                                                 max_seq_length=64)
        length = 64
        for idx, i in enumerate(encode_text[2]):
            if i == 0:
                length = idx+1
                break
        return {
            'input_ids': encode_text[0],
            'segment_ids': encode_text[1],
            'input_mask': encode_text[2],
            'lengths': length,
            'labels': raw_example['label']
        }

    def collate(self, examples):
        """ 处理多条数据
        """
        input_ids = [ex["input_ids"] for ex in examples]
        segment_ids = [ex["segment_ids"] for ex in examples]
        input_mask = [ex["input_mask"] for ex in examples]
        lengths = [ex["lengths"] for ex in examples]
        labels = [ex["labels"] for ex in examples]

        return tx.data.Batch(
            len(examples),
            inputs=torch.tensor(input_ids),
            sequence_length=torch.tensor(lengths),
            segment_ids=torch.tensor(segment_ids),
            input_mask=torch.tensor(input_mask),
            labels=torch.FloatTensor(labels))


def build_dataloader(df, cfg, device=None):
    """ 构建dataloader
    """
    return Dataset(df, cfg, device)
