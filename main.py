
import argparse
import logging

import torch
from torch.utils.data import DataLoader, RandomSampler, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from transformers import (AdamW, BertForSequenceClassification,
                          WarmupLinearSchedule)

from utils import get_dataset

parse = argparse.ArgumentParser()
parse.add_argument('--mode', default='train', help='以何种形式运行')
parse.add_argument('--data_path', type=str,
                   default='/media/yzq/HDD/Dataset/baifendian_data/',
                   help='数据集文件夹')

parse.add_argument('--max_length', default=128, help='句子最大长度')
parse.add_argument('--train_with_eval', default=True,
                   type=bool, help='是否一边训练一边验证')
parse.add_argument('--batch_size', default=8, type=int,
                   help='每次训练数据个数')
parse.add_argument('--epoch', default=10, help='Epoch')
parse.add_argument('--logging_step', default=50, help='打印间隔')
parse.add_argument('--eval_step', default=200)


args = parse.parse_args()
logger = logging.getLogger()
device = 'cuda' if torch.cuda.is_available else 'cpu'


def train(model, dataset):
    tb_writer = SummaryWriter()

    split_num = int(0.7*len(dataset))
    train_set, eval_set = random_split(dataset,
                                       [split_num,
                                        len(dataset)-split_num])

    train_sampler = RandomSampler(train_set)
    train_dataloader = DataLoader(train_set, sampler=train_sampler,
                                  batch_size=8)

    # 设置优化器和学习率衰减 (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5, eps=1e-8)
    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=0, t_total=len(dataset))

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0

    model.zero_grad()
    train_iterator = trange(args.epoch, desc="Epoch")
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for _, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],  # XLM don't use segment_ids
                      'labels':         batch[3]}
            outputs = model(**inputs)
            loss = outputs[0]

            tr_loss += loss.item()
            global_step += 1

            if global_step % args.logging_step == 0:

                tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar(
                    'loss',
                    (tr_loss - logging_loss)/args.logging_step,
                    global_step)

                logging_loss = tr_loss

            if global_step % args.eval_step == 0 and args.train_with_eval:
                pred, labels = predict(model, eval_set, eval=True)
                print(sum(pred == labels)*1.0/len(eval_set))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()  # 更新学习速率
            model.zero_grad()


def predict(model, dataset, eval=False):
    dataloader = DataLoader(dataset, batch_size=8)
    pred = torch.tensor([]).long()
    labels = torch.tensor([]).long()

    eval_iterator = tqdm(dataloader, desc="Eval")
    for _, batch in enumerate(eval_iterator):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2]}
            outputs = model(**inputs)
            logits = outputs[0]

            pred = torch.cat([pred, torch.argmax(logits, axis=1).cpu()])
            if eval:
                labels = torch.cat([labels, batch[3].cpu().squeeze()])

    return (pred, labels) if eval else (pred, )


def main():
    dataset = get_dataset(args)

    # 将两个句子拼接在一起然后使用句子分类模型
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-chinese').to('cuda')
    train(model, dataset)


if __name__ == '__main__':
    main()
