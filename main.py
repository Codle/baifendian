
import argparse
import logging
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from transformers import (AdamW, BertForSequenceClassification,
                          WarmupLinearSchedule)

from utils import get_dataset

parse = argparse.ArgumentParser()
parse.add_argument('--mode', default='train', type=str, help='以何种形式运行')
parse.add_argument('--data_path', default='./data', type=str, help='数据集文件夹')
parse.add_argument('--output_dir', default='./output',
                   type=str, help='模型保存文件夹')
parse.add_argument('--best_step', default=0, type=int, help='最佳迭代次数')
parse.add_argument('--max_length', default=128, help='句子最大长度')
parse.add_argument('--train_with_eval', default=True,
                   type=bool, help='是否一边训练一边验证')
parse.add_argument('--batch_size', default=8, type=int, help='每次训练数据个数')
parse.add_argument('--num_train_epoch', default=10, type=int, help='Epoch')
parse.add_argument('--logging_step', default=50, type=int, help='打印间隔')
parse.add_argument('--eval_step', default=2000, type=int, help='验证间隔')
parse.add_argument('--save_step', default=2000, type=int, help='保存间隔')
parse.add_argument('--eval_precent', default=0.8, type=float, help='验证比例')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = 'cuda' if torch.cuda.is_available else 'cpu'


def train(model, dataset, args):
    tb_writer = SummaryWriter()

    train_num = int(args.eval_precent*len(dataset))
    eval_num = len(dataset) - train_num
    train_set, eval_set = random_split(dataset, [train_num, eval_num])

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
    best_step, best_acc = 0, 0.0

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epoch}")
    logger.info(f"  Train batch size  = {args.batch_size}")
    logger.info(f"  Total optimization steps = {len(dataset)}")

    model.zero_grad()
    train_iterator = trange(args.num_train_epoch, desc="Epoch")
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for _, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels':         batch[3]}
            outputs = model(**inputs)
            loss = outputs[0]

            # 绘图
            tr_loss += loss.item()
            global_step += 1
            if global_step % args.logging_step == 0:

                tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar(
                    'loss',
                    (tr_loss - logging_loss)/args.logging_step,
                    global_step)

                logging_loss = tr_loss

            # 保存模型
            if global_step % args.save_step == 0:
                output_dir = os.path.join(
                    args.output_dir, 'checkpoint-{}'.format(global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model.save_pretrained(output_dir)
                # tokenizer.save_vocabulary(output_dir)
                torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                logger.info("Saving model checkpoint to %s", output_dir)

            # 验证
            if global_step % args.eval_step == 0 and args.train_with_eval:
                pred, labels = predict(model, eval_set, args, eval=True)
                acc = sum(pred == labels)*1.0/len(eval_set)
                tb_writer.add_scalar('acc', float(acc), global_step)
                if acc > best_acc:
                    best_acc = acc
                    best_step = global_step
                logger.info(f"Global step is {global_step},"
                            f"acc is {float(acc):.4f},")
                logger.info(f"Best step is {best_step},"
                            f"best acc is {float(best_acc):.4f}")

            # 反向传播、优化器优化等常规步骤
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()  # 更新学习速率
            model.zero_grad()


def predict(model, dataset, args, eval=False):
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
    args = parse.parse_args()
    dataset = get_dataset(args)

    if args.mode == 'train':
        # 将两个句子拼接在一起然后使用句子分类模型
        model = BertForSequenceClassification.from_pretrained(
            'bert-base-chinese').to(device)
        train(model, dataset, args)
    else:
        # args = torch.load(
        #     os.path.join(args.output_dir,
        #                  f'checkpoint-{args.best_step}/training_args.bin'))
        model = BertForSequenceClassification.from_pretrained(
            os.path.join(args.output_dir, f'checkpoint-{args.best_step}')
        ).to(device)
        pred = predict(model, dataset, args)[0]
        pred = pd.Series(pred.numpy().tolist())
        res_csv = pd.concat([dataset.df['qid'], pred], axis=1)
        res_csv.to_csv(os.path.join(args.output_dir, 'result.csv'),
                       header=False, index=False, sep='\t')


if __name__ == '__main__':
    main()
