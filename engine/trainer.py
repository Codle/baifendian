""" Train loop
"""
from typing import Any, Dict, Optional

import torch
import torch.functional as F
import torch.nn as nn
import torch.utils.data as data_utils

import texar.torch as tx
from utils.logging import setup_logger

logger = setup_logger('__train__')


def do_train(model: nn.Module,
             train_loader: data_utils.DataLoader,
             valid_loader: data_utils.DataLoader,
             optimizer,
             loss_fn: nn.Module = None,
             scheduler: Optional = None,
             cfg: Optional = None) -> None:
    """ Do Train Loop

    Args:
        cfg: Dict, config info;
        model: nn.Module, the model you use;
        train_loader: train data loader;
        valid_loader: valid data loader;
        optimizer: optimizer;
        scheduler: use to change learing rate;
        loss_fn: the function to compute the loss;
    """
    model.train()
    dataiterator = tx.data.DataIterator(train_loader)
    # dataiterator.switch_to_train_data()
    logging_loss = 0.0
    global_loss = 0.0

    for batch in dataiterator:
        # 训练数据计算loss
        outputs = model(inputs=batch['inputs'],
                        sequence_length=batch['sequence_length'],
                        segment_ids=batch['segment_ids'],
                        labels=batch['labels'])

        loss = outputs[0]

        global_loss += loss.item()
        # 反向传播
        loss.backward()
        optimizer.step()
        scheduler.step()

        # 打印日志等操作
        step = scheduler.last_epoch
        dis_steps = cfg['display_steps']
        if dis_steps > 0 and step % dis_steps == 0:
            logger.debug(f"step: {step}; loss: {global_loss - logging_loss}")
            logging_loss = global_loss

        eval_steps = cfg['eval_steps']
        if eval_steps > 0 and step % eval_steps == 0:
            # dataiterator.switch_to_val_data()
            do_valid(model, valid_loader)
            # dataiterator.switch_to_train_data()


@torch.no_grad()
def do_valid(model, valid_loader, cfg=None):
    model.eval()
    dataiterator = tx.data.DataIterator(valid_loader)
    cnt = 0
    for batch in dataiterator:
        outputs = model(inputs=batch['inputs'],
                        sequence_length=batch['sequence_length'],
                        segment_ids=batch['segment_ids'])
        logits = outputs[0]
        pred = torch.argmax(logits, axis=1)
        cnt += int(sum(batch['labels'] == pred))
    logger.debug(f"eval_acc: {cnt*1.0/len(valid_loader):.2f}")
