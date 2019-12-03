""" Using Texar to Implement the Sequence task
"""
import functools
import logging

import torch
import torch.nn.functional as F

import data_config
import texar.torch as tx
from dataset import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


def _compute_loss(logits, labels, num_classes=1):
    r"""Compute loss.
    """
    if num_classes == 1:
        loss = F.binary_cross_entropy(
            logits.view(-1), labels.view(-1), reduction='mean')
    else:
        loss = F.cross_entropy(
            logits.view(-1, num_classes),
            labels.view(-1), reduction='mean')
    return loss


def _train(model, dataset):
    """ Do Train
    """
    # 计算训练步数
    num_train_steps = int(dataset.batch_size / dataset.batch_size *
                          dataset.num_epochs)
    num_warmup_steps = int(num_train_steps *
                           data_config.warmup_proportion)

    static_lr = 2e-5
    # 设置优化器
    vars_with_decay = []
    vars_without_decay = []
    for name, param in model.named_parameters():
        # 对于 layer_norm 和 bias 不需要学习速率衰减
        if 'layer_norm' in name or name.endswith('bias'):
            vars_without_decay.append(param)
        else:
            vars_with_decay.append(param)

    # 优化器参数
    opt_params = [{
        # 对于需要衰减的参数
        'params': vars_with_decay,
        'weight_decay': 0.01,
    }, {
        'params': vars_without_decay,
        'weight_decay': 0.0,
    }]
    # BERT adam 优化器
    optim = tx.core.BertAdam(
        opt_params, betas=(0.9, 0.999), eps=1e-6, lr=static_lr)

    # 计时器用于调整学习速率
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim, functools.partial(data_config.get_lr_multiplier,
                                 total_steps=num_train_steps,
                                 warmup_steps=num_warmup_steps))

    dataiteor = tx.data.DataIterator(dataset)

    model.train()
    for batch in dataiteor:
        logits, preds = model(inputs=batch['inputs'],
                              sequence_length=batch['sequence_length'],
                              segment_ids=batch['segment_ids'])
        logits = F.sigmoid(logits)
        loss = _compute_loss(logits, batch['labels'])
        loss.backward()
        optim.step()
        scheduler.step()
        step = scheduler.last_epoch
        dis_steps = data_config.display_steps
        if dis_steps > 0 and step % dis_steps == 0:
            logging.info("step: %d; loss: %f", step, loss)


@torch.no_grad
def _test(model):
    pass


@torch.no_grad
def _valid(mode, dataset: Dataset):
    # 用于 eval 或者 valid
    pass


def main():
    dataset = Dataset(data_path=data_config.train_path,
                      hparams=data_config.train_hparams,
                      device=device)
    model = tx.modules.BERTClassifier('bert-base-chinese',
                                      hparams=data_config.model_hparams)
    model.to(device)

    _train(model, dataset)


if __name__ == '__main__':
    main()
