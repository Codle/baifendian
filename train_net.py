""" Train Pipline
"""
import torch
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


from data import build_dataloader
from engine import trainer
from models.bert import BertClassifier
from solver import build_optimizer, build_scheduler


device = torch.device('cuda')


def main():
    # 读取配置文件
    with open('config/default.yml') as fin:
        config = yaml.load(fin, Loader=yaml.SafeLoader)

    # 生成 train 和 valid 数据集
    train_config = config['dataset']['train']
    train_df = pd.read_csv(train_config['data_path'], sep='\t')
    train_df.sample(frac=1)
    train, valid = train_test_split(
        train_df, test_size=config['train_valid_split'])
    train_dataset = build_dataloader(train, train_config, device=device)
    valid_dataset = build_dataloader(valid, train_config, device=device)

    # 建立模型
    model_config = config['model']
    model = BertClassifier(model_config)
    model.to(device)
    optimizer = build_optimizer(model, config['optimizer'])

    # 计算训练步数
    num_train_steps = int(len(train_dataset) / train_dataset.batch_size *
                          config['num_epochs'])
    num_warmup_steps = int(num_train_steps *
                           config['optimizer']['warmup_proportion'])
    scheduler = build_scheduler(optimizer, num_train_steps, num_warmup_steps)

    # 训练
    trainer.do_train(model,
                     train_loader=train_dataset,
                     valid_loader=valid_dataset,
                     optimizer=optimizer,
                     scheduler=scheduler,
                     cfg=config)


if __name__ == '__main__':
    main()
