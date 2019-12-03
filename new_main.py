import yaml
import torch
from dataset import Dataset


with open('config/default.yml') as fin:
    config = yaml.load(fin)

print(config['hidden_size'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_config = config['dataset']['train']

dataset = Dataset(data_path=data_config['data_path'],
                  hparams=data_config['haprams'],
                  device=device)

print(len(dataset))
