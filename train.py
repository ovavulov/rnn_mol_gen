import os
import random

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import tqdm
import warnings
import yaml

from model import RNNmodel

SEED = 19
random.seed(SEED)
np.random.seed(SEED)
torch.random.manual_seed(SEED)
warnings.filterwarnings('ignore')

DATA_PATH = './data/preprocessed'
with open('conf.yml') as f:
    cfg = yaml.load(f.read())

node2idx = joblib.load(os.path.join(DATA_PATH, 'node2idx.pkl'))
idx2node = joblib.load(os.path.join(DATA_PATH, 'idx2node.pkl'))
# train = torch.Tensor(joblib.load(os.path.join(DATA_PATH, 'train.pkl')))
if cfg['data']['scaffolds']:
    test = torch.Tensor(joblib.load(os.path.join(DATA_PATH, 'test_scaffolds.pkl')))
else:
    test = torch.Tensor(joblib.load(os.path.join(DATA_PATH, 'test.pkl')))

# tr_frac = cfg['data']['train_sample']
# if tr_frac is not None:
#     idxs = np.random.randint(0, len(train), size=(int(len(train) * tr_frac), ))
#     train = train[idxs, :, :]

te_frac = cfg['data']['test_sample']
if te_frac is not None:
    idxs = np.random.randint(0, len(test), size=(int(len(test) * te_frac), ))
    test = test[idxs, :, :]

print(test.shape)

def get_dataloader(data, batch_size):
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


# train_dataloader = get_dataloader(train, batch_size=cfg['model']['batch_size'])
test_dataloader = get_dataloader(test, batch_size=cfg['model']['batch_size'])

model = RNNmodel()

for (seq,) in test_dataloader:
    x, y = seq[:, :-1, :], seq[:, 1:, :]
    print(x.shape, y.shape)
    outputs, (hn, cn) = model(x)
    print(outputs.shape, hn.shape, cn.shape)
    break

