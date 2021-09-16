import os
import random

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
import tqdm
import yaml

SEED = 19
random.seed(SEED)
np.random.seed(SEED)
torch.random.manual_seed(SEED)

DATA_PATH = './data/preprocessed'
with open('conf.yml') as f:
    cfg = yaml.load(f.read())

# node2idx = joblib.load(os.path.join(DATA_PATH, 'node2idx.pkl'))
# idx2node = joblib.load(os.path.join(DATA_PATH, 'idx2node.pkl'))
# train_ohe = joblib.load(os.path.join(DATA_PATH, 'train.pkl'))
# test_ohe = joblib.load(os.path.join(DATA_PATH, 'test.pkl'))



