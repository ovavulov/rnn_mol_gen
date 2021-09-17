import os
import random

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
import tqdm
import warnings
import yaml

SEED = 19
random.seed(SEED)
np.random.seed(SEED)
torch.random.manual_seed(SEED)
warnings.filterwarnings('ignore')

DATA_PATH = './data/preprocessed'
with open('conf.yml') as f:
    cfg = yaml.load(f.read())

print(cfg['data']['train_sample'])