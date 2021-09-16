import os

import joblib
import numpy as np

OUTPUT_DATA_PATH = './data/preprocessed'

node2idx = joblib.load(os.path.join(OUTPUT_DATA_PATH, 'node2idx.pkl'))
idx2node = joblib.load(os.path.join(OUTPUT_DATA_PATH, 'idx2node.pkl'))
train_ohe = joblib.load(os.path.join(OUTPUT_DATA_PATH, 'train.pkl'))
test_ohe = joblib.load(os.path.join(OUTPUT_DATA_PATH, 'test.pkl'))
test_sc_ohe = joblib.load(os.path.join(OUTPUT_DATA_PATH, 'test_sc.pkl'))

print(node2idx)
print(train_ohe[0, :, :].shape)
print(np.argmax(train_ohe[0, :, :], axis=1))