import torch
from torch import nn
import yaml

with open('conf.yml') as f:
    cfg = yaml.load(f.read())

class RNNmodel(nn.Module):
    def __init__(self):
        super(RNNmodel, self).__init__()
        self.lstm_layer = nn.LSTM(
            input_size=cfg['model']['input_size'],
            hidden_size=cfg['model']['output_size'],
            num_layers=cfg['model']['num_layers'],
            batch_first=True,
            dropout=cfg['model']['dropout'],
            bidirectional=cfg['model']['bidirectional']
        )

    def forward(self, x):
        output = self.lstm_layer(x)
        return output
