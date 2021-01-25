# Authors: Fabio Rodrigues Pereira
# E-mails: fabior@uio.no

from torchtext import datasets
from torchtext import data
import pandas as pd
import numpy as np
import torch


# data path:
DATAPATH = '~/Documents/IN5550_Neural_Methods_in_Natural_Language_Processing/Oblig1/' \
           'data/signal_20_obligatory1_train.tsv.gz'


# determine what device to use
DEVICE = torch.device(
  'cuda' if torch.cuda.is_available() else 'cpu'
)


class Feedforward(torch.nn.Module):

    def __init__(self, input_size, hidden_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.sigmoid(output)
        return output


if __name__ == '__main__':
    pass
