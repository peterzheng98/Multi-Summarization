import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import *
from torchvision import datasets, models, transforms
from util import *

# Bidirectional LSTM for word level encoder

class WordLevelEncoder(nn.Module):
    def __init__(self, config):
        '''
        :arg config: dict with all the configuration
        :param config: should at least include input size, hidden size, output size, num of layers
        '''
        super(WordLevelEncoder, self).__init__()
        self.config = config

        self.input_size = config["input_size"]
        self.hidden_size = config["hidden_size"]
        self.output_size = config["output_size"]
        self.num_layers = config["layers"]
        self.device = config["device"]
        self.dropout = 0.0
        try:
            self.dropout = config["dropout"]
        except ValueError as error:
            self.dropout = 0.5

        self.layer_core = nn.ModuleList()
        self.layer_core.append(
            nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.layers, batch_first=True,
                    dropout=self.dropout, bidirectional=False))
        self.layer_core.append(
            nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.layers, batch_first=True,
                    dropout=self.dropout, bidirectional=False))
        self.to(self.device)

    def initHiddenState(self, batchSize: int) -> (torch.Tensor, torch.Tensor):
        return (torch.zeros(self.num_layers * 2, batchSize, self.hidden_size).to(self.device),
                torch.zeros(self.num_layers * 2, batchSize, self.hidden_size).to(self.device))

    def forward(self, x, y, length):
        batchSize = x.size(0)
        maxLen = torch.max(length)
        x = x[:, 0:maxLen, :]
        y = y[:, 0:maxLen]
        x, y, length = sort_batch(x, y, length)
        x = x.to(self.device)
        y = y.to(self.device)

        hiddenState = [self.initHiddenState(batchSize) for _ in range(2)]

        out = [x, reverseSequence(x, length, True)]

        for idx in range(2):
            out[idx] = pack_padded_sequence(out[idx], length, True)
            out[idx], hiddenState[idx] = self.layer_core[idx](out[idx], hiddenState[idx])
            out[idx], _ = pad_packed_sequence(out[idx], True)
            if idx == 1:
                out[idx] = reverseSequence(out[idx], length, True)

        out = torch.cat(out, 2)
        # out = torch.squeeze(out)
        return y, out, length


class SentenceLevelEncoder(nn.Module):
    def __init__(self):