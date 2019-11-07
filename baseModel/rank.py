import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torch.tensor import Tensor
from torch import device as D
import pickle
from torch import FloatTensor as FT
from gensim.models.fasttext import FastText
import time
import logging

