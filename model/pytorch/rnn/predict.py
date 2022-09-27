import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader 
import time
from tqdm import tqdm
import logging
import random
import os
import torchmetrics
from torch.utils.data.dataset import random_split

from dataset import VibrationDataset
from model import RNNModel




if __name__ == '__main__':
    model_path = "check_points/rnn/"
    
    device = torch.device("cuda" if (not args.use_cpu) and torch.cuda.is_available() else "cpu")
    model = RNNModel(18, 8, 3, 5, 2).to(device)
    model.load_state_dict(torch.load(model_path))

    path = 'dataset/vibration/train/**/*.csv'
    dataset = VibrationDataset(path)