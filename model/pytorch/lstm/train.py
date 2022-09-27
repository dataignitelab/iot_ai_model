import pandas as pd
import pickle
from currentdataset import CurrentDataset
# from model.RNNModel import RNNModel

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


from model import LSTMModel



if __name__ == '__main__':
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = LSTMModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    
    dataset = CurrentDataset('didimdol/current_info.csv')
    train_loader = DataLoader(dataset,
                            batch_size=512,
                            shuffle=True,
                            num_workers=7,
                            pin_memory=True,
                            drop_last=False)

    f1socre = torchmetrics.F1Score(num_classes = 5)
    
    for epoch in range(100):
        progress = tqdm(train_loader)
        avg_cost = 0

        preds_hist = torch.tensor([],dtype= torch.int16).to(device)
        targets_hist = torch.tensor([],dtype= torch.int16).to(device)

        for data, targets in progress:
            data = data.to(device)
            targets = targets.to(device)

            preds = model(data)
            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_cost += loss
            progress.set_postfix({'loss' : loss.item()})

            preds = torch.max(preds.data, 1)[1]
            targets = torch.max(targets.data, 1)[1]

            preds_hist = torch.cat([preds_hist, preds])
            targets_hist = torch.cat([targets_hist, targets])

        f1 = f1socre(preds.to('cpu'), targets.to('cpu'))
        print('loss : {:.4f}, f1 : {:.4f}'.format(avg_cost / len(train_loader), f1))
