import glob
import os
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
import torch
import re
import csv

import time
import pickle

def load_csv(filename):
    with open(filename, encoding='utf8') as f:
        cur = None
        for idx, row in enumerate(f):
            col = row.split(',')
            if idx >= 9: # vib
                cur[idx - 9] = col[1:4]
            elif idx == 2: # label name 
                label_name = col[1].rstrip()
            elif idx == 3: # label no
                no = int(col[1])
            elif idx == 4: # motor spec
                rpm = int(col[2]) # rpm
                watt = float(col[3]) # watt
            elif idx == 5: # period
                p = re.compile('[0-9]').findall(col[1])
                if len(p) > 0 :
                    period = int(p[0])
                else:
                    period = 0
            elif idx == 6: #sample rate
                sample_rate = int(col[1])
            elif idx == 8:
                cur = np.zeros((int(col[1]), 3))
    return np.array(cur, dtype=np.float32), label_name, no, rpm, watt, period, sample_rate

class CurrentDataset(Dataset):
    
    def __init__(self, dataset_dir):
        self.labels = ['normal', 'def_baring', 'rotating_unbalance', 'def_shaft_alignment', 'loose_belt']
        self.num_classes = len(self.labels)
        self.dataset_path = dataset_dir
        self.file_path = glob.glob(dataset_dir, recursive=True)

    def loadItem(self, index):
        x, label_name, class_idx, rpm, watt, period, sample_rate = load_csv(self.file_path[index])
        # x = np.array(x)
        x = x - np.expand_dims(np.mean(x,axis=1),axis=1)
        return torch.tensor(x, dtype=torch.float32), one_hot(torch.tensor(class_idx), num_classes=self.num_classes).float()

    def __len__(self):
        return len(self.file_path)    

    def __getitem__(self, index):
        data, target = self.loadItem(index)
        return self.file_path[index], data, target

    
if __name__ == "__main__" :
    path = 'dataset/current/train/**/normal/*.csv'
    tmp = glob.glob(path, recursive=True)
    
    print('Data cleansing..')
    error_file = 0
    for path in tqdm(tmp):
        try:
            load_csv(path)
        except UnicodeDecodeError:
            error_file += 1
            os.remove(path)
                                   
    print(f'Removed err files : {error_file}')
    
    dataset = CurrentDataset(path)
    print(len(dataset))
    
    for x, target in dataset:
        print(x.shape, target)
        break