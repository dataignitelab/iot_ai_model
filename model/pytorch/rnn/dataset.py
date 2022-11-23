import glob
import os
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
import torch
import re

from Vibration_Feature_Extractor import Extract_Time_Features, Extract_Freq_Features
import time
import pickle
import codecs
import csv

def load_csv(filename):
    with open(filename, encoding='utf8') as f:
        for idx, row in enumerate(f):
            col = row.split(',')
            if idx >= 9: # vib
                vib[idx - 9] = float(col[1])
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
                vib = np.zeros(int(col[1]))
            
    return np.array([vib]), label_name, no, rpm, watt, period, sample_rate


class VibrationDataset(Dataset):
    
    def __init__(self, dataset_dir, mode='csv'):
        self.labels = ['normal', 'def_baring', 'rotating_unbalance', 'def_shaft_alignment', 'loose_belt']
        self.num_classes = len(self.labels)
        
        self.dataset_path = dataset_dir
        self.file_path = glob.glob(dataset_dir, recursive=True)
                                   
            
    def loadItem(self, index):
        path = self.file_path[index]
        x, label_name, class_idx, rpm, watt, period, sample_rate = load_csv(path)
        
        seq = None
        for subset in range(0, x.shape[1], sample_rate):
            tmp = x[0][subset : subset + sample_rate].reshape(1,sample_rate)

            freq = Extract_Freq_Features(tmp, rpm, sample_rate)
            features1 = freq.Features() # 8
            time_vib = Extract_Time_Features(tmp)
            features2 = time_vib.Features() # 9
            features = np.concatenate([features1, features2, np.array([watt])])
            features = features.reshape(1, features.shape[0])
            if seq is None:
                seq = features
            else:
                seq = np.append(seq, features, axis=0)

        return torch.tensor(seq, dtype=torch.float32), one_hot(torch.tensor(class_idx), self.num_classes).float()

    def __len__(self):
        return len(self.file_path)    

    def __getitem__(self, index):
        data, target = self.loadItem(index)
        return self.file_path[index], data, target
    

if __name__ == "__main__" :
    path = 'dataset/vibration/train/**/**/*.csv'
                                   
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
    dataset = VibrationDataset(path)
    print(len(dataset))
    
    for x, target in dataset:
        print(x, target)
        break