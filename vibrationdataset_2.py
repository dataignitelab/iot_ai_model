import glob
import os
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import re
import pandas as pd
from multiprocessing import Queue, Pool, Manager

from .Vibration_Feature_Extractor import Extract_Time_Features, Extract_Freq_Features
import time
import pickle

import csv

def load_csv(filename):
    with open(filename) as f:
        for idx, row in enumerate(f):
            col = row.split(',')
            if idx >= 9: # vib
                vib[idx - 9] = float(col[1])
            elif idx == 2: # label name 
                label_name = col[1].rstrip()
                # self.labels.append(label_name)
            elif idx == 3: # label no
                no = int(col[1])
                # self.idx_of_label.append(int(col[1]))
                # self.idx_to_label[int(col[1])] = label_name
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
    return np.array([vib]), label_name, no, rpm, watt, period, sample_rate,F.one_hot(torch.tensor(no)).float()

def convert_csv_to_pk(filename):
    with open(filename) as f:
        for idx, row in enumerate(f):
            col = row.split(',')
            if idx >= 9: # vib
                vib[idx - 9] = float(col[1])
            elif idx == 2: # label name 
                label_name = col[1].rstrip()
                # self.labels.append(label_name)
            elif idx == 3: # label no
                no = int(col[1])
                # self.idx_of_label.append(int(col[1]))
                # self.idx_to_label[int(col[1])] = label_name
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
    basename = os.path.basename(filename)

    with open(os.path.join('dataset/iot_sensor_pickle/', (basename + '_' +str(time.time())+'.pk')), 'wb') as file:
        pickle.dump({
            'path': filename,
            'data': np.array([vib]), 
            'label': label_name, 
            'no' : no, 
            'rpm' : rpm, 
            'watt': watt, 
            'period': period, 
            'sample_rate': sample_rate,
            'one_hot_target': F.one_hot(torch.tensor(no)).float()
        }, file)

class ConvertCSVtoPickle():
     def __init__(self, dataset_glob_path, workers=1):
        datapath = glob.glob(dataset_glob_path, recursive=True)

        with Pool(workers) as p:
            async_processes = [p.apply_async(convert_csv_to_pk, (filename, )) for filename in datapath]
            rets = [proc.get() for proc in tqdm(async_processes)]

class VibrationDataset(Dataset):
    
    def __init__(self, info_path, workers=1, mode='csv'):
        self.dataset_path = info_path
        self.file_path = []
        self.labels = []
        self.data = []
        self.idx_of_label = []
        self.idx_to_label = {}
        self.rpm = []
        self.watt = []
        self.sample_rate = []
        self.period = []
        self.one_hot_target = []
        self.mode = mode
        self.regex = re.compile('[0-9]')

        with open(info_path) as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in tqdm(reader):
                self.file_path.append(row[0])
                self.labels.append(row[2])
                self.idx_of_label.append(int(row[3]))
                self.rpm.append(int(row[4]))
                self.sample_rate.append(int(row[5]))
                self.period.append(int(row[6]))
                self.watt.append(float(row[7]))

                self.idx_to_label[row[3]] = row[2]
        self.num_classes = len(self.idx_to_label)
            
    def loadItem(self, index):
        sample_rate = self.sample_rate[index]
        rpm = self.rpm[index]
        class_idx = self.idx_of_label[index]

        with open(self.file_path[index], 'rb') as file:
            data = pickle.load(file)
            x = data['data']

        seq = None
        for subset in range(0, x.shape[1], sample_rate):
            tmp = x[0][subset : subset + sample_rate].reshape(1,sample_rate)

            freq = Extract_Freq_Features(tmp, rpm, sample_rate)
            features = freq.Features()
            
            # time_vib = Extract_Time_Features(tmp)
            # features = time_vib.Features()

            features = features.reshape(1, features.shape[0])
            # print(index, features.shape)
            if seq is None:
                seq = features
            else:
                seq = np.append(seq, features, axis=0)

        return torch.tensor(seq, dtype=torch.float32), F.one_hot(torch.tensor(class_idx), num_classes = self.num_classes).float()


    def __len__(self):
        return len(self.file_path)    

    def __getitem__(self, index):
        data, target = self.loadItem(index)
        return data, target

if __name__ == "__main__" :
    import pickle

    path = 'dataset/iot_sensor/vibration/**/**/*.csv'
    dataset = VibrationDataset(path , workers=6)