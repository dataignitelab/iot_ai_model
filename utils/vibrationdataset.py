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
    
    def __init__(self, dataset_glob_path, workers=1, mode='csv', cached = False):
        self.dataset_path = dataset_glob_path
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
        self.cached = cached

        datapath = glob.glob(dataset_glob_path, recursive=True)
        self.file_path = datapath

        self.is_cached = [False for i in datapath]
        self.one_hot_target = [0 for i in datapath]
        self.data = [0 for i in datapath]

        a1 = self.labels.append
        a2 = self.idx_of_label.append
        a3 = self.rpm.append
        a4 = self.watt.append
        a5 = self.period.append
        a6 = self.sample_rate.append
        a7 = self.data.append
        a8 = self.one_hot_target.append

        if self.mode == 'csv':
            if (workers <= 1):
                for filename in tqdm(datapath):
                    csv = load_csv(filename)
                    a1(csv['label_name'])
                    a2(csv['idx'])
                    a3(csv['rpm'])
                    a4(csv['watt'])
                    a5(csv['period'])
                    a6(csv['sample_rate'])
                    a7(csv['data'])
                    a8(csv['one_hot_target'])
                    self.idx_to_label[csv['idx']] = csv['label_name']

            else:
                with Pool(workers) as p:
                    async_processes = [p.apply_async(load_csv, (filename, )) for filename in datapath]
                    rets = [proc.get() for proc in tqdm(async_processes)]
                    
                    for csv in tqdm(rets):
                        # ret = proc.get()
                        a1(csv['label_name'])
                        a2(csv['idx'])
                        a3(csv['rpm'])
                        a4(csv['watt'])
                        a5(csv['period'])
                        a6(csv['sample_rate'])
                        a7(csv['data'])
                        a8(csv['one_hot_target'])
                        self.idx_to_label[csv['idx']] = csv['label_name']

        # if self.cached:
        #     for idx, _ in tqdm(enumerate(datapath)):
        #         data, target = self.loadItem(idx)
        #         a7(data)
        #         a8(target)
            
    def loadItem(self, index):
        if self.mode == 'csv':
                x = self.data[index]
                sample_rate = self.sample_rate[index]
                class_idx = self.no[index]
                rpm = self.rpm[index]
                sample_rate = self.sample_rate[index]
        else:
            with open(self.file_path[index], 'rb') as file:
                data = pickle.load(file)
                x = data['data']
                sample_rate = data['sample_rate']
                class_idx = data['no']
                rpm = data['rpm']
                sample_rate = data['sample_rate']

        #if (x.shape[1] % sample_rate) != 0 : print(data['path'])
        # print(sample_rate, x.shape[1])
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
        return torch.tensor(seq, dtype=torch.float32), F.one_hot(torch.tensor(class_idx), num_classes=5).float()


    def __len__(self):
        return len(self.file_path)    

    def __getitem__(self, index):
        data, target = self.loadItem(index)
        return data, target

if __name__ == "__main__" :
    import pickle

    path = 'dataset/iot_sensor/vibration/**/**/*.csv'



    dataset = VibrationDataset(path , workers=6)