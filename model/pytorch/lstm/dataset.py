import glob
import os
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import re
# from multiprocessing import Queue, Pool, Manager
import csv

# from Current_Feature_Extractor import Extract_Time_Features, Extract_Freq_Features
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
                cur = np.zeros((int(col[1]), 3))
    return np.array(cur, dtype=np.float32), label_name, no, rpm, watt, period, sample_rate

def convert_csv_to_pk(filename):
    with open(filename) as f:
        for idx, row in enumerate(f):
            col = row.split(',')
            if idx >= 9: # vib
                vib[idx - 9] = [float(col[1]), float(col[2]), float(col[3])]
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
                vib = np.zeros(int(col[1]), 3)
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

    

class CurrentDataset(Dataset):
    
    def __init__(self, dataset_dir):
        self.labels = ['normal', 'def_baring', 'rotating_unbalance', 'def_shaft_alignment', 'loose_belt']
        self.num_classes = len(self.labels)
        self.dataset_path = dataset_dir
        self.file_path = glob.glob(dataset_dir, recursive=True)
        
#     def loadItem_(self, index):
#         with open(self.file_path[index], 'rb') as file:
#             data = pickle.load(file)
#             x = data['data']
#             sample_rate = self.sample_rate[index]
#             class_idx = self.idx_of_label[index]
#             rpm = self.rpm[index]

#         x = x - np.expand_dims(np.mean(x,axis=1),axis=1)

#         return torch.tensor(x, dtype=torch.float32), F.one_hot(torch.tensor(class_idx), num_classes=self.num_classes).float()

    def loadItem(self, index):
        x, label_name, class_idx, rpm, watt, period, sample_rate = load_csv(self.file_path[index])
        # x = np.array(x)
        x = x - np.expand_dims(np.mean(x,axis=1),axis=1)

#         seq_list = []
#         seq_size = 200
#         for idx in range(len(x) - seq_size + 1) :
#             seq_list.append(x[idx:idx+seq_size])

#         x = np.array(seq_list)
        
        # y = [class_idx for _ in range(len(x))]
        return torch.tensor(x, dtype=torch.float32), F.one_hot(torch.tensor(class_idx), num_classes=self.num_classes).float()


    def __len__(self):
        return len(self.file_path)    

    def __getitem__(self, index):
        data, target = self.loadItem(index)
        # print(data.shape, target.shape)
        return self.file_path[index], data, target

if __name__ == "__main__" :
    path = 'dataset/current/train/**/**/*.csv'
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