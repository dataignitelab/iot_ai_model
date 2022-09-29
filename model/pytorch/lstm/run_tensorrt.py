import numpy as np
import time
import logging
import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
import torchmetrics

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from dataset import CurrentDataset
from model import LSTMModel

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

    
class TrtModel:
    
    def __init__(self,engine_path,max_batch_size=1,dtype=np.float32):
        self.engine_path = engine_path
        self.dtype = dtype
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(self.runtime, self.engine_path)
        self.max_batch_size = max_batch_size
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        self.context = self.engine.create_execution_context()
        
                
    @staticmethod
    def load_engine(trt_runtime, engine_path):
        trt.init_libnvinfer_plugins(None, "")
        print('load', engine_path)
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine
    
    
    def allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.max_batch_size
            host_mem = cuda.pagelocked_empty(size, self.dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        
        return inputs, outputs, bindings, stream
    
            
    def __call__(self,x:np.ndarray,batch_size=2):
        x = x.astype(self.dtype)
        np.copyto(self.inputs[0].host,x.ravel())
        
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
        
        self.context.execute_async(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream) 
            
        self.stream.synchronize()
        return [out.host.reshape(batch_size,-1) for out in self.outputs]
    

def inference(model_path, data_path):
    labels = ['normal', 'def_baring', 'rotating_unbalance', 'def_shaft_alignment', 'loose_belt']
    num_classes = len(labels)
    use_cpu = False
    
    device = torch.device("cuda" if (use_cpu) and torch.cuda.is_available() else "cpu")

    model = TrtModel(model_path)
    shape = model.engine.get_binding_shape(0)
    
    dataset = CurrentDataset(data_path)
    total = len(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True,
                            drop_last=False)
    
    preds = torch.tensor([],dtype= torch.int16).to(device)
    targets = torch.tensor([],dtype= torch.int16).to(device)

    criterion = nn.CrossEntropyLoss()
    f1socre = torchmetrics.F1Score(num_classes = num_classes)
    cm = torchmetrics.ConfusionMatrix(num_classes = num_classes)
    
    avg_cost = .0
    cnt = 0
    start_time = time.time()
    with torch.no_grad():
        # progress = tqdm(dataloader)
        for samples in dataloader:
            cnt+=1
            file_path, x_train, y_train = samples

            x_train = x_train.numpy()
            y_train = y_train

            # H(x) 계산
            print(x_train.shape)
            outputs = model(x_train)
            outputs = torch.tensor(outputs)
            
            loss = criterion(outputs, y_train)
            avg_cost += loss
            
            out = torch.max(outputs.data, 1)[1]
            y = torch.max(y_train.data, 1)[1]

            preds = torch.cat([preds, out])
            targets = torch.cat([targets, y])
            
            logger.info('{}/{} - {}, Predicted : {}, Actual : {}, Correct : {}, loss : {:.4f}'.format(cnt, total, file_path[0], labels[out[0]], labels[y[0]], out[0] == y[0], loss))
    
    f1 = f1socre(preds.to('cpu'), targets.to('cpu'))
    avg_cost = avg_cost / total
    
    logger.info('time : {:.2f}, loss : {:.4f}, f1-score : {:.4f}'.format(
            time.time()-start_time,
            avg_cost,
            f1
        )
    )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='lstm')
    
    parser.add_argument('--model-path', dest='model_path', type=str, default='check_points/lstm/model.engine')
    parser.add_argument('--data-path', dest='data_path', type=str, default='dataset/current/test/**/*.csv')
    # parser.add_argument('--display', dest='display', type=str2bool, default=False)
    
    args = parser.parse_args()
    logger.info(args)
    inference(args.model_path, args.data_path)