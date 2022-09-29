import tensorrt as trt
import numpy as np
import os
from PIL import Image
from time import time
import cv2
from glob import glob 
import argparse

import pycuda.driver as cuda
import pycuda.autoinit

from dataset import ImageDataset
import torch
from model import inceptionv4
from torchvision import transforms 
from torchmetrics import F1Score
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', '1'):
        return True
    elif v.lower() in ('false', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_img_np_nchw(img_path):
    img = Image.open(img_path)
    assert img is not None, 'Image not found {}'.format(self.file_path[index])

    ##img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = transform(img)
    # img = img.reshape((1,3,224,224))
    img = torch.unsqueeze(img, dim=0)
    img = np.array(img)
    return img

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


def inference(model_path, data_path, display = False):
    logger.info('model loading.. {}'.format(model_path))
    labels = ["normal", "defect"]
    batch_size = 1
     # os.path.join("..","models","main.trt")
    model = TrtModel(model_path)
    shape = model.engine.get_binding_shape(0)
    
    # data_paths = glob(dataset_path)
    
    
    logger.info('dataset loading..')
    tranform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageDataset(data_path, labels, tranform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    
    total = len(dataset)
    logger.info('number of test dataset : {}'.format(total))
    
    logger.info('start inferencing')
    f1 = F1Score(num_classes=2, threshold=0.5)
    
    preds = []
    targets = []
    cnt = 0
    
    start_time = time()
    pre_elap = 0.0
    fps = 0.0
    for path, data, target in dataloader:
        img = np.array(data)
        output = model(img, batch_size)
        
        loss = output[0][0]
        output = 1 if output[0][0] >= 0.5 else 0
        target = int(target[0])
        preds.append(output)
        targets.append(target)

        cnt += 1
        
        logger.info('{}/{} - {}, Predicted : {}, Actual : {}, Correct : {}, fps: {}'.format(cnt, total, path[0], labels[output], labels[target], output == target, loss))

        if(display):
            img = cv2.imread(path[0])

            cv2.putText(img, 'Result: {}, Correct: {} '.format(labels[output], output == target), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
            cv2.putText(img, 'FPS: {:.2f}'.format(fps), (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
            cv2.imshow('img', img)
            cv2.waitKey(1)
        
        elap = time() - start_time
        fps = max(0.0, 1.0 / (elap - pre_elap))
        pre_elap = elap
        
    if(display):
        cv2.destroyAllWindows()

    preds = torch.tensor(preds)
    targets = torch.tensor(targets)
    # acc = (correct/len(dataset))
    f1_score = f1(preds, targets) 
    
    elap = time() - start_time
    fps = total / elap
    logger.info('f1-score : {:.4f}, fps : {:.4f}'.format(float(f1_score), fps))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inceptionv4')
    
    parser.add_argument('--model-path', dest='model_path', type=str, default='check_points/inception/model.engine')
    parser.add_argument('--data-path', dest='data_path', type=str, default='dataset/casting_data/test')
    parser.add_argument('--display', dest='display', type=str2bool, default=False)
    
    args = parser.parse_args()
    logger.info(args)
    inference(args.model_path, args.data_path, args.display)