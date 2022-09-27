import tensorrt as trt
import numpy as np
import os
from PIL import Image
import cv2
from glob import glob 

import pycuda.driver as cuda
import pycuda.autoinit

import torch
from model.pytorch.inception.model import inceptionv4
from torchvision import transforms 


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


        
import time


def inference():
    batch_size = 1

    trt_engine_path = 'check_points/inception/inceptionv4_trt.engine' # os.path.join("..","models","main.trt")
    model = TrtModel(trt_engine_path)
    shape = model.engine.get_binding_shape(0)
    
    data_paths = glob('dataset/casting_data/test/defect/*.jpeg')
    
    start_time = time.time()
    for idx, p in enumerate(data_paths):
        img = get_img_np_nchw(p)
        result = model(img, batch_size)
        result = 1 if result[0][0] >= 0.5 else 0
        print(idx, p, result)
    
    elap = time.time() - start_time
    print(elap, idx+1)

if __name__ == "__main__":
    inference()