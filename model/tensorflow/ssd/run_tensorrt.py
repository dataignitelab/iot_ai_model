
import os
import numpy as np
# import tensorflow as tf
from PIL import Image

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import torch
from torchvision import transforms 
from torchmetrics import F1Score
import logging

import cv2
import argparse
import tensorflow as tf
import os
import sys
import numpy as np
import yaml
from tqdm import tqdm

from anchor import generate_default_boxes
from box_utils_numpy import decode, compute_nms
from voc_data import VOCDataset
from image_utils import ImageVisualizer
from losses import create_losses
from network import create_ssd
from PIL import Image

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']

NUM_CLASSES = 11
BATCH_SIZE = 1

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', '1'):
        return True
    elif v.lower() in ('false', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def softmax(x, axis=1):
    max = np.max(x,axis=axis,keepdims=True) #returns max of each row and keeps same dims
    e_x = np.exp(x - max) #subtracts each row with its max value
    sum = np.sum(e_x,axis=axis,keepdims=True) #returns sum of each row and keeps same dims
    f_x = e_x / sum 
    return f_x

# def softmax(x):

#     y = np.exp(x - np.max(x))
#     f_x = y / np.sum(np.exp(x))
#     return f_x
        
        
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
       
            
    def __call__(self,x:np.ndarray,batch_size=1):
        
        x = x.astype(self.dtype)
        
        np.copyto(self.inputs[0].host,x.ravel())
        
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
        
        self.context.execute_async(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream) 
            
        
        self.stream.synchronize()
        return [out.host.reshape(batch_size,-1) for out in self.outputs]


def inference(model_path, data_path, display = False, save = False):
    logger.info('model loading.. {}'.format(model_path))
    labels = ["defect", "normal"]
    batch_size = 1
    
    model = TrtModel(model_path)
    shape = model.engine.get_binding_shape(0)
    
    
    with open('model/tensorflow/ssd/config.yml') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    try:
        config = cfg[args.arch.upper()]
    except AttributeError:
        raise ValueError('Unknown architecture: {}'.format(args.arch))
    
    use_tensor = False
    default_boxes = generate_default_boxes(config, use_tensor = use_tensor)
    
    voc = VOCDataset(data_path, default_boxes,
                     config['image_size'], -1, augmentation = False, use_tensor = use_tensor)
    
    visualizer = ImageVisualizer(['0','1','2','3','4','5','6','7','8','9'], save_dir='check_points/ssd/outputs/images')
    
    idx = 0
    
    if display:
        cv2.resizeWindow('img', 300, 300) 
    
    for filename, org_img, img, gt_confs, gt_locs in tqdm(voc.generate()):
        img = np.expand_dims(img, 0)
        confs, locs = model(img)
        
        confs = np.squeeze(confs, 0)
        locs = np.squeeze(locs, 0)
        
        confs = confs.reshape((8732, 11))
        locs = locs.reshape((8732, 4))
        
        confs = softmax(confs)
        classes = np.argmax(confs, axis=-1)
        scores = np.max(confs, axis=-1)
        
        boxes = decode(default_boxes, locs)
        
        out_boxes = []
        out_labels = []
        out_scores = []
        
        for c in range(1, NUM_CLASSES):
            cls_scores = confs[:, c]

            score_idx = cls_scores > 0.4
            
            cls_boxes = boxes[score_idx]
            cls_scores = cls_scores[score_idx]

            nms_idx = compute_nms(cls_boxes, cls_scores, 0.4, 20)
            cls_boxes = np.take(cls_boxes, nms_idx, axis=0)
            cls_scores = np.take(cls_scores, nms_idx, axis=0)
            cls_labels = [c] * cls_boxes.shape[0]

            out_boxes.append(cls_boxes)
            out_labels.extend(cls_labels)
            out_scores.append(cls_scores)

        out_boxes = np.concatenate(out_boxes, axis=0)
        out_scores = np.concatenate(out_scores, axis=0)

        boxes = np.minimum(np.maximum(out_boxes, 0.0), 1.0)
        classes = np.array(out_labels)
        scores = out_scores
        
        original_image = org_img
        boxes *= original_image.size * 2
        # break
        
        if display:
            visualizer.display_image(original_image, boxes, classes, '{:d}'.format(idx))
        
        if save:
            visualizer.save_image(original_image, boxes, classes, '{:d}'.format(idx))
        idx = idx + 1
    
    if(display):
        cv2.destroyAllWindows()
    
    # data_paths = glob(dataset_path)
    
    
#     logger.info('dataset loading..')
#     # gen, total = create_batch_generator(data_path)
#     gen = Dataset(data_path)
#     total = len(gen)
    
#     logger.info('number of test dataset : {}'.format(total))
    
#     logger.info('start inferencing')
#     f1 = F1Score(num_classes=2, threshold=0.5)
    
#     preds = []
#     targets = []
#     cnt = 0
    
#     start_time = time()
#     pre_elap = 0.0
#     fps = 0.0
#     for path, org_img, img, target in gen:
#         img = np.array([img])
#         path = np.array([path])
#         target = np.array([target])
        
#         output = model(img, batch_size)
        
#         loss = output[0][0]
#         output = 1 if output[0][0] >= 0.5 else 0
#         target = int(target[0])
#         preds.append(output)
#         targets.append(target)

#         cnt += 1
        
#         logger.info('{}/{} - {}, Predicted : {}, Actual : {}, Correct : {}, fps: {:.1f}'.format(cnt, total, path[0], labels[output], labels[target], output == target, fps))

#         if(display):
#             img = cv2.cvtColor(np.array(org_img), cv2.COLOR_RGB2BGR)
#             cv2.putText(img, 'Result: {}, Correct: {} '.format(labels[output], output == target), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
#             cv2.putText(img, 'FPS: {:.2f}'.format(fps), (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
#             cv2.imshow('img', img)
#             cv2.waitKey(1)
        
#         elap = time() - start_time
#         fps = max(0.0, 1.0 / (elap - pre_elap))
#         pre_elap = elap
        
#     elap = time() - start_time
#     fps = total / elap
    
#     if(display):
#         cv2.destroyAllWindows()

#     preds = torch.tensor(preds)
#     targets = torch.tensor(targets)
#     # acc = (correct/len(dataset))
#     f1_score = f1(preds, targets) 
    
#     logger.info('f1-score : {:.4f}, fps : {:.4f}'.format(float(f1_score), fps))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='resnet50')
    parser.add_argument('--model-path', dest='model_path', type=str, default='check_points/ssd/model.engine')
    # parser.add_argument('--data-path', dest='data_path', type=str, default='dataset/casting_data/test')
    parser.add_argument('--display', dest='display', type=str2bool, default=False)
    parser.add_argument('--save', dest='save', type=str2bool, default=False)
    parser.add_argument('--anno-path', default='dataset/server_room/test_digit.txt')
    parser.add_argument('--arch', default='ssd300')
    parser.add_argument('--num-examples', default=-1, type=int)
    parser.add_argument('--pretrained-type', default='specified')
    parser.add_argument('--gpu-id', default='0')
    
    args = parser.parse_args()
    logger.info(args)
    inference(args.model_path, args.anno_path, args.display, args.save)