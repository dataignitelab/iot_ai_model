from glob import glob
from tqdm import tqdm
from time import time
import argparse
import logging
import os
import cv2
import numpy as np
from PIL import Image
import base64
# from matplotlib import pyplot as plt

from model import Unet
from dataset import load_image
from tensorrt_model import TrtModel

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
        
def dice_loss(inputs, targets, smooth=1):
    inputs = inputs.reshape(-1)
    targets = targets.reshape(-1)

    intersection = (inputs * targets).sum()                            
    dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)  

    return dice 

def display_image(img, mask, local = False):
    img = img[0]
    mask = mask[0]
    
    img = np.transpose(img, (1,2,0))
    mask = np.transpose(mask, (1,2,0))
    
    img = img * 255
    img = np.minimum(np.maximum(img, 0), 255)
    mask[mask > 0.5] = 255
    mask[mask <= 0.5] = 0
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = img.astype(np.int16)
    green = np.zeros_like(mask)
    green[:,:,1] = mask[:,:,1]
    img[green >= 255] = img[green >= 255] * 3
    img[img >= 255] = 255

    other = np.zeros_like(mask)
    other[:,:,[0,2]] = mask[:,:,[0,2]] 
    img[other >= 255] = img[other >= 255] * 0.3
    
    # plt.imshow(img)
    cv2.imshow('img', img)
    cv2.waitKey(1)

def inference(model_path, data_path, display = False):
    logger.info('model loading.. {}'.format(model_path))
    batch_size = 1
     # os.path.join("..","models","main.trt")
    model = TrtModel(model_path)
    shape = model.engine.get_binding_shape(0)
    
    # data_paths = glob(dataset_path)
    
    
    logger.info('dataset loading..')
   
    with open(data_path, 'r') as f:
        line = f.readlines()

    total = len(line)
    logger.info('number of test dataset : {}'.format(total))
    
    logger.info('start inferencing')
    preds = []
    targets = []
    cnt = 0
    
    base_dir = os.path.dirname(data_path)
    imgs = []
    masks = []
    filepaths = []
    
    for row in tqdm(line):
        img_path, mask_path = row.rstrip().split(',')
        
        img = load_image(os.path.join(base_dir, img_path))
        mask = load_image(os.path.join(base_dir, mask_path))
        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
        
        imgs.append(img)
        masks.append(mask)
        filepaths.append(os.path.basename(img_path))
    
    start_time = time()
    pre_elap = 0.0
    fps = 0.0
    cost = .0
    loss = .0
    for idx, (filename, img, mask) in enumerate(zip(filepaths, imgs, masks)):
        img = load_image(os.path.join(base_dir, img_path))
        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
        
        output = model(img)
        output = output[0].reshape(img.shape)
        
        loss = dice_loss(output, mask)
        
        cost += loss
        
        logger.info('{}/{} - {},  fps: {:.1f}, dice loss: {:.1f}'.format(idx+1, total, filename, fps, (1-loss)))

        if(display):
            display_image(img, output)
        
        elap = time() - start_time
        fps = max(0.0, 1.0 / (elap - pre_elap))
        pre_elap = elap
        
    if(display):
        cv2.destroyAllWindows()

    # preds = torch.tensor(preds)
    # targets = torch.tensor(targets)
    # # acc = (correct/len(dataset))
    # f1_score = f1(preds, targets) 
    
    elap = time() - start_time
    fps = total / elap
    logger.info('dice coefficient: {:.4f}, fps: {:.4f}'.format(cost/total, fps))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='unet')
    
    parser.add_argument('--model-path', dest='model_path', type=str, default='check_points/unet/model.engine')
    parser.add_argument('--data-path', dest='data_path', type=str, default='dataset/supervisely_person/test_data_list.txt')
    parser.add_argument('--display', dest='display', type=str2bool, default=False)
    
    args = parser.parse_args()
    logger.info(args)
    inference(args.model_path, args.data_path, args.display)