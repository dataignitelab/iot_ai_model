from glob import glob
from tqdm import tqdm
from time import time
import argparse
import logging
import os
import cv2
import numpy as np
from PIL import Image

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

    return 1-dice 

def display_image(img, mask, gui = True):
    img = img.reshape(img.shape[1], img.shape[2], img.shape[3])
    mask = mask.reshape(mask.shape[1], mask.shape[2], mask.shape[3])
   
    img = img.astype(np.float32)
    img = np.transpose(img, (1,2,0))
    mask = np.transpose(mask, (1,2,0))
    
    img = img * 255
    img = np.minimum(np.maximum(img, 0), 255)
    mask[mask > 0.5] = 255
    mask[mask <= 0.5] = 0
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #img = img.astype(np.uint8)
    green = np.zeros_like(mask)
    green[:,:,1] = mask[:,:,1]
    img[green >= 255] = img[green >= 255] * 2
    img[img >= 255] = 255

    other = np.zeros_like(mask)
    other[:,:,[0,2]] = mask[:,:,[0,2]] 
    img[other >= 255] = img[other >= 255] * 0.5
    
    img = img.astype(np.uint8)
    if gui :
        cv2.imshow('img', img)
        cv2.waitKey(1)
    else:
        cv2.imwrite('unet.jpg', img)

def inference(model_path, data_path, display = False):
    logger.info('model loading.. {}'.format(model_path))
    batch_size = 1
    
    model = TrtModel(model_path)
    shape = model.engine.get_binding_shape(0)
    
    logger.info('dataset loading..')
    with open(data_path, 'r') as f:
        line = f.readlines()

    total = len(line)
    preds = []
    targets = []
    cnt = 0
    base_dir = os.path.dirname(data_path)
    # imgs = []
    masks = []
    # filepaths = []
    
    for row in tqdm(line):
        img_path, mask_path = row.rstrip().split(',')
        
        # img = load_image(os.path.join(base_dir, img_path))
        mask = load_image(os.path.join(base_dir, mask_path))
        # img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
        mask = mask.reshape(1, mask.shape[0], mask.shape[1], mask.shape[2])
        
        # imgs.append(img)
        masks.append(mask)
        # filepaths.append(os.path.basename(img_path))
    
    logger.info('number of test dataset : {}'.format(total))
    
    start_time = time()
    pre_elap = 0.0
    fps = 0.0
    cost = .0
    loss = .0
    # for idx, (filename, img, mask) in enumerate(zip(filepaths, imgs, masks)):
    
    logger.info('start inferencing')
    for idx, row in enumerate(line):
        filename, mask_path = row.rstrip().split(',')
        
        img = load_image(os.path.join(base_dir, filename))
        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
        
        # mask = load_image(os.path.join(base_dir, mask_path))
        # mask = mask.reshape(1, mask.shape[0], mask.shape[1], mask.shape[2])
        mask = masks[idx]
        
        output = model(img)
        output = output[0].reshape(mask.shape)
        
        loss = dice_loss(output, mask)
        cost += loss
        
        logger.info('{}/{} - {},  fps: {:.1f}, dice coefficient: {:.1f}'.format(idx+1, total, filename, fps, (1-loss)))
        if(display):
            display_image(img, mask)
        
        elap = time() - start_time
        fps = max(0.0, 1.0 / (elap - pre_elap))
        pre_elap = elap
        
    if(display):
        cv2.destroyAllWindows()

    
    elap = time() - start_time
    fps = total / elap
    logger.info('dice coefficient: {:.4f}, fps: {:.4f}'.format(1 - (cost/total), fps))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='unet')
    
    parser.add_argument('--model-path', dest='model_path', type=str, default='check_points/unet/model.engine')
    parser.add_argument('--data-path', dest='data_path', type=str, default='dataset/supervisely_person/test_data_list.txt')
    parser.add_argument('--display', dest='display', type=str2bool, default=False)
    
    args = parser.parse_args()
    logger.info(args)
    inference(args.model_path, args.data_path, args.display)
