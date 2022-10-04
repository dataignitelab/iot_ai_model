from glob import glob
from tqdm import tqdm
from time import time
import logging
import os
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import random_split

from model import Unet
from dataset import ImageDataset, load_image

import torch
from model import Unet

BATCH_SIZE = 1
EPOCHS = 100
LR = 0.0001

checkpoints_path = 'check_points/unet'
data_path = 'dataset/supervisely_person'

paths = glob(os.path.join(data_path,"**/*.png"))

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
    inputs = inputs.contiguous().view(-1)
    targets = targets.contiguous().view(-1)

    intersection = (inputs * targets).sum()                            
    dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)  

    return 1-dice 

def display_image(img, mask, local = False):
    img = img[0]
    mask = mask[0]
    
    img = np.transpose(img, (1,2,0))
    mask = np.transpose(mask, (1,2,0))
    
    img = img * 255
    img = np.minimum(np.maximum(img, 255), 0)
    mask[mask > 0.5] = 255
    mask[mask <= 0.5] = 0
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
    logger.info('dataset loading..')
   
    # with open(data_path, 'r') as f:
    #     line = f.readlines()

    
    logger.info('start inferencing')
    preds = []
    targets = []
    cnt = 0
    
    base_dir = os.path.dirname(data_path)
    imgs = []
    masks = []
    filepaths = []
    
    checkpoints_path = 'check_points/unet'
    model_path = f"{checkpoints_path}/model_state_dict_latest.pt"
    output = f'{checkpoints_path}/model.onnx'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Unet().to(device)
    model.load_state_dict(torch.load(model_path))
    model = model.eval()
    
#     dataset = ImageDataset(data_path, augmentation=False, preload= False)
#     loader = torch.utils.data.DataLoader(dataset, batch_size= BATCH_SIZE, shuffle=False)
    
#     total = len(dataset)
#     logger.info('number of test dataset : {}'.format(total))
    
    with open(data_path, 'r') as f:
        line = f.readlines()

    total = len(line)
    logger.info('number of test dataset : {}'.format(total))
    
    
    start_time = time()
    pre_elap = 0.0
    fps = 0.0
    cost = .0
    loss = .0
    
    base_dir = os.path.dirname(data_path)
    imgs = []
    masks = []
    filepaths = []
    
    for row in tqdm(line):
        img_path, mask_path = row.rstrip().split(',')
        img = torch.tensor(load_image(os.path.join(base_dir, img_path))).type(torch.float32)
        mask = torch.tensor(load_image(os.path.join(base_dir, mask_path))).type(torch.float32)
        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
        mask = mask.reshape(1, mask.shape[0], mask.shape[1], mask.shape[2])
        
        imgs.append(img)
        masks.append(mask)
        filepaths.append(os.path.basename(img_path))
    
    start_time = time()
    pre_elap = 0.0
    fps = 0.0
    cost = .0
    loss = .0
    
    with torch.no_grad():
        # for idx, (filename, img, mask) in enumerate(loader):
        for idx, (filename, img, mask) in enumerate(zip(filepaths, imgs, masks)):
            img = img.to(device)
            mask = mask.to(device)

            output = model(img)
            loss = dice_loss(output, mask)

            cost += loss.cpu().item()

            logger.info('{}/{} - {},  fps: {:.1f}, dice loss: {:.1f}'.format(idx+1, total, filename, fps, (loss)))

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