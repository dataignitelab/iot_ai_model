
import os
import numpy as np
# import tensorflow as tf
from PIL import Image

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
from time import time

from anchor import generate_default_boxes
from box_utils_numpy import decode, compute_nms
from image_utils import ImageVisualizer
from evaluate import evaluate
from PIL import Image

from network import create_ssd

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

labels = ['0','1','2','3','4','5','6','7','8','9']

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

def inference(model_path, data_path, display = False, save = False):
    logger.info('model loading.. {}'.format(model_path))
    batch_size = 1
    new_size = 300
    
    model = create_ssd(num_classes=11, arch='ssd300', pretrained_type=None, checkpoint_path=model_path)
    
    with open('model/tensorflow/ssd/config.yml') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    try:
        config = cfg[args.arch.upper()]
    except AttributeError:
        raise ValueError('Unknown architecture: {}'.format(args.arch))
    
    use_tensor = False
    default_boxes = generate_default_boxes(config, use_tensor = use_tensor)
    
    visualizer = ImageVisualizer(labels, save_dir='check_points/ssd/outputs/images')
    
    list_filename = []
    list_classes = []
    list_boxes = []
    list_scores = []
    
    with open(data_path, 'r') as anno:
        lines = anno.readlines()

    total = len(lines)
    start_time = time()
    pre_elap = 0.0
    fps = 0.0    

    for image_idx, row in enumerate(lines):
        col = row.split()
        filename = os.path.join('dataset/server_room',col[0])
        org_img = Image.open(filename)
        img = np.array(org_img.resize((new_size, new_size)), dtype=np.float32)
        img = (img / 127.0) - 1.0
        img = img.reshape(1,img.shape[0],img.shape[1],img.shape[2])
        confs, locs = model(img)
        confs = np.squeeze(confs, 0)
        locs = np.squeeze(locs, 0)
        
        # confs = confs.reshape((8732, 11))
        # locs = locs.reshape((8732, 4))
        
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

            nms_idx = compute_nms(cls_boxes, cls_scores, 0.1, 100)
            cls_boxes = np.take(cls_boxes, nms_idx, axis=0)
            cls_scores = np.take(cls_scores, nms_idx, axis=0)
            cls_labels = [c] * cls_boxes.shape[0]

            out_boxes.append(cls_boxes)
            out_labels.extend(cls_labels)
            out_scores.append(cls_scores)

        out_boxes = np.concatenate(out_boxes, axis=0)
        out_scores = np.concatenate(out_scores, axis=0)

        boxes = np.minimum(np.maximum(out_boxes, 0.0), 1.0)
        # classes = out_labels # np.array(out_labels)
        # scores = out_scores
        
        out_boxes *= org_img.size * 2
        boxes = out_boxes.astype(dtype=np.int16)
        
        result_str = []
        for idx in range(len(boxes)):
            box = out_boxes[idx]
            cls = out_labels[idx]
            result_str.append( f'{int(box[0])},{int(box[1])},{int(box[2])},{int(box[3])},{cls}')
        result_str = ' '.join(result_str)
        logger.info('{}/{} - {}, Predicted : {} - fps: {:.1f}'.format(image_idx + 1, total, os.path.basename(filename), result_str, fps))
        
        # break
        
        if display:
            visualizer.display_image(org_img, boxes, out_labels)
        
        if save:
            visualizer.save_image(org_img, boxes, out_labels, '{:d}'.format(image_idx))
        
        list_filename.append(filename)
        list_classes.append(out_labels)
        list_boxes.append(boxes)
        list_scores.append(out_scores)
        
        elap = time() - start_time
        fps = max(0.0, 1.0 / (elap - pre_elap))
        pre_elap = elap
        
    elap = time() - start_time
    fps = total / elap
    
    if(display):
        cv2.destroyAllWindows()
        
    log_file = os.path.join('check_points/ssd/outputs/detects', '{}.txt')
    logger.info('calcurate mAP..')
    
    for cls in labels:
        f = log_file.format(cls)
        if os.path.exists(f):
            os.remove(f)
    
    for filename, classes, boxes, scores in zip(list_filename, list_classes, list_boxes, list_scores):    
        for cls, box, score in zip(classes, boxes, scores):
            cls_name = labels[cls - 1]
            with open(log_file.format(cls_name), 'a') as f:
                f.write('{} {} {} {} {} {}\n'.format(
                    os.path.basename(filename),
                    score,
                    *[coord for coord in box]))
    
    iou_thresh = 0.75
    mAP = evaluate(display = False, iou_thresh = iou_thresh)
    
    for key, value in mAP.items():
        if key == 'mAP': continue
        logger.info('Class {}: AP {:.4f}'.format(key, value))
    logger.info('mAP@{}: {:.4f}, fps: {:.4f}'.format(iou_thresh, mAP['mAP'], fps))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ssd resnet18')
    parser.add_argument('--model-path', dest='model_path', type=str, default='check_points/ssd/ssd_epoch_latest.h5')
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