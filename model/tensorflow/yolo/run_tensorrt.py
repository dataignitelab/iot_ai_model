# fp16 으로 변환 해야 성능 손실 적음

import os
import numpy as np
from PIL import Image

import logging
import cv2
import argparse
import os
import sys
import numpy as np
from tqdm import tqdm
from time import time

from tensorrt_model import TrtModel
from box_utils_numpy import compute_nms
from image_utils import ImageVisualizer
from eval import evaluate

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

INPUT_SIZE = 416
NUM_CLASS = 10
EPOCHS = 100
BATCH_SIZE = 64
IOU_THRESHOLD = 0.3

labels = ['0','1','2','3','4','5','6','7','8','9']
ANCHORS        = [23,27, 37,58, 81,82, 81,82, 135,169, 344,319]
STRIDES       =  [16, 32]
XYSCALE       = [1.05, 1.05]
ANCHOR_PER_SCALE     = 3

palette = [(255, 56, 56),
    (255, 157, 151),
    (255, 112, 31),
    (255, 178, 29),
    (207, 210, 49),
    (72, 249, 10),
    (146, 204, 23),
    (61, 219, 134),
    (26, 147, 52),
    (0, 212, 187),
    (44, 153, 168),
    (0, 194, 255),
    (52, 69, 147),
    (100, 115, 255),
    (0, 24, 236),
    (132, 56, 255),
    (82, 0, 133),
    (203, 56, 255),
    (255, 149, 200),
    (255, 55, 199)]

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
    max = np.max(x,axis=axis,keepdims=True)   # returns max of each row and keeps same dims
    e_x = np.exp(x - max)                     # subtracts each row with its max value
    sum = np.sum(e_x,axis=axis,keepdims=True) # returns sum of each row and keeps same dims
    f_x = e_x / sum 
    return f_x
        
def inference(model_path, data_path, display = False, save = False):
    logger.info('model loading.. {}'.format(model_path))
    batch_size = 1
    
    model = TrtModel(model_path)
    shape = model.engine.get_binding_shape(0)
    visualizer = ImageVisualizer(labels, save_dir='check_points/yolo/outputs/images')
    
    image_idx = 0
    
    list_filename = []
    list_classes = []
    list_boxes = []
    list_scores = []
    
    with open(data_path, 'r') as anno:
        lines = anno.readlines()
    
    dir_path = os.path.dirname(data_path)
        
    total = len(lines)
    start_time = time()
    pre_elap = 0.0
    fps = 0.0
        
    for row in lines:
        col = row.split()
        filename = os.path.join(dir_path, col[0])
        
        org_img = cv2.imread(filename)
        org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(org_img, (INPUT_SIZE, INPUT_SIZE))
        img = img.astype(np.float32) / 255.
        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
        
        h,w,_ = org_img.shape
        
        preds = model(img)
        
        locs = preds[1].reshape(-1, 4)
        confs = preds[0].reshape(-1, 10)
        
        out_boxes = []
        out_labels = []
        out_scores = []
        
        for c in range(0, NUM_CLASS):
            cls_scores = confs[:, c]

            score_idx = cls_scores > 0.8
            
            cls_boxes = locs[score_idx]
            cls_scores = cls_scores[score_idx]
            
            cls_boxes[..., :2] = cls_boxes[..., :2] - (cls_boxes[..., 2:] / 2)
            cls_boxes[..., 2:] = cls_boxes[..., 2:] + cls_boxes[..., :2]
            
            nms_idx = compute_nms(cls_boxes, cls_scores , 0.6, 5)
            
            cls_boxes = np.take(cls_boxes, nms_idx, axis=0)
            cls_scores = np.take(cls_scores, nms_idx, axis=0)
            cls_labels = [c] * cls_boxes.shape[0]

            out_boxes.append(cls_boxes)
            out_labels.extend(cls_labels)
            out_scores.append(cls_scores)
            
        out_boxes = np.concatenate(out_boxes, axis=0)
        out_scores = np.concatenate(out_scores, axis=0)

        out_boxes = out_boxes / INPUT_SIZE  * [w,h,w,h]
        out_boxes = out_boxes.astype(dtype=int)
        
        
        result_str = []
        for idx in range(len(out_boxes)):
            box = out_boxes[idx]
            cls = out_labels[idx]
            result_str.append( f'{box[0]},{box[1]},{box[2]},{box[3]},{cls}')
        result_str = ' '.join(result_str)
        logger.info('{}/{} - {}, Predicted : {} - fps: {:.1f}'.format(image_idx + 1, total, os.path.basename(filename), result_str, fps))
        
        if display:
            visualizer.display_image(org_img, out_boxes, out_labels, '{:d}'.format(image_idx))
        
        if save:
            visualizer.save_image(org_img, out_boxes, out_labels, '{:d}'.format(image_idx))
            
        image_idx = image_idx + 1
        
        list_filename.append(filename)
        list_classes.append(out_labels)
        list_boxes.append(out_boxes)
        list_scores.append(out_scores)
        
        elap = time() - start_time
        fps = max(0.0, 1.0 / (elap - pre_elap))
        pre_elap = elap
        
    elap = time() - start_time
    fps = total / elap
    
    
    if(display):
        cv2.destroyAllWindows()
        
    log_file = os.path.join('check_points/yolo/outputs/detects', '{}.txt')
    logger.info('calcurate mAP.. ')
    
    for cls in labels:
        f = log_file.format(cls)
        if os.path.exists(f):
            os.remove(f)
    
    for filename, classes, boxes, scores in zip(list_filename, list_classes, list_boxes, list_scores):    
        for cls, box, score in zip(classes, boxes, scores):
            cls_name = labels[cls]
            with open(log_file.format(cls_name), 'a') as f:
                f.write('{} {} {} {} {} {}\n'.format(
                    os.path.basename(filename),
                    score,
                    *[coord for coord in box]))
    
    iou_thresh = 0.70
    mAP = evaluate(display = False, iou_thresh = iou_thresh)
    
    for key, value in mAP.items():
        if key == 'mAP': continue
        logger.info('Class {}: AP {:.4f}'.format(key, value))
    logger.info('mAP@{}: {:.4f}, fps: {:.4f}'.format(iou_thresh, mAP['mAP'], fps))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='yolo')
    parser.add_argument('--model-path', dest='model_path', type=str, default='check_points/yolo/model.engine')
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