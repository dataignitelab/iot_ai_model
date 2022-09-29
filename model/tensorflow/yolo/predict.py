import tensorflow as tf
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import cv2
import logging
import os

from yolo import createModel, decode, compute_loss, decode_train
from dataset import Dataset

plt.rcParams["figure.figsize"] = (20,10)

check_point_path = 'check_points/yolo'

INPUT_SIZE = 416
NUM_CLASS = 10
EPOCHS = 100
BATCH_SIZE = 64
IOU_LOSS_THRESH = 0.3

CLASSES = ['0','1','2','3','4','5','6','7','8','9']
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


def infer(model, image_path, IOU_THRESHOLD = 0.4, INPUT_SIZE= 416):
    o_image = cv2.imread(image_path)
    o_image = cv2.cvtColor(o_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(o_image, (INPUT_SIZE, INPUT_SIZE))
    image = image / 255.
    

    images_data = []
    pred_bbox = []
    for i in range(1):
        images_data.append(image)
    images_data = np.asarray(images_data).astype(np.float32)
    batch_data = tf.constant(images_data)
    
    preds = model(batch_data, training=False)

    box_list = []
    conf_list = []
    for idx, output in enumerate(preds):
        if idx % 2 == 0 : continue
        boxes = output[:, :, :, :,  0:4]
        pred_conf = output[:, :, :, :, 4:]

        bs, xi, yi, anc, xywh = boxes.shape
        box_list.append(tf.reshape(boxes, (bs, -1, 1, xywh)))
        bs, xi, yi, anc, conf = pred_conf.shape
        conf_list.append(tf.reshape(pred_conf, (bs, -1, conf)))

    boxes = tf.concat([box_list[0], box_list[1]], 1).numpy()
    pred_conf =  tf.concat([conf_list[0], conf_list[1]], 1).numpy()

    classes_prob =  pred_conf[:, :, 1:] * pred_conf[:, :, :1]

    boxes[:,:,:,0] = boxes[:,:,:,0] - (boxes[:,:,:, 2] / 2) # x1
    boxes[:,:,:,1] = boxes[:,:,:,1] - (boxes[:,:,:, 3] / 2) # y1

    boxes[:,:,:,2] = boxes[:,:,:,0] + boxes[:,:,:, 2] # x2
    boxes[:,:,:,3] = boxes[:,:,:,1] + boxes[:,:,:, 3] # y2

    o_boxes, scores, classes, detections = tf.image.combined_non_max_suppression(
        boxes=boxes,
        scores=classes_prob,
        max_output_size_per_class= 20,
        max_total_size=30,
        iou_threshold=IOU_THRESHOLD,
        score_threshold=0.25,
        clip_boxes = False
    )

    return o_image, o_boxes, scores, classes, detections

def bbox_iou(box1, box2):
    w = max(min(box1[2], box2[2]) - max(box1[0], box2[0]), 0)
    h = max(min(box1[3], box2[3]) - max(box1[1], box2[1]), 0)

    g = ((box1[2] - box1[0]) * (box1[3] - box1[1])) + ((box2[2] - box2[0]) * (box2[3] - box2[1]))
  
    iou = (w * h)/g if (w * h) != 0 else 0.

    return iou



if __name__ == '__main__':
    annot_path = 'dataset/server_room/test_digit.txt'
    
    with open(annot_path, 'r') as anno:
        lines = anno.readlines()

        preds = []
        trues = []

        for row in tqdm(lines):
            col = row.split()
            img_path = col[0]

            img, o_boxes, scores, classes, detections = infer(model, img_path, IOU_THRESHOLD=IOU_LOSS_THRESH) 

            num_objects = len(col[1:])

            collect_count = 0

            pred_label = np.zeros(shape=[num_objects], dtype=np.int16)
            # pred_label += 10

            img_w, img_h, _ = img.shape
            for i, bbox in enumerate(col[1:]):
                x1, y1, x2, y2, label = bbox.split(',')

                t_box = np.array([x1, y1, x2, y2], dtype = np.float32)
                t_label = int(label)

                trues.append(t_label)

                batch_idx = 0
                cur_iou = IOU_LOSS_THRESH
                for j in range(detections[0].numpy()):
                    p_box = o_boxes[batch_idx][j]

                    p_box = p_box / INPUT_SIZE
                    x1 = int(img_w * p_box[0])
                    y1 = int(img_h * p_box[1])
                    x2 = int(img_w * p_box[2])
                    y2 = int(img_h * p_box[3])

                    p_box = [x1,y1,x2,y2]

                    iou = bbox_iou(t_box, p_box)
                    if iou >= cur_iou:
                        pred_label[i] = int(classes[batch_idx][j])
                        cur_iou = iou

            preds += pred_label.tolist()
            
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, plot_confusion_matrix
    import matplotlib.pyplot as plt

    mat_con = (confusion_matrix(trues, preds, labels=[0,1,2,3,4,5,6,7,8,9,10]))
    plt.matshow(mat_con, cmap=plt.cm.Blues, alpha=0.5)
    for m in range(mat_con.shape[0]):
        for n in range(mat_con.shape[1]):
            plt.text(x=n,y=m,s=mat_con[m, n], va='center', ha='center', size='xx-large')

    plt.xticks([0,1,2,3,4,5,6,7,8,9,10], labels=['0','1','2','3','4','5','6','7','8','9','none'])
    plt.yticks([0,1,2,3,4,5,6,7,8,9,10], labels=['0','1','2','3','4','5','6','7','8','9','none'])
    plt.xlabel('Predictions', fontsize=16)
    plt.ylabel('Trues', fontsize=16)
    plt.title('Confusion Matrix', fontsize=15)

    plt.show()