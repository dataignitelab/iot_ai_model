import os
import numpy as np
import xml.etree.ElementTree as ET
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--detect-dir', default='check_points/ssd/outputs/detects')
args = parser.parse_args()

def compute_ap(rec, prec):
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(rec >= t) == 0:
            p = 0
        else:
            p = np.max(prec[rec >= t])
        ap = ap + p / 11.

    return ap


def model_eval(det_file, anno, cls_name, iou_thresh=0.75):
    with open(det_file, 'r') as f:
        lines = f.readlines()

    lines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in lines]
    confs = np.array([float(x[1]) for x in lines])
    boxes = np.array([[float(z) for z in x[2:]] for x in lines])

    gts = {}
    cls_gts = {}
    npos = 0
    for image_id in image_ids:
        if image_id in cls_gts.keys():
            continue
        gt_boxes = np.array([ bbox[:4] for bbox in anno[image_id] if bbox[4] == cls_name ])
        
        npos = npos + gt_boxes.shape[0]
        cls_gts[image_id] = gt_boxes
        
    sorted_ids = np.argsort(-confs)
    sorted_scores = np.sort(-confs)
    
    boxes = boxes[sorted_ids, :]
    image_ids = [image_ids[x] for x in sorted_ids]

    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    
    for d in range(nd):
        gt_box = cls_gts[image_ids[d]].astype(float)
        box = boxes[d, :].astype(float)
        iou_max = -np.inf

        if gt_box.size > 0:
            ixmin = np.maximum(gt_box[:, 0], box[0])
            ixmax = np.minimum(gt_box[:, 2], box[2])
            iymin = np.maximum(gt_box[:, 1], box[1])
            iymax = np.minimum(gt_box[:, 3], box[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            uni = ((box[2] - box[0] + 1.0) * (box[3] - box[1] + 1.0) +
                   (gt_box[:, 2] - gt_box[:, 0] + 1.0) *
                   (gt_box[:, 3] - gt_box[:, 1] + 1.0) - inters)

            ious = inters / uni
            iou_max = np.max(ious)
            jmax = np.argmax(ious)

        if iou_max > iou_thresh:
            tp[d] = 1.0
        else:
            fp[d] = 1.0
            
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    
    recall = tp / float(npos)
    precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    
    ap = compute_ap(recall, precision)

    return recall, precision, ap


def evaluate():
    aps = {
        '0': 0.0,
        '1': 0.0,
        '2': 0.0,
        '3': 0.0,
        '4': 0.0,
        '5': 0.0,
        '6': 0.0,
        '7': 0.0,
        '8': 0.0,
        '9': 0.0,
        '0': 0.0,
        'mAP': []
    }
    
    anno = {}
    image_path = []
    bbox = []
    with open('dataset/server_room/test_digit.txt', 'r') as f:
        for row in f.readlines():
            col = row.split()
            filename = os.path.basename(col[0])
            for box in col[1:]:
                if filename not in anno.keys():
                    anno[filename] = []
                anno[filename].append(box.split(','))
                
    for cls_name in aps.keys():
        det_path = os.path.join(args.detect_dir, '{}.txt')
        
        if os.path.exists(det_path.format(cls_name)):
            recall, precision, ap = model_eval(det_path.format(cls_name), anno, cls_name)
            aps[cls_name] = ap
            aps['mAP'].append(ap)

    aps['mAP'] = np.mean(aps['mAP'])
    for key, value in aps.items():
        print('{}: {}'.format(key, value))

if __name__ == '__main__':
    evaluate()
    
