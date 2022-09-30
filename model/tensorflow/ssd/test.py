import argparse
import tensorflow as tf
import os
import sys
import numpy as np
import yaml
from tqdm import tqdm

from anchor import generate_default_boxes
from box_utils import decode, compute_nms
from voc_data import create_batch_generator
from image_utils import ImageVisualizer
from losses import create_losses
from network import create_ssd
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('--anno-path', default='dataset/server_room/test_digit.txt')
parser.add_argument('--arch', default='ssd300')
parser.add_argument('--num-examples', default=-1, type=int)
parser.add_argument('--pretrained-type', default='specified')
parser.add_argument('--checkpoint-dir', default='')
parser.add_argument('--checkpoint-path', default='check_points/ssd/model.h5') # latest
parser.add_argument('--gpu-id', default='0')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

NUM_CLASSES = 11
BATCH_SIZE = 1


def predict(imgs, default_boxes):
    confs, locs = ssd(imgs)
    confs = tf.squeeze(confs, 0)
    locs = tf.squeeze(locs, 0)

    confs = tf.math.softmax(confs, axis=-1)
    classes = tf.math.argmax(confs, axis=-1)
    scores = tf.math.reduce_max(confs, axis=-1)

    boxes = decode(default_boxes, locs)
    # print(boxes)

    out_boxes = []
    out_labels = []
    out_scores = []

    for c in range(1, NUM_CLASSES):
        cls_scores = confs[:, c]

        score_idx = cls_scores > 0.7
        
        # print((c-1), max(cls_scores))
        # cls_boxes = tf.boolean_mask(boxes, score_idx)
        # cls_scores = tf.boolean_mask(cls_scores, score_idx)
        cls_boxes = boxes[score_idx]
        cls_scores = cls_scores[score_idx]

        nms_idx = compute_nms(cls_boxes, cls_scores, 0.1, 200)
        cls_boxes = tf.gather(cls_boxes, nms_idx)
        cls_scores = tf.gather(cls_scores, nms_idx)
        cls_labels = [c] * cls_boxes.shape[0]

        out_boxes.append(cls_boxes)
        out_labels.extend(cls_labels)
        out_scores.append(cls_scores)

    out_boxes = tf.concat(out_boxes, axis=0)
    out_scores = tf.concat(out_scores, axis=0)

    boxes = tf.clip_by_value(out_boxes, 0.0, 1.0).numpy()
    classes = np.array(out_labels)
    scores = out_scores.numpy()

    return boxes, classes, scores


if __name__ == '__main__':
    print('model loading..')
    with open('model/tensorflow/ssd/config.yml') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    try:
        config = cfg[args.arch.upper()]
    except AttributeError:
        raise ValueError('Unknown architecture: {}'.format(args.arch))

    default_boxes = generate_default_boxes(config)

    batch_generator, info = create_batch_generator(
        args.anno_path, default_boxes,
        config['image_size'],
        BATCH_SIZE, args.num_examples, mode='test', augmentation = False)

    try:
        ssd = create_ssd(NUM_CLASSES, args.arch,
                         args.pretrained_type,
                         args.checkpoint_dir,
                         args.checkpoint_path)
    except Exception as e:
        print(e)
        print('The program is exiting...')
        sys.exit()

    os.makedirs('check_points/ssd/outputs/images', exist_ok=True)
    os.makedirs('check_points/ssd/outputs/detects', exist_ok=True)
    visualizer = ImageVisualizer(info['idx_to_name'], save_dir='check_points/ssd/outputs/images')
    
    print('run inferencing')
    progress = tqdm(batch_generator, total=info['length'], desc='Testing...', unit='images')
    for i, (filename, _, imgs, gt_confs, gt_locs) in enumerate(progress):
        boxes, classes, scores = predict(imgs, default_boxes)
        filename = filename[0].numpy().decode()
        original_image = Image.open(filename)
        boxes *= original_image.size * 2
        # break
        visualizer.save_image(original_image, boxes, classes, '{:d}'.format(i))

        log_file = os.path.join('check_points/ssd/outputs/detects', '{}.txt')

        for cls, box, score in zip(classes, boxes, scores):
            cls_name = info['idx_to_name'][cls - 1]
            with open(log_file.format(cls_name), 'a') as f:
                f.write('{} {} {} {} {} {}\n'.format(
                    filename,
                    score,
                    *[coord for coord in box]))
