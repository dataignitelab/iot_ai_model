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
import argparse

from yolo import createModel, decode, compute_loss, decode_train
from dataset import Dataset

plt.rcParams["figure.figsize"] = (20,10)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno-path', default='dataset/server_room/train_digit.txt')
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--num-batches', default=-1, type=int)
    parser.add_argument('--initial-lr', default=1e-3, type=float)
    parser.add_argument('--num-epochs', default=100, type=int)
    parser.add_argument('--checkpoint-dir', default='./check_points/yolo')
    parser.add_argument('--checkpoint-path', default='check_points/yolo/epoch_latest.h5') # latest

    args = parser.parse_args()
    
    check_point_path = args.checkpoint_dir

    INPUT_SIZE = 416
    NUM_CLASS = len(CLASSES)
    EPOCHS = args.num_epochs
    BATCH_SIZE = args.batch_size
    IOU_LOSS_THRESH = 0.5
    START_LR = args.initial_lr
    END_LR = 1e-6
    annot_path = args.anno_path
    
    model = createModel(NUM_CLASS, INPUT_SIZE, STRIDES, ANCHORS, XYSCALE)
    
    # model = tf.keras.models.load_model('check_points/yolo/400_best', compile = False)
    
    training_name = 'yolo_server_room'
    logger = logging.getLogger('train_log')
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    start_time = time.time()
    
    
    model_save_path = "{}/log/{}/{:.0f}/".format(check_point_path,training_name, start_time)

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
        
    logger.handlers.clear()
    logging_path = "{}train.log".format(model_save_path)
    fh = logging.FileHandler(filename=logging_path)
    fh.setLevel(logging.INFO)
    # logger.addHandler(ch)
    logger.addHandler(fh)

    logger.info('logging file path : {}'.format(logging_path))
    
    trainset = Dataset(annot_path, INPUT_SIZE, BATCH_SIZE,  CLASSES, ANCHORS, ANCHOR_PER_SCALE, STRIDES, data_aug=True, is_training=True)
    logger.info('dataset loaded : {}'.format(len(trainset)))

    progress = tqdm(range(EPOCHS))
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    total_steps = len(trainset) * EPOCHS

    elapsed = 0.
    pre_total_loss= tf.constant(0.)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=START_LR)

    for epoch in progress:

        total_giou_loss = 0.
        total_conf_loss = 0.
        total_prob_loss = 0.
        total_loss = 0.

        start_time = time.time()

        # progress = tqdm(trainset)
        for image_data, target in trainset:
            with tf.GradientTape() as tape:
                pred_result = model(image_data, training=True)
                giou_loss = conf_loss = prob_loss = 0

                # optimizing process
                for i in range(2):
                    conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                    loss_items = compute_loss(pred, conv, target[i][0], target[i][1], STRIDES=STRIDES, NUM_CLASS=NUM_CLASS, IOU_LOSS_THRESH=IOU_LOSS_THRESH, i=i)
                    giou_loss += loss_items[0]
                    conf_loss += loss_items[1]
                    prob_loss += loss_items[2]

                step_loss = giou_loss + conf_loss + prob_loss

                gradients = tape.gradient(step_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                global_steps.assign_add(1)
                lr = END_LR + 0.5 * (START_LR - END_LR) * (
                        (1 + tf.cos((global_steps) / (total_steps) * np.pi))
                    )
                optimizer.lr.assign(lr.numpy())

                total_loss += step_loss

                total_giou_loss += giou_loss
                total_conf_loss += conf_loss
                total_prob_loss += prob_loss

                progress.set_postfix({'pre_elapsed': elapsed, 'pre_total_loss': pre_total_loss.numpy(), 'cur step loss' : step_loss.numpy(), 'lr': lr.numpy()})

        pre_total_loss = total_loss / len(trainset)
        total_conf_loss = total_conf_loss / len(trainset)
        total_giou_loss = total_giou_loss / len(trainset)
        total_prob_loss = total_prob_loss / len(trainset)

        elapsed = time.time()- start_time


        if(epoch % 10) == 0 and epoch > 0:
            log = 'trn {:d}: elapsed {:.2f}s, iou loss {:.4f}, conf loss {:.4f}, prob loss {:.4f}, total loss {:.4f}, lr {:.6f}'.format(
                    epoch, 
                    elapsed, 
                    total_giou_loss.numpy(), 
                    total_conf_loss.numpy(), 
                    total_prob_loss.numpy(), 
                    pre_total_loss.numpy(),
                    lr.numpy())
            logger.info(log)
            model.save("{}/yolo_epoch_{}.h5".format(check_point_path, epoch))

    model.save("{}/yolo_epoch_latest.h5".format(check_point_path))