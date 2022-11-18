import argparse
import tensorflow as tf
import os
import sys
import time
import yaml
from tqdm import tqdm

from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from dataset import create_batch_generator
from anchor import generate_default_boxes
from network import create_ssd
from losses import create_losses

@tf.function
def train_step(imgs, gt_confs, gt_locs, ssd, criterion, optimizer):
    with tf.GradientTape() as tape:
        confs, locs = ssd(imgs)

        conf_loss, loc_loss = criterion(
            confs, locs, gt_confs, gt_locs)

        loss = conf_loss + loc_loss
        l2_loss = [tf.nn.l2_loss(t) for t in ssd.trainable_variables]
        l2_loss = args.weight_decay * tf.math.reduce_sum(l2_loss)
        loss += l2_loss

    gradients = tape.gradient(loss, ssd.trainable_variables)
    optimizer.apply_gradients(zip(gradients, ssd.trainable_variables))

    return loss, conf_loss, loc_loss, l2_loss


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--anno-path', default='dataset/server_room/train_digit.txt')
    parser.add_argument('--data-year', default='2007')
    parser.add_argument('--arch', default='ssd300')
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--num-batches', default=-1, type=int)
    parser.add_argument('--neg-ratio', default=3, type=int)
    parser.add_argument('--initial-lr', default=1e-3, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--num-epochs', default=100, type=int)
    parser.add_argument('--checkpoint-dir', default='./check_points/ssd')
    parser.add_argument('--checkpoint-path', default=None) # latest 'check_points/ssd/ssd_epoch_latest.h5'
    parser.add_argument('--pretrained-type', default='base')
    parser.add_argument('--gpu-id', default='0')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    
    labels = ['0','1','2','3','4','5','6','7','8','9', '.']
    
    NUM_CLASSES = len(labels) + 1

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
        args.batch_size, args.num_batches,
        mode='train', augmentation = False,labels = labels)  # the patching algorithm is currently causing bottleneck sometimes   , augmentation=['flip']
    
    try:
        ssd = create_ssd(NUM_CLASSES, args.arch,
                        args.pretrained_type,
                        checkpoint_dir=args.checkpoint_path, checkpoint_path=args.checkpoint_path)
    except Exception as e:
        print(e)
        print('The program is exiting...')
        sys.exit()

    criterion = create_losses(args.neg_ratio, NUM_CLASSES)

    steps_per_epoch = info['length'] // args.batch_size

    lr_fn = PiecewiseConstantDecay(
        boundaries=[int(steps_per_epoch * args.num_epochs * 2 / 3),
                    int(steps_per_epoch * args.num_epochs * 5 / 6)],
        values=[args.initial_lr, args.initial_lr * 0.1, args.initial_lr * 0.01])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_fn)

    train_log_dir = './check_points/ssd/logs/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    for epoch in range(args.num_epochs):
        avg_loss = 0.0
        avg_conf_loss = 0.0
        avg_loc_loss = 0.0
        start = time.time()
        i = 0
        progress = tqdm(batch_generator)
        for _, imgs, gt_confs, gt_locs in progress:
            loss, conf_loss, loc_loss, l2_loss = train_step(imgs, gt_confs, gt_locs, ssd, criterion, optimizer)
            
            avg_loss = (avg_loss * i + loss.numpy()) / (i + 1)
            avg_conf_loss = (avg_conf_loss * i + conf_loss.numpy()) / (i + 1)
            avg_loc_loss = (avg_loc_loss * i + loc_loss.numpy()) / (i + 1)
            
            progress.set_description('Epoch: {} Batch {} Time: {:.2}s | Loss: {:.4f} Conf: {:.4f} Loc: {:.4f}'.format(
                epoch + 1, i + 1, time.time() - start, avg_loss, avg_conf_loss, avg_loc_loss))
            
            i = i + 1
        
        # avg_val_loss = 0.0
        # avg_val_conf_loss = 0.0
        # avg_val_loc_loss = 0.0
        # for i, (_, imgs, gt_confs, gt_locs) in enumerate(val_generator):
        #     val_confs, val_locs = ssd(imgs)
        #     val_conf_loss, val_loc_loss = criterion(val_confs, val_locs, gt_confs, gt_locs)
        #     val_loss = val_conf_loss + val_loc_loss
        #     avg_val_loss = (avg_val_loss * i + val_loss.numpy()) / (i + 1)
        #     avg_val_conf_loss = (avg_val_conf_loss * i + val_conf_loss.numpy()) / (i + 1)
        #     avg_val_loc_loss = (avg_val_loc_loss * i + val_loc_loss.numpy()) / (i + 1)

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', avg_loss, step=epoch)
            tf.summary.scalar('conf_loss', avg_conf_loss, step=epoch)
            tf.summary.scalar('loc_loss', avg_loc_loss, step=epoch)

        if (epoch + 1) % 10 == 0:
            ssd.save_weights(os.path.join(args.checkpoint_dir, 'ssd_epoch_{}.h5'.format(epoch + 1)))
            
    if (epoch + 1) % 10 == 0:
            ssd.save_weights(os.path.join(args.checkpoint_dir, 'ssd_epoch_latest.h5'))
