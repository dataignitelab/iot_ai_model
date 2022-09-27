import tensorflow as tf
import os
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
import random

from box_utils import compute_target
from image_utils import random_resize, random_translate
from functools import partial


class VOCDataset():
    """ Class for VOC Dataset

    Attributes:
        root_dir: dataset root dir (ex: ./data/VOCdevkit)
        year: dataset's year (2007 or 2012)
        num_examples: number of examples to be used
                      (in case one wants to overfit small data)
    """

    def __init__(self, data_anno_path, default_boxes,
                 new_size, num_examples=-1, augmentation=True):
        super(VOCDataset, self).__init__()
        self.idx_to_name = ['0','1','2','3','4','5','6','7','8','9']
        self.name_to_idx = dict([(v, k) for k, v in enumerate(self.idx_to_name)])
        self.base_dir = os.path.dirname(data_anno_path) 
        self.image_path = []
        self.bbox = []
        with open(data_anno_path ,'r') as f:
            for row in f.readlines():
                col = row.split()
                self.image_path.append(os.path.join(self.base_dir, col[0]))
                self.bbox.append(col[1:])

        # self.data_dir = os.path.join(root_dir, 'VOC{}'.format(year))
        
        # self.image_dir = os.path.join(self.data_dir, 'JPEGImages')
        # self.anno_dir = os.path.join(self.data_dir, 'Annotations')
        self.ids = self.image_path

        self.default_boxes = default_boxes
        self.new_size = new_size

        if num_examples != -1:
            self.ids = self.ids[:num_examples]

        self.train_ids = self.ids[:int(len(self.ids) * 1)]
        self.val_ids = self.ids[int(len(self.ids) * 0.75):]

        # if augmentation == None:
        #     self.augmentation = ['original']
        # else:
        #     self.augmentation = augmentation + ['original']
        self.augmentation = augmentation

    def __len__(self):
        return len(self.ids)

    def _get_image(self, index):
        """ Method to read image from file
            then resize to (300, 300)
            then subtract by ImageNet's mean
            then convert to Tensor

        Args:
            index: the index to get filename from self.ids

        Returns:
            img: tensor of shape (3, 300, 300)
        """
        # filename = self.ids[index]
        img_path = self.image_path[index]
        img = Image.open(img_path)

        return img

    def _get_annotation(self, index, orig_shape):
        """ Method to read annotation from file
            Boxes are normalized to image size
            Integer labels are increased by 1

        Args:
            index: the index to get filename from self.ids
            orig_shape: image's original shape

        Returns:
            boxes: numpy array of shape (num_gt, 4)
            labels: numpy array of shape (num_gt,)
        """
        h, w = orig_shape
        # filename = self.ids[index]
        # anno_path = os.path.join(self.anno_dir, filename + '.xml')
        # objects = ET.parse(anno_path).findall('object')
        boxes = []
        labels = []

        objects = self.bbox[index]
        for obj in objects:
            v = obj.split(',')
            name_idx = float(v[4])
            xmin = float(v[0]) / w
            ymin = float(v[1]) / h
            xmax = float(v[2]) / w
            ymax = float(v[3]) / h
            boxes.append([xmin, ymin, xmax, ymax])

            labels.append(name_idx + 1)

        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)

    def generate(self, subset=None):
        """ The __getitem__ method
            so that the object can be iterable

        Args:
            index: the index to get filename from self.ids

        Returns:
            img: tensor of shape (300, 300, 3)
            boxes: tensor of shape (num_gt, 4)
            labels: tensor of shape (num_gt,)
        """
        if subset == 'train':
            indices = self.train_ids
        elif subset == 'val':
            indices = self.val_ids
        else:
            indices = self.ids
        for index in range(len(indices)):
            # img, orig_shape = self._get_image(index)
            filename = indices[index]
            img = self._get_image(index)
            w, h = img.size
            boxes, labels = self._get_annotation(index, (h, w))

            if self.augmentation :
                if random.random() < 0.5:
                    img, boxes = random_resize(img, boxes)
                if random.random() < 0.5:
                    img, boxes = random_translate(img, boxes)
            
            # augmentation_method = np.random.choice(self.augmentation)
            # if augmentation_method == 'patch':
            #     img, boxes, labels = random_patching(img, boxes, labels)
            # elif augmentation_method == 'flip':
            #     img, boxes, labels = horizontal_flip(img, boxes, labels)
            # elif(
            
            boxes = tf.constant(boxes, dtype=tf.float32)
            labels = tf.constant(labels, dtype=tf.int64)

            img = np.array(img.resize(
                (self.new_size, self.new_size)), dtype=np.float32)
            img = (img / 127.0) - 1.0
            img = tf.constant(img, dtype=tf.float32)

            # print(boxes, labels)
            
            gt_confs, gt_locs = compute_target(self.default_boxes, boxes, labels)
            
            # print(gt_confs[:,)
            
            yield filename, img, gt_confs, gt_locs


def create_batch_generator(data_anno_path, default_boxes,
                           new_size, batch_size, num_batches,
                           mode,
                           augmentation=True):
    num_examples = batch_size * num_batches if num_batches > 0 else -1
    voc = VOCDataset(data_anno_path, default_boxes,
                     new_size, num_examples, augmentation)

    info = {
        'idx_to_name': voc.idx_to_name,
        'name_to_idx': voc.name_to_idx,
        'length': len(voc)
        # 'image_dir': voc.image_dir,
        # 'anno_dir': voc.anno_dir
    }

    if mode == 'train':
        train_gen = partial(voc.generate, subset='train')
        train_dataset = tf.data.Dataset.from_generator(
            train_gen, (tf.string, tf.float32, tf.int64, tf.float32))
        val_gen = partial(voc.generate, subset='val')
        val_dataset = tf.data.Dataset.from_generator(
            val_gen, (tf.string, tf.float32, tf.int64, tf.float32))

        train_dataset = train_dataset.shuffle(40).batch(batch_size)
        val_dataset = val_dataset.batch(batch_size)

        return train_dataset.take(num_batches), val_dataset.take(-1), info
    else:
        dataset = tf.data.Dataset.from_generator(
            voc.generate, (tf.string, tf.float32, tf.int64, tf.float32))
        dataset = dataset.batch(batch_size)
        return dataset.take(num_batches), info
