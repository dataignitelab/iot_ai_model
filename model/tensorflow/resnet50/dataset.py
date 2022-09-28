#! /usr/bin/env python
# coding=utf-8

import os
import numpy as np
import tensorflow as tf
from PIL import Image
# from model.tensorflow.yolo.config import cfg

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']

class Dataset():
    """implement Dataset here"""

    def __init__(self, data_path, label_names = ['defect', 'normal']):
        self.data_path = data_path
        self.new_size = 224
        self.label_map = {}
        
        for idx, cls in enumerate(label_names):
            self.label_map[cls] = idx

        _, self.file_path, self.labels = self._getfile_list(data_path)
        self.label_idx = [ float(self.label_map[label]) for label in self.labels ]

    def __len__(self):
        return len(self.file_path)
    
    def _getfile_list(self, path, parent=None):
        file_paths = []
        file_names = []
        labels = []
        for filename in os.listdir(path):
            fullpath = os.path.join(path, filename)
            format = filename.split('.')[-1].lower()
            if os.path.isfile(fullpath):
                if (parent is not None) and (format in img_formats):
                    file_paths.append(fullpath)
                    file_names.append(filename)
                    labels.append(parent)
            else:
                f, p, l = self._getfile_list(fullpath, filename)
                file_names += f
                file_paths += p
                labels += l
        return file_names, file_paths, labels
    
    def generate(self):
        """ The __getitem__ method
            so that the object can be iterable

        Args:
            index: the index to get filename from self.ids

        Returns:
            img: tensor of shape (300, 300, 3)
            boxes: tensor of shape (num_gt, 4)
            labels: tensor of shape (num_gt,)
        """
        img_preprocessing = tf.keras.Sequential(
          [
            tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
            tf.keras.layers.experimental.preprocessing.Normalization(mean=[0.485, 0.456, 0.406], variance=[0.229, 0.224, 0.225])
          ]
        )
        for index in range(len(self.file_path)):
            org_img = Image.open(self.file_path[index])
            img = np.array(org_img.resize((self.new_size, self.new_size)), dtype=np.float32)
            
            img = tf.constant(img, dtype=tf.float32)
            img = img_preprocessing(img)
            
            yield self.file_path[index], org_img, img, self.label_idx[index]
            

def create_batch_generator(data_path, batch_size = 1):
    dataset = Dataset(data_path)
    gen = tf.data.Dataset.from_generator(dataset.generate, (tf.string, tf.float32, tf.float32))
    gen = gen.batch(batch_size)
    return gen.take(-1), len(dataset)