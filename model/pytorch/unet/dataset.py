from glob import glob
from tqdm import tqdm
import numpy as np
import os
import cv2
import random

import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import transforms

from torch.utils.data import Dataset

from PIL import Image
from io import BytesIO

def load_image(path, use_opencv= True):
    if use_opencv:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        img = np.transpose(img, (2, 0, 1))
        img = img.astype(np.float32) / 255.
    else:
        img = Image.open(BytesIO(buf))
        
    return img

def load_data(root_path, name='images', trim=None):
    paths = glob(os.path.join(root_path, name, "*.png"))
    if trim is not None:
        paths = paths[:trim]
    images = np.zeros(shape=(len(paths), 3, 256, 256)).astype(np.float32)
    list_path = []
    for i, path in tqdm(enumerate(paths), desc=f"{name} Loading"):
        images[i] = load_image(path)
        list_path.append(path)
        
    return list_path, images

class ImageDataset(Dataset):
    
    def __init__(self, dataset_dir, augmentation = True, preload = True):
        self.preload = preload
        if preload:
            path, images = load_data(dataset_dir, 'images') # , trim=1100
            mask_path, masks = load_data(dataset_dir, 'masks')
            self.images = images
            self.masks = masks
            self.path = path
            self.mask_path = mask_path
        else:
            self.path = glob(os.path.join(dataset_dir, 'images', "*.png"))
            self.mask_path = glob(os.path.join(dataset_dir, 'masks', "*.png"))
            
        self.augmentation = augmentation

    def __len__(self):
        return len(self.path)    

    def __getitem__(self, index):
        if self.preload:
            img = torch.tensor(self.images[index])
            mask = torch.tensor(self.masks[index])
        else:
            img = load_image(self.path[index])
            mask = load_image(self.mask_path[index])

        if self.augmentation:
            if (np.random.random() > 0.5):
                img = TF.hflip(img)
                mask = TF.hflip(mask)

            if (np.random.random() > 0.5):
                img = TF.vflip(img)
                mask = TF.vflip(mask)

            if np.random.random() > 0.5:
                angle = np.random.randint(-30, 30)
                img = TF.rotate(img, angle)
                mask = TF.rotate(mask, angle)
        
        return self.path[index], img, mask
