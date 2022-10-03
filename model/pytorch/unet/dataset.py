from glob import glob
from tqdm import tqdm
import numpy as np
import os
import cv2

from torch.utils.data import Dataset

def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img = img.astype(np.float32) / 255.
    img = np.transpose(img, (2, 0, 1))
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
    
    def __init__(self, dataset_dir):
        path, images = load_data(dataset_dir, 'images') # , trim=1100
        mask_path, masks = load_data(dataset_dir, 'masks')
        self.images = images
        self.masks = masks
        self.path = path
        self.mask_path = mask_path

    def __len__(self):
        return len(self.path)    

    def __getitem__(self, index):
        return self.path[index], self.images[index], self.masks[index]
