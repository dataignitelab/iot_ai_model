import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms 

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']

class ImageDataset(Dataset):
    def __init__(self, dataset_path, label_names = ['normal', 'defect'], tranform = None):
        self.dataset_path = dataset_path
        self.file_path = []
        self.labels = []
        self.data = []
        self.label_idx = []
        self.label_map = {}
        self.transform = tranform

        for idx, cls in enumerate(label_names):
            self.label_map[cls] = idx

        _, self.file_path, self.labels = self._getfile_list(dataset_path)

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

    def __getitem__(self, index):
        img = Image.open(self.file_path[index])
        assert img is not None, 'Image not found {}'.format(self.file_path[index])

        ##img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        return self.file_path[index], img, self.label_idx[index]
