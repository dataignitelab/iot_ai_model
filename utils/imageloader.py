import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms 

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']

class ImageDataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.file_path = []
        self.labels = []
        self.data = []
        self.label_to_idx = []
        self.labels_map = {}

        _, self.file_path, self.labels = self._getfile_list(dataset_path)

        s = set(self.labels)
        for index, key in enumerate(s):
            self.labels_map[key] = index

        self.label_to_idx = [ float(self.labels_map[label]) for label in self.labels ]

        self.data_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.labels)    

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
        img = self.data_transform(img)
        return img, self.label_to_idx[index]

