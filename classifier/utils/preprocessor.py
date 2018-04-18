from __future__ import absolute_import
import os.path as osp

from PIL import Image
import torch
import os
import numpy as np


class Preprocessor(object):
    def __init__(self, dataset, images_dir, default_size, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.images_dir = images_dir
        self.transform = transform
        self.default_size = default_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        img_name, bin_label = self.dataset[index]
        bin_label = np.array(bin_label).astype(np.float32)
        fpath = osp.join(self.images_dir, img_name)
        if os.path.isfile(fpath):
            img = Image.open(fpath).convert('RGB')
        else:
            img = Image.new('RGB', self.default_size)
            print('No such images: {:s}'.format(fpath))
        if self.transform is not None:
            img = self.transform(img)
        return img, img_name, bin_label
