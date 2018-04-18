from __future__ import print_function, absolute_import
import os.path as osp

from ..utils.dataset import Dataset
from ..utils.basic import str2list


class SUN(Dataset):

    def __init__(self, root):
        super(SUN, self).__init__(root)
        self.train_file = osp.join(self.root, 'label', 'train.txt')
        self.test_file = osp.join(self.root, 'label', 'test.txt')
        self.load()
        self.print_summary()

    @property
    def images_dir(self):
        return osp.join(self.root, 'images')

    @property
    def num_class(self):
        return 102

    def read_meta(self, meta_file):
        with open(meta_file) as f:
            lines = f.readlines()
        data_list = []
        for line in lines:
            words = line.strip().split()
            img_name = words[0].strip()
            label_list = str2list(words[1].strip())
            bin_label = [0 for x in range(self.num_class)]
            for x in label_list:
                bin_label[x] = 1
            data_list.append((img_name, bin_label))
        return data_list

    def load(self):
        self.train = self.read_meta(self.train_file)
        self.test = self.read_meta(self.test_file)
