from __future__ import print_function
import os.path as osp

import numpy as np

from .serialization import read_json


class Dataset(object):
    def __init__(self, root):
        self.root = root
        self.train, self.val, self.test = [], [], []

    def print_summary(self):
        print(self.__class__.__name__, "dataset loaded")
        print('     subset  | # samples')
        print('---------------------------------')
        print('    train    | {:8d}'.format(len(self.train)))
        print('    val      | {:8d}'.format(len(self.val)))
        print('    test     | {:8d}'.format(len(self.test)))
