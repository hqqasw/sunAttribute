from __future__ import absolute_import

from torchvision.transforms import *
from PIL import Image
import random
import math
import time


class RandomSizedRectCrop(object):
    def __init__(self, height, width, interpolation=Image.BILINEAR, ratio=(0.8, 1.2)):
        self.height = height
        self.width = width
        self.interpolation = interpolation
        self.ratio = ratio

    def __call__(self, img):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.64, 1.0) * area
            if len(self.ratio) == 1:
                aspect_ratio = self.ratio[0]
            else:
                aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))

                img = img.resize((self.width, self.height), self.interpolation)
                return img

        return img.resize((self.width, self.height), self.interpolation)
