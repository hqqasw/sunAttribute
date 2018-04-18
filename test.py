from __future__ import print_function, absolute_import
import argparse
import os.path as osp

import numpy as np
import sys
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboard import SummaryWriter

from classifier.dataset import get_dataset
from classifier.utils import save_checkpoint, load_checkpoint
from classifier.utils import Logger, mkdir_if_missing
from classifier.utils import transforms
from classifier.utils import Preprocessor
from classifier.models import ResNet, VGGNet
from classifier.evaluator import Evaluator


def get_data(
        dataset_name, data_dir,
        crop_w, crop_h,
        batch_size, workers):

    # read dataset
    dataset = get_dataset(dataset_name, data_dir)

    # data transforms
    normalizer = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    test_transformer = transforms.Compose([
        transforms.Resize(size=(crop_h, crop_w)),
        transforms.ToTensor(),
        normalizer,
    ])

    # data loaders
    num_class = dataset.num_class
    test_loader = DataLoader(
        Preprocessor(dataset.test, dataset.images_dir, default_size=(crop_w, crop_h), transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, num_class, test_loader


def main(args):

    mkdir_if_missing(args.logs_dir)
    writer = SummaryWriter(args.logs_dir)

    sys.stdout = Logger(osp.join(args.logs_dir, 'test_log.txt'))
    print(args)

    cudnn.benchmark = True

    # create data loaders
    data_dir = args.data_dir
    dataset, num_class, test_loader = \
        get_data(
            args.dataset, data_dir,
            args.crop_w, args.crop_h,
            args.batch_size, args.workers)

    # create model
    model = VGGNet(
        args.depth, with_bn=True, pretrained=True, num_class=num_class,
        input_size=(args.crop_w, args.crop_h))
    model = model.cuda()

    # load from checkpoint
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    args.start_epoch = checkpoint['epoch']
    best_recall5 = checkpoint['best_recall5']
    print("=> get epoch {}  best top5 recall {:.1%}".format(args.start_epoch, best_recall5))

    # create trainer
    evaluator = Evaluator(model)

    # test
    print('Test with best model:')
    evaluator.test(test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='sun',
                        choices=['sun'])
    parser.add_argument('-b', '--batch-size', type=int, default=16)
    parser.add_argument('-j', '--workers', type=int, default=12)
    # data
    parser.add_argument('--crop_w', type=int, default=128)
    parser.add_argument('--crop_h', type=int, default=128)
    # model
    parser.add_argument('--depth', type=int, default=16,
                        choices=[16, 19])
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))

    args = parser.parse_args()
    main(args)
