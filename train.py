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
from classifier.trainer import Trainer
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
    train_transformer = transforms.Compose([
        transforms.RandomSizedRectCrop(crop_h, crop_w),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalizer,
    ])
    val_transformer = transforms.Compose([
        transforms.Resize(size=(crop_h, crop_w)),
        transforms.ToTensor(),
        normalizer,
    ])

    # data loaders
    num_class = dataset.num_class
    train_loader = DataLoader(
        Preprocessor(dataset.train, dataset.images_dir, default_size=(crop_w, crop_h), transform=train_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=True, pin_memory=True, drop_last=True)

    val_loader = DataLoader(
        Preprocessor(dataset.train, dataset.images_dir, default_size=(crop_w, crop_h), transform=val_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, num_class, train_loader, val_loader


def main(args):

    mkdir_if_missing(args.logs_dir)
    writer = SummaryWriter(args.logs_dir)

    sys.stdout = Logger(osp.join(args.logs_dir, 'train_log.txt'))
    print(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

    # create data loaders
    data_dir = args.data_dir
    dataset, num_class, train_loader, val_loader = \
        get_data(
            args.dataset, data_dir,
            args.crop_w, args.crop_h,
            args.batch_size, args.workers)

    # create model
    model = VGGNet(
        args.depth, with_bn=True, pretrained=True, num_class=num_class, dropout=args.dropout,
        input_size=(args.crop_w, args.crop_h))
    model = model.cuda()

    # load from checkpoint
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        args.start_epoch = checkpoint['epoch']
        best_recall5 = checkpoint['best_recall5']
        print("=> start epoch {}  best top5 recall {:.1%}"
              .format(args.start_epoch, best_recall5))
    else:
        best_recall5 = 0

    # criterion
    criterion = torch.nn.BCEWithLogitsLoss()
    criterion.cuda()

    # optimizer
    if args.optimizer == 'sgd':
        param_groups = model.parameters()
        base_param_ids = set(map(id, model.base.parameters()))
        new_params = [p for p in model.parameters() if
                      id(p) not in base_param_ids]
        param_groups = [
            {'params': model.base.parameters(), 'lr_mult': 0.1},
            {'params': new_params, 'lr_mult': 1.0}]
        optimizer = torch.optim.SGD(param_groups, lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    nesterov=True)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
    else:
        raise ValueError("Cannot recognize optimizer type:", args.optimizer)

    # create trainer and evaluator
    trainer = Trainer(model, criterion)
    evaluator = Evaluator(model)

    # Schedule learning rate
    def adjust_lr(epoch):
        if args.optimizer == 'sgd':
            lr = args.lr * (0.1 ** (epoch // 30))
        elif args.optimizer == 'adam':
            lr = args.lr if epoch <= 50 else \
                args.lr * (0.01 ** (epoch - 50) / 30)
        else:
            raise ValueError("Cannot recognize optimizer type:", args.optimizer)
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    # start training
    val_prec, val_recall = evaluator.evaluate(val_loader)
    for epoch in range(args.start_epoch, args.epochs):
        adjust_lr(epoch)
        loss, prec, recall = trainer.train(epoch, train_loader, optimizer)
        writer.add_scalar('Train loss', loss, epoch+1)
        writer.add_scalar('Train prec', prec, epoch+1)
        writer.add_scalar('Train recall', recall, epoch+1)

        val_prec, val_recall = evaluator.evaluate(val_loader)
        writer.add_scalar('Val prec', val_prec, epoch+1)
        writer.add_scalar('Val recall', val_recall, epoch+1)

        is_best = val_recall > best_recall5
        best_recall5 = max(val_recall, best_recall5)
        save_checkpoint({
            'state_dict': model.state_dict(),
            'epoch': epoch + 1,
            'best_recall5': best_recall5,
        }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

        print('\n * Finished epoch {:3d}  top5 recall: {:5.1%}  best: {:5.1%}{}\n'.
              format(epoch, val_recall, best_recall5, ' *' if is_best else ''))

    # final test
    print('Test with best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(val_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='sun',
                        choices=['sun'])
    parser.add_argument('-b', '--batch-size', type=int, default=32)
    parser.add_argument('-j', '--workers', type=int, default=12)
    # data
    parser.add_argument('--crop_w', type=int, default=128)
    parser.add_argument('--crop_h', type=int, default=128)
    # model
    parser.add_argument('--depth', type=int, default=16,
                        choices=[16, 19])
    parser.add_argument('--dropout', type=float, default=0.5)
    # optimizer
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['sgd', 'adam'])
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))

    args = parser.parse_args()
    main(args)
