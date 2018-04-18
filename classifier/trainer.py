from __future__ import print_function, absolute_import
import time

import torch
from torch.autograd import Variable

from .utils import get_accuracy
from .utils import AverageMeter


class BaseTrainer(object):
    def __init__(self, model, criterion):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion

    def train(self, epoch, data_loader, optimizer, print_freq=10):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()

        end = time.time()
        iter_count = len(data_loader)
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            loss, prec3, recall3 = self._forward(*self._parse_data(inputs))

            losses.update(loss.data[0])
            precisions.update(prec3)
            recalls.update(recall3)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      'Recall {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, iter_count,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg,
                              recalls.val, recalls.avg))

        return losses.avg, precisions.avg, recalls.avg

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, _, labels = inputs
        img_inputs = Variable(imgs).cuda()
        targets = Variable(labels.cuda())
        return img_inputs, targets

    def _forward(self, inputs, targets):
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        prec, recall = get_accuracy(outputs.data, targets.data)
        top5_prec = prec[2]
        top5_recall = recall[2]
        return loss, top5_prec, top5_recall
