from __future__ import print_function, absolute_import
import time
from collections import OrderedDict

import torch

from .utils import AverageMeter
from torch.autograd import Variable
import numpy as np
from .utils import get_accuracy


class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def get_results(self, data_loader, print_freq=10):
        self.model.eval()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        results = OrderedDict()
        labels = OrderedDict()

        end = time.time()
        for i, (imgs, img_names, targets) in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs = Variable(imgs).cuda()
            outputs = self.model(inputs)

            for img_name, result, target in zip(img_names, outputs, targets):
                results[img_name] = result.cpu().data
                labels[img_name] = target

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print(
                    'Extract Features: [{}/{}]\t'
                    'Time {:.3f} ({:.3f})\t'
                    'Data {:.3f} ({:.3f})\t'
                    .format(
                        i + 1, len(data_loader),
                        batch_time.val, batch_time.avg,
                        data_time.val, data_time.avg))

        return results, labels

    def evaluate(self, data_loader):

        results, labels = self.get_results(data_loader)

        label_mat = torch.cat([labels[key].unsqueeze(0) for key in labels.keys()], 0)
        result_mat = torch.cat([results[key].unsqueeze(0) for key in labels.keys()], 0)

        prec, recall = get_accuracy(result_mat, label_mat)
        top5_prec = prec[2]
        top5_recall = recall[2]

        print('Top 5 Prec {:.2%} \t'
              'Top 5 Recall {:.2%} \t'
              .format(top5_prec, top5_recall))

        return top5_prec, top5_recall

    def test(self, data_loader, topk=(1, 3, 5)):
        results, labels = self.get_results(data_loader)

        label_mat = torch.cat([labels[key].unsqueeze(0) for key in labels.keys()], 0)
        result_mat = torch.cat([results[key].unsqueeze(0) for key in labels.keys()], 0)

        prec, recall = get_accuracy(result_mat, label_mat)

        print('  Prec: \t Top_1: {:.2%} \t Top_3: {:.2%} \t Top_5: {:.2%}\n'
              'Recall: \t Top_1: {:.2%} \t Top_3: {:.2%} \t Top_5: {:.2%}\n'
              .format(prec[0], prec[1], prec[2], recall[0], recall[1], recall[2]))

        return prec, recall
