from __future__ import absolute_import

import numpy as np


def get_accuracy(output, target, topk=(1, 3, 5)):
    target = target.cpu().numpy()
    output = output.cpu().numpy()
    target_sum = target.sum(axis=-1)
    batch_size, num_class = target.shape
    idx = np.argsort(-output)
    precision_list = []
    recall_list = []
    for k in topk:
        result_mask = np.zeros((batch_size, num_class))
        for i in range(k):
            result_mask[[x for x in range(batch_size)], idx[:, i]] = 1
        hit = np.logical_and(target, result_mask).sum(axis=-1)
        precision = (hit / k).mean()
        recall = (hit / target_sum).mean()
        precision_list.append(precision)
        recall_list.append(recall)

    return precision_list, recall_list
