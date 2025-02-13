import numpy as np
import torch
import time,pdb

def converter(data):
    if isinstance(data,torch.Tensor):
        data = data.cpu().data.numpy().flatten()
    return data.flatten()
def fast_hist(label_pred, label_true,num_classes):
    #pdb.set_trace()
    hist = np.bincount(num_classes * label_true.astype(int) + label_pred, minlength=num_classes ** 2)
    hist = hist.reshape(num_classes, num_classes)
    return hist

class AccuracyMetric:
    def __init__(self):
        self.correct_points = 0
        self.total_points = 0

    def reset(self):
        self.correct_points = 0
        self.total_points = 0

    def update(self, pred, gt):
        pred = pred.cpu().data.numpy()
        pred = np.argmax(pred, axis=1)
        pred = converter(pred)
        gt = gt.cpu().data.numpy()
        gt = np.argmax(gt, axis=1)
        gt = converter(gt)
        self.correct_points += np.sum(pred == gt)
        self.total_points += len(gt)
    def get(self):
        return self.correct_points / self.total_points if self.total_points > 0 else 0.0


class FalsePositiveRate:
    def __init__(self):
        self.false_positives = 0
        self.total_predictions = 0

    def reset(self):
        self.false_positives = 0
        self.total_predictions = 0

    def update(self, F_pred, N_pred):
        self.false_positives += F_pred
        self.total_predictions += N_pred

    def get(self):
        return self.false_positives / self.total_predictions if self.total_predictions > 0 else 0.0


class FalseNegativeRate:
    def __init__(self):
        self.missed_predictions = 0
        self.total_ground_truth = 0

    def reset(self):
        self.missed_predictions = 0
        self.total_ground_truth = 0

    def update(self, M_pred, N_gt):
        self.missed_predictions += M_pred
        self.total_ground_truth += N_gt

    def get(self):
        return self.missed_predictions / self.total_ground_truth if self.total_ground_truth > 0 else 0.0


class IoUMetric:
    def __init__(self):
        self.intersections = 0
        self.unions = 0

    def reset(self):
        self.intersections = 0
        self.unions = 0

    def update(self, intersection, union):
        self.intersections += intersection
        self.unions += union

    def get(self):
        return self.intersections / self.unions if self.unions > 0 else 0.0

def update_metrics(metric_dict, pair_data):
    for i in range(len(metric_dict['name'])):
        metric_op = metric_dict['op'][i]
        data_src = metric_dict['data_src'][i]
        metric_op.update(pair_data[data_src[0]], pair_data[data_src[1]])


def reset_metrics(metric_dict):
    for op in metric_dict['op']:
        op.reset()
