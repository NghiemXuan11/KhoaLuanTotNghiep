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
        pred = torch.argmax(pred, dim=1)
        pred = pred.flatten()
        gt = torch.argmax(gt, dim=1)
        gt = gt.flatten()
        self.correct_points += torch.sum(pred == gt).item()
        self.total_points += len(gt)
    def get(self):
        return self.correct_points / (self.total_points + 1e-6)


class F1Metric:
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
    def update(self, pred, gt):
        num_classes = pred.shape[1]
        pred = torch.argmax(pred, dim=1)
        pred = pred.flatten()
        gt = torch.argmax(gt, dim=1)
        gt = gt.flatten()
        for i in range(1, num_classes):
            self.tp += torch.sum((pred == i) & (gt == i)).item()
            self.fp += torch.sum((pred == i) & (gt != i)).item()
            self.fn += torch.sum((pred != i) & (gt == i)).item()
    def get(self):
        return 2 * self.tp / (2 * self.tp + self.fp + self.fn + 1e-6)

def update_metrics(metric_dict, pair_data):
    for i in range(len(metric_dict['name'])):
        metric_op = metric_dict['op'][i]
        data_src = metric_dict['data_src'][i]
        metric_op.update(pair_data[data_src[0]], pair_data[data_src[1]])


def reset_metrics(metric_dict):
    for op in metric_dict['op']:
        op.reset()
