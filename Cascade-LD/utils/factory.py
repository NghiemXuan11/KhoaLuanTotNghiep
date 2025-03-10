import torch
from utils.metrics import AccuracyMetric, F1Metric
from utils.loss import BinaryCrossEntropy, InstanceLoss, FocalLoss


def get_optimizer(net,cfg):
    training_params = filter(lambda p: p.requires_grad, net.parameters())
    if cfg.optimizer == 'Adam':
        optimizer = torch.optim.Adam(training_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'SGD':
        optimizer = torch.optim.SGD(training_params, lr=cfg.learning_rate, momentum=cfg.momentum,
                                    weight_decay=cfg.weight_decay)
    else:
        raise NotImplementedError
    return optimizer

def get_scheduler(optimizer, cfg, iters_per_epoch):
    if cfg.scheduler == 'multi':
        scheduler = MultiStepLR(optimizer, cfg.steps, cfg.gamma, iters_per_epoch, cfg.warmup, iters_per_epoch if cfg.warmup_iters is None else cfg.warmup_iters)
    elif cfg.scheduler == 'cos':
        scheduler = CosineAnnealingLR(optimizer, cfg.epoch * iters_per_epoch, eta_min = 0, warmup = cfg.warmup, warmup_iters = cfg.warmup_iters)
    else:
        raise NotImplementedError
    return scheduler

def get_loss_dict(cfg):
    loss_dict = {
        'name': ['cls_ext', 'ins_ext', 'focus'],
        'weight': [1.0, 1.0, 1.0], # loss weight
        'op': [BinaryCrossEntropy(weight=torch.tensor(cfg.weight_cls).cuda()), InstanceLoss(), FocalLoss(alpha=torch.tensor(cfg.weight_cls).cuda(), reduction='mean')],
        'data_src': [('seg_preds', 'seg_labels'), ('seg_preds', 'seg_labels'), ('seg_preds', 'seg_labels')],
    }
    
    assert len(loss_dict['name']) == len(loss_dict['op']) == len(loss_dict['data_src']) == len(loss_dict['weight'])
    return loss_dict

def get_metric_dict(cfg):

    metric_dict = {
        'name': ['acc'],#'f1'],
        'op': [AccuracyMetric()],# F1Metric()],
        'data_src': [('seg_preds', 'seg_labels')],# ('seg_preds', 'seg_labels')],
    }

    assert len(metric_dict['name']) == len(metric_dict['op']) == len(metric_dict['data_src'])
    return metric_dict


class MultiStepLR:
    def __init__(self, optimizer, steps, gamma = 0.1, iters_per_epoch = None, warmup = None, warmup_iters = None):
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.optimizer = optimizer
        self.steps = steps
        self.steps.sort()
        self.gamma = gamma
        self.iters_per_epoch = iters_per_epoch
        self.iters = 0
        self.base_lr = [group['lr'] for group in optimizer.param_groups]

    def step(self, external_iter = None):
        self.iters += 1
        if external_iter is not None:
            self.iters = external_iter
        if self.warmup == 'linear' and self.iters < self.warmup_iters:
            rate = self.iters / self.warmup_iters
            for group, lr in zip(self.optimizer.param_groups, self.base_lr):
                group['lr'] = lr * rate
            return

        # multi policy
        if self.iters % self.iters_per_epoch == 0:
            epoch = int(self.iters / self.iters_per_epoch)
            power = -1
            for i, st in enumerate(self.steps):
                if epoch < st:
                    power = i
                    break
            if power == -1:
                power = len(self.steps)
            # print(self.iters, self.iters_per_epoch, self.steps, power)

            for group, lr in zip(self.optimizer.param_groups, self.base_lr):
                group['lr'] = lr * (self.gamma ** power)
import math
class CosineAnnealingLR:
    def __init__(self, optimizer, T_max , eta_min = 0, warmup = None, warmup_iters = None):
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min

        self.iters = 0
        self.base_lr = [group['lr'] for group in optimizer.param_groups]

    def step(self, external_iter = None):
        self.iters += 1
        if external_iter is not None:
            self.iters = external_iter
        if self.warmup == 'linear' and self.iters < self.warmup_iters:
            rate = self.iters / self.warmup_iters
            for group, lr in zip(self.optimizer.param_groups, self.base_lr):
                group['lr'] = lr * rate
            return

        # cos policy

        for group, lr in zip(self.optimizer.param_groups, self.base_lr):
            group['lr'] = self.eta_min + (lr - self.eta_min) * (1 + math.cos(math.pi * self.iters / self.T_max)) / 2

