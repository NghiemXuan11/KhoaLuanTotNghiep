import os
from utils.common import get_test_loader
import torch
import tqdm
import json
import numpy as np

def run_test(net, cfg):
    loader = get_test_loader(cfg)
    preds = []
    seg_labels = []
    for data in tqdm.tqdm(loader):
        imgs, seg_labels_tmp = data['images'], data['seg_labels']
        imgs = imgs.cuda()
        with torch.no_grad():
            pred = net(imgs)
            pred = pred.argmax(dim=1)
            preds.append(pred)

        seg_labels_tmp = seg_labels_tmp.cuda().argmax(dim=1)
        seg_labels.append(seg_labels_tmp)

        preds.append(pred)
    res = {'preds': preds, 'seg_labels': seg_labels}
    return res

def call_tusimple_eval(res, num_lanes):
    TP, FP, FN, TN = 0., 0., 0., 0.

    for pred, seg_label in zip(res['preds'], res['seg_labels']):
        pred = pred != 0
        seg_label = seg_label != 0
        TP += torch.sum(pred & seg_label).item()
        FP += torch.sum(pred & ~seg_label).item()
        FN += torch.sum(~pred & seg_label).item()
        TN += torch.sum(~pred & ~seg_label).item()
    Acc = (TP + TN) / (TP + FP + FN + TN)
    # 1e-6 to avoid division by zero
    Presicion = TP / (TP + FP + 1e-6) 
    Recall = TP / (TP + FN + 1e-6)
    F1 = 2 * Presicion * Recall / (Presicion + Recall + 1e-6) 
    print(Acc, Presicion, Recall, F1)
    res = [{'name': 'Acc', 'value': Acc, 'order': 'asc'},
           {'name': 'Presicion', 'value': Presicion, 'order': 'asc'},
           {'name': 'Recall', 'value': Recall, 'order': 'asc'},
           {'name': 'F1', 'value': F1, 'order': 'asc'}]
    return json.dumps(res)


def eval_lane(net, cfg, epoch, logger):
    net.eval()
    res = run_test(net, cfg)
    res = call_tusimple_eval(res, cfg.num_lanes)
    res = json.loads(res)
    for r in res:
        print(r['name'], r['value'])
        if logger is not None:
            logger.add_scalar('TuEval/'+r['name'],r['value'],global_step = epoch)
    for r in res:
        if r['name'] == 'F1':
            return r['value']