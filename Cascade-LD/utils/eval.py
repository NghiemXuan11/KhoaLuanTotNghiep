import os
from utils.common import get_test_loader
import torch
import tqdm
import json
import numpy as np
import  cv2
import torch.nn.functional as F
from data.dali_data import get_image_results


def run_test(net, cfg):
    loader = get_test_loader(cfg)
    preds = []
    seg_labels = []
    imgs = []
    for data in tqdm.tqdm(loader):
        imgs_tmp, seg_labels_tmp = data['images'], data['seg_labels']
        imgs_tmp = imgs_tmp.cuda()
        with torch.no_grad():
            pred = net(imgs_tmp)
            pred = pred.argmax(dim=1)
            preds.append(pred)

        seg_labels_tmp = seg_labels_tmp.cuda().argmax(dim=1)
        seg_labels.append(seg_labels_tmp)
        imgs.append(imgs_tmp.cpu())

        preds.append(pred)
    res = {'preds': preds, 'seg_labels': seg_labels,'imgs':imgs}
    return res

def call_tusimple_eval(res, num_lanes):
    threshold_iou = 0.3
    Accs = []
    FPs = []
    FNs = []
    for preds, seg_labels, imgs in zip(res['preds'], res['seg_labels'], res['imgs']):
        preds_1 = (preds > 0).float()
        seg_labels_1 = (seg_labels > 0).float()
        correct_pixels = (preds_1 == seg_labels_1).sum().item()
        B, H, W = seg_labels_1.shape
        Acc = correct_pixels/(B*H*W)
        
        for pred, seg_label, img in zip(preds, seg_labels, imgs):
            lanes_pred = pred.unique()
            lanes_pred = lanes_pred[lanes_pred!=0]
            lanes_seg_label = seg_label.unique()
            lanes_seg_label = lanes_seg_label[lanes_seg_label!=0]
            lanes_correct_pred = {}
            for lane_id_pred in lanes_pred:
                for lane_id_label in  lanes_seg_label:
                    intersection = torch.logical_and(pred==lane_id_pred, seg_label==lane_id_label).sum().item()
                    union = torch.logical_or(pred==lane_id_pred, seg_label==lane_id_label).sum().item()
                    iou = intersection/ (union + 1e-6)
                    if iou >= threshold_iou:
                        lanes_correct_pred[lane_id_label] = lane_id_pred
                        break
            # num_lanes_correct_pred = len(lanes_correct_pred)
            # FP = (len(lanes_pred) - num_lanes_correct_pred) / (len(lanes_pred) + 1e-6)
            # FN = (len(lanes_seg_label) - num_lanes_correct_pred) / (len(lanes_seg_label) + 1e-6)
            # FPs.append(FP)
            # FNs.append(FN)

        Accs.append(Acc)

    # save pred as image gray scale
    img = img.numpy().transpose(1, 2, 0)* [0.229 * 255, 0.224 * 255, 0.225 * 255] + [0.485 * 255, 0.456 * 255, 0.406 * 255]
    img = img.astype(np.uint8)
    pred = pred.cpu().numpy()
    pred = pred.astype(np.uint8)
    im_seg = np.array(get_image_results(img, pred, H, W))
    cv2.imwrite('images/pred.png', im_seg)
    res = [{'name': 'Acc', 'value': np.mean(Accs), 'order': 'asc'}]
        #    {'name': 'FN', 'value': np.mean(FNs), 'order': 'desc'},
        #    {'name': 'FP', 'value': np.mean(FPs), 'order': 'desc'}]
    return json.dumps(res)


def eval_lane(net, cfg, epoch = None, logger = None):
    net.eval()
    res = run_test(net, cfg)
    res = call_tusimple_eval(res, cfg.num_lanes)
    res = json.loads(res)
    for r in res:
        print(r['name'], r['value'])
        if logger is not None:
            logger.add_scalar('TuEval/'+r['name'],r['value'],global_step = epoch)
    for r in res:
        if r['name'] == 'Acc':
            return r['value']