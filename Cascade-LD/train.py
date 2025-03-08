import torch
import torchvision
import numpy as np
import random
import math
import os 
import time
import datetime
import tqdm

# utils
from utils.factory import get_metric_dict, get_loss_dict, get_optimizer, get_scheduler
from utils.metrics import update_metrics, reset_metrics

from utils.common import calc_loss, get_model, get_train_loader, inference, merge_config, save_model
from utils.common import get_work_dir, get_logger

from utils.eval import eval_lane

def train(net, data_loader, loss_dict, optimizer, scheduler,logger, epoch, metric_dict):
    net.train()
    progress_bar = tqdm.tqdm(train_loader, desc="epoch "+str(epoch+1))
    for b_idx, data_label in enumerate(progress_bar):
        global_step = epoch * len(data_loader) + b_idx

        results = inference(net, data_label)

        loss = calc_loss(loss_dict, results, logger, global_step, epoch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step(global_step)


        if global_step % 20 == 0:
            reset_metrics(metric_dict)
            update_metrics(metric_dict, results)
            for me_name, me_op in zip(metric_dict['name'], metric_dict['op']):
                logger.add_scalar('metric/' + me_name, me_op.get(), global_step=global_step)
            logger.add_scalar('meta/lr', optimizer.param_groups[0]['lr'], global_step=global_step)

            if hasattr(progress_bar,'set_postfix'):
                kwargs = {me_name: '%.3f' % me_op.get() for me_name, me_op in zip(metric_dict['name'], metric_dict['op'])}
                new_kwargs = {}
                for k,v in kwargs.items():
                    if 'lane' in k:
                        continue
                    new_kwargs[k] = v
                progress_bar.set_postfix(loss = '%.3f' % float(loss), 
                                        **new_kwargs)





if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()
    work_dir = get_work_dir(cfg)
    cfg.test_work_dir = work_dir

    
    print(datetime.datetime.now().strftime('[%Y/%m/%d %H:%M:%S]') + ' start training...')
    print(cfg)
    assert cfg.backbone in ['erfnet']

    train_loader = get_train_loader(cfg)
    net = get_model(cfg)
    optimizer = get_optimizer(net, cfg)

    if cfg.finetune is not None:
        print('finetune from ', cfg.finetune)
        state_all = torch.load(cfg.finetune)['model']
        state_clip = {}  # only use backbone parameters
        for k,v in state_all.items():
            if 'module.' in k:
                state_clip[k[7:]] = v
            else:
                state_clip[k] = v
        net.load_state_dict(state_clip, strict = True)
    if cfg.resume is not None:
        print('==> Resume model from ' + cfg.resume)
        resume_dict = torch.load(cfg.resume, map_location='cpu')
        net.load_state_dict(resume_dict['model'])
        if 'optimizer' in resume_dict.keys():
            optimizer.load_state_dict(resume_dict['optimizer'])
        resume_epoch = int(os.path.split(cfg.resume)[1][2:5]) + 1
    else:
        resume_epoch = 0

    scheduler = get_scheduler(optimizer, cfg, len(train_loader))
    metric_dict = get_metric_dict(cfg)
    loss_dict = get_loss_dict(cfg)
    logger = get_logger(work_dir, cfg)
    max_res = 0
    res = None
    for epoch in range(resume_epoch, cfg.epoch):
        train(net, train_loader, loss_dict, optimizer, scheduler,logger, epoch, metric_dict)
        train_loader.reset()

        res = eval_lane(net, cfg, epoch = epoch, logger = logger)

        if res is not None and res > max_res:
            max_res = res
        save_model(net, optimizer, work_dir)
        logger.add_scalar('TuEval/X',max_res,global_step = epoch)

    logger.close()