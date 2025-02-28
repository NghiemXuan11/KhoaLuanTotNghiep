import os, argparse
from data.dali_data import TrainCollect, TestCollect
from utils.config import Config
import torch
import time
from torch.utils.tensorboard import SummaryWriter

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help = 'path to config file')
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--data_root', default = None, type = str)
    parser.add_argument('--epoch', default = None, type = int)
    parser.add_argument('--batch_size', default = None, type = int)
    parser.add_argument('--optimizer', default = None, type = str)
    parser.add_argument('--learning_rate', default = None, type = float)
    parser.add_argument('--weight_decay', default = None, type = float)
    parser.add_argument('--momentum', default = None, type = float)
    parser.add_argument('--scheduler', default = None, type = str)
    parser.add_argument('--steps', default = None, type = int, nargs='+')
    parser.add_argument('--gamma', default = None, type = float)
    parser.add_argument('--warmup', default = None, type = str)
    parser.add_argument('--warmup_iters', default = None, type = int)
    parser.add_argument('--backbone', default = None, type = str)
    parser.add_argument('--use_aux', default = None, type = str2bool)
    parser.add_argument('--sim_loss_w', default = None, type = float)
    parser.add_argument('--shp_loss_w', default = None, type = float)
    parser.add_argument('--note', default = None, type = str)
    parser.add_argument('--log_path', default = None, type = str)
    parser.add_argument('--finetune', default = None, type = str)
    parser.add_argument('--resume', default = None, type = str)
    parser.add_argument('--test_model', default = None, type = str)
    parser.add_argument('--test_work_dir', default = None, type = str)
    parser.add_argument('--num_lanes', default = None, type = int)
    parser.add_argument('--auto_backup', action='store_false', help='automatically backup current code in the log path')
    parser.add_argument('--var_loss_power', default = None, type = float)
    parser.add_argument('--train_width', default = None, type = int)
    parser.add_argument('--train_height', default = None, type = int)
    parser.add_argument('--mean_loss_w', default = None, type = float)
    parser.add_argument('--fc_norm', default = None, type = str2bool)
    parser.add_argument('--soft_loss', default = None, type = str2bool)
    parser.add_argument('--eval_mode', default = None, type = str)
    parser.add_argument('--eval_during_training', default = None, type = str2bool)
    parser.add_argument('--split_channel', default = None, type = str2bool)
    parser.add_argument('--match_method', default = None, type = str, choices = ['fixed', 'hungarian'])
    parser.add_argument('--selected_lane', default = None, type = int, nargs='+')
    parser.add_argument('--cumsum', default = None, type = str2bool)
    parser.add_argument('--masked', default = None, type = str2bool)
    
    
    return parser

import numpy as np
def merge_config():
    args = get_args().parse_args()
    cfg = Config.fromfile(args.config)

    items = ['data_root','epoch','batch_size','optimizer','learning_rate',
    'weight_decay','momentum','scheduler','steps','gamma','warmup','warmup_iters',
    'use_aux','backbone','sim_loss_w','shp_loss_w','note','log_path',
    'finetune','resume', 'test_model','test_work_dir', 'num_lanes', 'var_loss_power', 'train_width', 'train_height',
    'mean_loss_w','fc_norm','soft_loss', 'eval_mode', 'eval_during_training', 'split_channel', 'match_method', 'selected_lane', 'cumsum', 'masked']
    for item in items:
        if getattr(args, item) is not None:
            print('merge ', item, ' config')
            setattr(cfg, item, getattr(args, item))
    
    return args, cfg

def inference(net, data_label):
    pred = net(data_label['images'])
    seg_labels = data_label['seg_labels']

    res_dict = {'seg_preds': pred, 'seg_labels': seg_labels}

    return res_dict

def save_model(net, optimizer,save_path):
    model_state_dict = net.state_dict()
    state = {'model': model_state_dict, 'optimizer': optimizer.state_dict()}
    # state = {'model': model_state_dict}
    assert os.path.exists(save_path)
    model_path = os.path.join(save_path, 'model_best.pth')
    torch.save(state, model_path)


import datetime, os
def get_work_dir(cfg):
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    hyper_param_str = '_lr_%1.0e_b_%d' % (cfg.learning_rate, cfg.batch_size)
    work_dir = os.path.join(cfg.log_path, now + hyper_param_str + cfg.note)
    return work_dir

def get_logger(work_dir, cfg):
    logger = SummaryWriter(work_dir)
    config_txt = os.path.join(work_dir, 'cfg.txt')
    
    with open(config_txt, 'w') as fp:
        fp.write(str(cfg))

    return logger

def initialize_weights(*models):
    for model in models:
        real_init_weights(model)
def real_init_weights(m):

    if isinstance(m, list):
        for mini_m in m:
            real_init_weights(mini_m)
    else:
        if isinstance(m, torch.nn.Conv2d):    
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0.0, std=0.01)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m,torch.nn.Module):
            for mini_m in m.children():
                real_init_weights(mini_m)
        else:
            print('unkonwn module', m)
            
import importlib
def get_model(cfg):
    return importlib.import_module('model.'+cfg.backbone.lower()).get_model(cfg)

def get_train_loader(cfg):
    train_loader = TrainCollect(cfg.batch_size, 4, cfg.data_root, os.path.join(cfg.data_root, 'train_gt.txt'),
                                cfg.train_width, cfg.train_height, cfg.num_lanes)
    return train_loader 

def get_test_loader(cfg):
    test_loader = TestCollect(cfg.batch_size, 4, cfg.data_root, os.path.join(cfg.data_root, 'test.txt'),
                                cfg.train_width, cfg.train_height, cfg.num_lanes)
    return test_loader

def calc_loss(loss_dict, results, logger, global_step, epoch):
    loss = 0

    for i in range(len(loss_dict['name'])):

        if loss_dict['weight'][i] == 0:
            continue
            
        data_src = loss_dict['data_src'][i]

        datas = [results[src] for src in data_src]

        loss_cur = loss_dict['op'][i](*datas)

        if global_step % 20 == 0:
            logger.add_scalar('loss/'+loss_dict['name'][i], loss_cur, global_step)

        loss += loss_cur * loss_dict['weight'][i]

    return loss