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

def train(net, data_loader, loss_dict, optimizer, scheduler,logger, epoch, metric_dict):
    net.train()
    progress_bar = tqdm.tqdm(train_loader)
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

def run_test(cfg, net):
    pass
    # """
    # Hàm thực hiện việc test mô hình trên dataset.

    # Args:
    #     cfg: Cấu hình chứa các tham số như dataset, data_root, batch_size, v.v.
    #     net: Mô hình cần test.
    # """
    # # Đặt mô hình vào chế độ đánh giá
    # net.eval()

    # # Tạo thư mục đánh giá nếu chưa có
    # os.makedirs(cfg.test_work_dir, exist_ok=True)

    # # Lấy DataLoader cho chế độ test bằng hàm get_test_loader
    # test_loader = get_test_loader(
    #     batch_size=cfg.batch_size,
    #     data_root=cfg.data_root,
    #     dataset=cfg.dataset,
    #     train_width=cfg.train_width,
    #     train_height=cfg.train_height
    # )

    # # Không tính gradient khi đánh giá
    # with torch.no_grad():
    #     for i, (inputs, labels) in enumerate(test_loader):
    #         inputs, labels = inputs.to(cfg.device), labels.to(cfg.device)

    #         # Dự đoán đầu ra từ mô hình
    #         outputs = net(inputs)

    #         # Tiến hành lưu kết quả dự đoán
    #         save_predictions(outputs, labels, cfg.test_work_dir, i)  # Hàm lưu kết quả bạn cần tự định nghĩa

    # print("Test complete.")

def eval_lane(net, cfg, ep=0, logger=None):
    pass
    # """
    # Đánh giá các metric cho mô hình phát hiện làn đường.

    # Args:
    #     net: Mô hình đang được đánh giá.
    #     cfg: Cấu hình của mô hình và dataset.
    #     ep: Số epoch hiện tại (nếu có).
    #     logger: Trình ghi log (nếu có).

    # Returns:
    #     Kết quả Precision, Recall, F-measure.
    # """
    # # Đánh giá mô hình trên tập dữ liệu test
    # run_test(cfg.dataset, net, cfg.data_root, 'culane_eval_tmp', cfg.test_work_dir,
    #         cfg.train_width, cfg.train_height)
    
    # # Lấy kết quả từ việc đánh giá
    # res = call_culane_eval(cfg.data_root, 'culane_eval_tmp', cfg.test_work_dir)
    
    # TP, FP, FN = 0, 0, 0
    
    # # Duyệt qua các kết quả và tính toán tổng số TP, FP, FN
    # for k, v in res.items():
    #     val = float(v['Fmeasure']) if 'nan' not in v['Fmeasure'] else 0
    #     val_tp, val_fp, val_fn = int(v['tp']), int(v['fp']), int(v['fn'])
        
    #     TP += val_tp
    #     FP += val_fp
    #     FN += val_fn
        
    #     # Ghi log các kết quả F-measure theo tên của k
    #     if logger is not None:
    #         if k == 'res_cross':
    #             logger.add_scalar('CuEval_cls/' + k, val_fp, global_step=ep)
    #             continue
    #         logger.add_scalar('CuEval_cls/' + k, val, global_step=ep)

    # # Tính Precision (P), Recall (R), F-measure (F)
    # if TP + FP == 0:
    #     P = 0
    #     print("nearly no results!")
    # else:
    #     P = TP / (TP + FP)
    
    # if TP + FN == 0:
    #     R = 0
    #     print("nearly no results!")
    # else:
    #     R = TP / (TP + FN)
    
    # if (P + R) == 0:
    #     F = 0
    # else:
    #     F = 2 * P * R / (P + R)
    
    # # Ghi lại các giá trị vào logger nếu có
    # if logger is not None:
    #     logger.add_scalar('CuEval/total', F, global_step=ep)
    #     logger.add_scalar('CuEval/P', P, global_step=ep)
    #     logger.add_scalar('CuEval/R', R, global_step=ep)
    
    # return P, R, F



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
            if 'model' in k:
                state_clip[k] = v
        net.load_state_dict(state_clip, strict=False)
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

        # res = eval_lane(net, cfg, ep = epoch, logger = logger)

        # if res is not None and res > max_res:
        #     max_res = res
        save_model(net, optimizer, work_dir)
        logger.add_scalar('CuEval/X',max_res,global_step = epoch)

    logger.close()