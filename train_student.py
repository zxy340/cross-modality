# YOLOv3 ?? by Ultralytics, GPL-3.0 license
"""
Train a  model on a custom dataset

Usage:
    $ python path/to/train.py --data coco128.yaml --weights yolov3.pt --img 640
"""
import argparse
import math
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import cv2
import torch.optim as optim
from models.util import ConvReg, SelfA, SRRL, SimKD
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, Adam, lr_scheduler
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import val  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.datasets import create_dataloader
from utils.downloads import attempt_download
from utils.general import (LOGGER, NCOLS, check_dataset, check_file, check_git_status, check_img_size,
                           check_requirements, check_suffix, check_yaml, colorstr, get_latest_run, increment_path,
                           init_seeds, intersect_dicts, labels_to_class_weights, labels_to_image_weights, methods,
                           one_cycle, print_args, print_mutation, strip_optimizer)
from utils.loggers import Loggers
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_evolve, plot_labels
from utils.torch_utils import EarlyStopping, ModelEMA, de_parallel, select_device, torch_distributed_zero_first

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def train(hyp,  # path/to/hyp.yaml or hyp dictionary
          opt,
          device,
          callbacks
          ):
    save_dir, epochs, batch_size, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze, = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze

    # Directories
    w = save_dir / 'weights'  # weights dir
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'

    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)
    data_dict = None

    # Loggers
    if RANK in [-1, 0]:
        loggers = Loggers(save_dir, opt, hyp, LOGGER)  # loggers instance

        # Register actions
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

    # Config
    plots = not evolve  # create plots
    cuda = device.type != 'cpu'
    init_seeds(1 + RANK)
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data)  # check if None

    train_path, val_path = data_dict['train'], data_dict['val']
    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, f'{len(names)} names found for nc={nc} dataset in {data}'  # check

    # Model
    model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create

    # .....................Cross-modality teacher model loading.................................
    teacher_path = './runs/train/new_Kinect_yolov3-tiny/weights/best.pt'
    LOGGER.info(f'Load the teacher model from {teacher_path}')
    ckpt_t = torch.load(teacher_path, map_location=device)  # load checkpoint
    model_t = Model(cfg or ckpt_t['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
    exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
    csd_t = ckpt_t['model'].float().state_dict()  # checkpoint state_dict as FP32
    csd_t = intersect_dicts(csd_t, model_t.state_dict(), exclude=exclude)  # intersect
    model_t.load_state_dict(csd_t, strict=False)  # load
    # ..........................................................................................

    # Image size
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    # Batch size
    if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
        batch_size = check_train_batch_size(model, imgsz)

    # Optimizer
    # nbs = 64  # nominal batch size
    # accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    # hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    # LOGGER.info(f"Scaled weight_decay = {hyp['weight_decay']}")
    #
    # g0, g1, g2 = [], [], []  # optimizer parameter groups
    # for v in model.modules():
    #     if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
    #         g2.append(v.bias)
    #     if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
    #         g0.append(v.weight)
    #     elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
    #         g1.append(v.weight)
    #
    # if opt.adam:
    #     optimizer = Adam(g0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    # else:
    #     optimizer = SGD(g0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    #
    # optimizer.add_param_group({'params': g1, 'weight_decay': hyp['weight_decay']})  # add g1 with weight_decay
    # optimizer.add_param_group({'params': g2})  # add g2 (biases)
    # LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__} with parameter groups "
    #             f"{len(g0)} weight, {len(g1)} weight (no decay), {len(g2)} bias")
    # del g0, g1, g2

    # .................cross-modality parameters.....................
    data = torch.randn(4, 3, 416, 416).cuda()
    model_t.eval()
    model.eval()
    feat_t_raw, _ = model_t(data, is_feat=True)
    feat_t = []
    for feat_idx in range(len(feat_t_raw)):
        if feat_t_raw[feat_idx] != None:
            feat_t.append(feat_t_raw[feat_idx])
    feat_s_raw, _ = model(data, is_feat=True)
    feat_s = []
    crosslist_s = [-1]
    for feat_idx in range(len(feat_s_raw)):
        if feat_s_raw[feat_idx] != None:
            feat_s.append(feat_s_raw[feat_idx])
            crosslist_s.append(feat_idx)
    crosslist_s.append(len(feat_s_raw))
    trainable_list = nn.ModuleList([])
    trainable_list.append(model)

    if opt.beta:
        model_simkd = nn.ModuleList([])
        if not freeze:
            for feat_idx in range(len(feat_s)):
                s_n = feat_s[feat_idx].shape[1]
                t_n = feat_t[feat_idx].shape[1]
                model_simkd.append(SimKD(s_n=s_n, t_n=t_n, factor=opt.factor))
        else:
            if freeze <= len(feat_s):
                s_n = feat_s[freeze - 1].shape[1]
                t_n = feat_t[freeze - 1].shape[1]
                model_simkd.append(SimKD(s_n=s_n, t_n=t_n, factor=opt.factor))
        model_simkd = model_simkd.to(device)
        trainable_list.append(model_simkd)
        criterion_kd = nn.MSELoss()  # other knowledge distillation loss

    # Optimizer
    optimizer = Adam(trainable_list.parameters(), lr=opt.learning_rate)
    # ..............................................................

    # Scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma = 0.955)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA
    ema = ModelEMA(model) if RANK in [-1, 0] else None

    # Resume
    start_epoch, best_fitness = 0, 0.0

    # Trainloader
    train_loader, dataset = create_dataloader(train_path, imgsz, batch_size // WORLD_SIZE, gs, single_cls,
                                              hyp=hyp, augment=False, cache=opt.cache, rect=opt.rect, rank=LOCAL_RANK,
                                              workers=workers, image_weights=opt.image_weights, quad=opt.quad,
                                              prefix=colorstr('train: '), shuffle=True)
    mlc = int(np.concatenate(dataset.labels, 0)[:, 0].max())  # max label class
    nb = len(train_loader)  # number of batches
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'

    # Process 0
    if RANK in [-1, 0]:
        val_loader = create_dataloader(val_path, imgsz, batch_size // WORLD_SIZE * 2, gs, single_cls,
                                       hyp=hyp, cache=None if noval else opt.cache, rect=True, rank=-1,
                                       workers=workers, pad=0.5,
                                       prefix=colorstr('val: '))[0]

        if not resume:
            labels = np.concatenate(dataset.labels, 0)
            # c = torch.tensor(labels[:, 0])  # classes
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
            # model._initialize_biases(cf.to(device))
            if plots:
                plot_labels(labels, names, save_dir)

            # Anchors
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
            model.half().float()  # pre-reduce anchor precision

        callbacks.run('on_pretrain_routine_end')

    # Model parameters
    nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
    hyp['box'] *= 3 / nl  # scale to layers
    hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model_t.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names
   
    # .........................Cross-modality.................................
    compute_loss = ComputeLoss(model_t)  # init loss class
    if not opt.skip_validation:
        # validate teacher accuracy
        print("==> Teacher model validation...")
        results, _, _ = val.run(data_dict,
                                batch_size=batch_size // WORLD_SIZE * 2,
                                imgsz=imgsz,
                                model=attempt_load(teacher_path[2:], device).half(),
                                iou_thres=0.60,
                                single_cls=single_cls,
                                dataloader=val_loader,
                                save_dir=save_dir,
                                verbose=True,
                                plots=True,
                                callbacks=callbacks,
                                compute_loss=compute_loss,
                                val_t=True)  # val best model with plots
        print('Loss of Teacher model validation is {}'.format(results))
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")
    else:
        print('Skipping teacher validation.')
    # ...............................................................

    # Start training
    t0 = time.time()
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    stopper = EarlyStopping(patience=opt.patience)
    compute_loss = ComputeLoss(model)  # init loss class
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs...')
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        # set modules as train()
        model.train()
        if opt.beta:
            model_simkd.train()
        # set teacher as eval()
        model_t.eval()

        # Freeze
        if freeze:
            freeze_idx = [f'model.{x}.' for x in range(crosslist_s[freeze - 1] + 1, crosslist_s[freeze] + 1)]  # layers to train
            for k, v in model.named_parameters():
                v.requires_grad = False  # train all layers
                if any(x in k for x in freeze_idx):
                    LOGGER.info(f'training {k}')
                    v.requires_grad = True

        mloss = torch.zeros(3, device=device)  # mean losses
        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'labels', 'img_size'))
        if RANK in [-1, 0]:
            pbar = tqdm(pbar, total=nb, ncols=NCOLS, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
        optimizer.zero_grad()

        # .......................Cross-modality learning...............................
        for i, (imgs_s, targets_s, imgs_t, targets_t, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs_s = imgs_s.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0
            imgs_t = imgs_t.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

            # Forward
            with amp.autocast(enabled=cuda):
                feat_s_raw, logit_s = model(imgs_s, is_feat=True)
                feat_s = []
                for feat_idx in range(len(feat_s_raw)):
                    if feat_s_raw[feat_idx] != None:
                        feat_s.append(feat_s_raw[feat_idx])
                with torch.no_grad():
                    feat_t_raw, logit_t = model_t(imgs_t, is_feat=True)
                    feat_t = []
                    for feat_idx in range(len(feat_t_raw)):
                        if feat_t_raw[feat_idx] != None:
                            feat_t.append(feat_t_raw[feat_idx])
                    feat_t = [f.detach() for f in feat_t]

                # other kd loss
                loss_kd = 0
                if opt.beta:
                    if len(model_simkd):
                        if not freeze:
                            for feat_idx in range(len(model_simkd)):
                                trans_feat_s, trans_feat_t = model_simkd[feat_idx](feat_s[feat_idx], feat_t[feat_idx])
                                loss_kd = loss_kd + criterion_kd(trans_feat_s, trans_feat_t.detach())
                        else:
                            trans_feat_s, trans_feat_t = model_simkd[0](feat_s[freeze - 1], feat_t[freeze - 1])
                            loss_kd = loss_kd + criterion_kd(trans_feat_s, trans_feat_t)

                pred = model(imgs_s)  # forward
                loss_cls, loss_items = compute_loss(pred, targets_s.to(device))  # loss scaled by batch_size
                # if i % 100 == 0:
                #     LOGGER.info(f'The {i}th batch loss_cls and loss_kd are {loss_cls} and {loss_kd}, repectively')
                loss = opt.cls * loss_cls + opt.beta * loss_kd

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            scaler.step(optimizer)  # optimizer.step
            scaler.update()
            optimizer.zero_grad()
            if ema:
                ema.update(model)

            # Log
            if RANK in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%10s' * 2 + '%10.4g' * 5) % (
                    f'{epoch}/{epochs - 1}', mem, *mloss, targets_s.shape[0], imgs_s.shape[-1]))
                callbacks.run('on_train_batch_end', ni, model, imgs_s, targets_s, paths, plots, opt.sync_bn)
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        print(lr)
        scheduler.step()

        if RANK in [-1, 0]:
            # mAP
            callbacks.run('on_train_epoch_end', epoch=epoch)
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            if not noval or final_epoch:  # Calculate mAP
                results, maps, _ = val.run(data_dict,
                                           batch_size=batch_size // WORLD_SIZE * 2,
                                           imgsz=imgsz,
                                           model=ema.ema,
                                           single_cls=single_cls,
                                           dataloader=val_loader,
                                           save_dir=save_dir,
                                           plots=False,
                                           callbacks=callbacks,
                                           compute_loss=compute_loss)
            print('Loss of val is {}'.format(results))
            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi
            log_vals = list(mloss) + list(results) + lr
            callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)

            # Save model
            if (not nosave) or (final_epoch and not evolve):  # if save
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'model': deepcopy(de_parallel(model)).half(),
                        'ema': deepcopy(ema.ema).half(),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict(),
                        'date': datetime.now().isoformat()}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    print('The best model is from {} epoch!'.format(epoch))
                    torch.save(ckpt, best)
                if (epoch > 0) and (opt.save_period > 0) and (epoch % opt.save_period == 0):
                    torch.save(ckpt, w / f'epoch{epoch}.pt')
                del ckpt
                callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)

            # Stop Single-GPU
            if RANK == -1 and stopper(epoch=epoch, fitness=fi):
                break
        # end epoch ----------------------------------------------------------------------------------------------------

    # end training -----------------------------------------------------------------------------------------------------
    if RANK in [-1, 0]:
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is best:
                    LOGGER.info(f'\nValidating {f}...')
                    results, _, _ = val.run(data_dict,
                                            batch_size=batch_size // WORLD_SIZE * 2,
                                            imgsz=imgsz,
                                            model=attempt_load(f, device).half(),
                                            iou_thres=0.6,  # best pycocotools results at 0.65
                                            single_cls=single_cls,
                                            dataloader=val_loader,
                                            save_dir=save_dir,
                                            verbose=True,
                                            plots=True,
                                            callbacks=callbacks,
                                            compute_loss=compute_loss)  # val best model with plots
                    print('Loss of val is {}'.format(results[4]))

        callbacks.run('on_train_end', last, best, plots, epoch, results)
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")

    torch.cuda.empty_cache()
    return results


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', type=int, default=0, help='Number of layers to freeze. backbone=10, all=24')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')

    # Weights & Biases arguments
    parser.add_argument('--entity', default=None, help='W&B: Entity')
    parser.add_argument('--upload_dataset', action='store_true', help='W&B: Upload dataset as artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='W&B: Set bounding-box image logging interval')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='W&B: Version of dataset artifact to use')

    # Cross-modality arguments
    parser.add_argument('--print_freq', type=int, default=200, help='print frequency')
    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    # dataset and model
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')
    # distillation
    parser.add_argument('--trial', type=str, default='1', help='trial id')
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')
    parser.add_argument('--distill', type=str, default='kd', choices=['kd', 'hint', 'attention', 'similarity', 'vid',                                                              'crd', 'semckd','srrl', 'simkd'])
    parser.add_argument('-c', '--cls', type=float, default=1.0, help='weight for classification')
    parser.add_argument('-b', '--beta', type=float, default=0.0, help='weight balance for other losses')
    parser.add_argument('-f', '--factor', type=int, default=2, help='factor size of SimKD')
    parser.add_argument('-s', '--soft', type=float, default=1.0, help='attention scale of SemCKD')
    # hint layer
    parser.add_argument('--hint_layer', default=1, type=int, choices=[0, 1, 2, 3, 4])
    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')
    # multiprocessing
    parser.add_argument('--dali', type=str, choices=['cpu', 'gpu'], default=None)
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23451', type=str,
                    help='url used to set up distributed training')
    parser.add_argument('--deterministic', action='store_true', help='Make results reproducible')
    parser.add_argument('--skip-validation', action='store_true', help='Skip validation of teacher')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main(opt, callbacks=Callbacks()):
    # Checks
    if RANK in [-1, 0]:
        print_args(FILE.stem, opt)
        check_git_status()
        check_requirements(exclude=['thop'])

    # Resume
    opt.data, opt.cfg, opt.hyp, opt.project = \
        check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.project)  # checks
    assert len(opt.cfg), 'either --cfg must be specified'
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)

    # Train
    train(opt.hyp, opt, device, callbacks)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
