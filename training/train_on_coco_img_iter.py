'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import argparse
import os
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from src.blip import blip_decoder,clip_encoder_blip_decoder
import utils
from utils import cosine_lr_schedule,get_world_size
from data import create_dataset, create_sampler, create_loader
from data.utils import save_result, coco_caption_eval
from src.logger import LOGGER as logger
from src.logger import add_log_to_file
from mmcv import Config

def train(model, data_loader, optimizer, epoch, device,fp16=False):
    model.train()  
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50
    if(fp16):
        scaler = torch.cuda.amp.GradScaler()
    for i, (image, caption, weights) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device)
        if(fp16):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = model(image, caption)
                if(weights[0]!=0):
                    print(loss.shape)
                    print(weights.shape)
                    loss = loss * weights
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            loss = model(image, caption)      
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()    
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger.global_avg()}")     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  
@torch.no_grad()
def evaluate(model, data_loader, device, config):
    model.eval() 
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Caption generation:'
    print_freq = 10
    result = []
    for image, image_id in metric_logger.log_every(data_loader, print_freq, header): 
        image = image.to(device)       
        captions = model.generate(image, sample=False, num_beams=config['num_beams'], max_length=config['max_length'], 
                                  min_length=config['min_length'])
        for caption, img_id in zip(captions, image_id):
            result.append({"image_id": img_id.item(), "caption": caption})
    return result
def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    if(utils.is_main_process()):
        add_log_to_file(os.path.join(args.output_dir,'log.txt'))
    else:
        logger.disabled = True
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    logger.info("Creating captioning dataset")
    train_dataset,val_dataset,test_dataset = create_dataset(config['dataset_type'], config)
    logger.info(f"training set: {len(train_dataset)}")
    train_loader,val_loader,test_loader = create_loader([train_dataset,val_dataset,test_dataset],
                                                        batch_size=[config['batch_size']]*3,max_epochs=[config["max_epoch"],1,1],
                                                        num_workers=[config["num_workers"]]*3,is_trains=[True,False,False], 
                                                        collate_fns=[None,None,None])
    logger.info("Creating model")
    model = blip_decoder(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                           vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                           prompt=config['prompt'],med_config=config['med_config'],vit_ckpt_path=config['vit_ckpt_path'],
                           bert_ckpt_path=config['bert_ckpt_path'],clip_ckpt_path=config['clip_ckpt_path'])
    model = model.to(device)   
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters=True)
        model_without_ddp = model.module
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])
    best = 0
    best_epoch = 0
    best_iteration = 0
    logger.info("Start training")
    start_time = time.time()    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    print_freq = 50
    fp16 = config["fp16"]
    save_steps = config["save_steps"]
    epoch_steps = len(train_dataset)//(config["batch_size"]*get_world_size())
    if(fp16):
        scaler = torch.cuda.amp.GradScaler()
    for iteration, (image, caption, weights) in enumerate(metric_logger.log_every(train_loader, print_freq)):
        iteration += 1 
        model.train()
        image = image.to(device)
        weights = weights.to(device)
        if(fp16):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = model(image, caption,reduction='mean') 
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            optimizer.zero_grad()
            loss = model(image, caption,reduction='mean')
            loss.backward()
            optimizer.step()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        train_stats = {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}
        if(iteration % epoch_steps == 0):
            epoch = iteration // epoch_steps
            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
        if(iteration % save_steps == 0):
            metric_logger.synchronize_between_processes()
            logger.info(f"Averaged stats: {metric_logger.global_avg()}") 
            val_result = evaluate(model_without_ddp, val_loader, device, config)
            val_result_file = save_result(val_result, args.result_dir, 'val_iter%d'%iteration, remove_duplicate='image_id') 
            if utils.is_main_process():   
                coco_val = coco_caption_eval(config['eval_ann_root'],val_result_file,'val',config['eval_type'])
                coco_test = coco_val
                if args.evaluate:            
                    log_stats = {**{f'val_{k}': v for k, v in coco_val.eval.items()},
                                **{f'test_{k}': v for k, v in coco_test.eval.items()},                       
                                }
                    for k,v in log_stats.items():
                        if("CLIPScore" in k):
                            log_stats[k] = v.astype(np.float64)
                    with open(os.path.join(args.output_dir, "evaluate.txt"),"a") as f:
                        f.write(json.dumps(log_stats) + "\n")                   
                else:             
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'config': config,
                        'iteration': iteration,
                    }
                    if coco_val.eval['CIDEr'] + coco_val.eval['Bleu_4'] > best:
                        best = coco_val.eval['CIDEr'] + coco_val.eval['Bleu_4']
                        best_iteration = iteration                
                        torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))
                    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                **{f'val_{k}': v for k, v in coco_val.eval.items()},
                                **{f'test_{k}': v for k, v in coco_test.eval.items()},                       
                                'iter': iteration,
                                'best_iteration': best_iteration,
                                }
                    with open(os.path.join(args.output_dir, "metric.txt"),"a") as f:
                        f.write(json.dumps(log_stats) + "\n")     
            if args.evaluate: 
                break
            dist.barrier()  
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str)) 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/caption_coco.yaml')
    parser.add_argument('--output_dir', default='output/Caption_coco')       
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--local_rank', default=-1, type=int)
    args = parser.parse_args()
    config = Config.fromfile(args.config)
    args.result_dir = os.path.join(args.output_dir, 'result')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    config.dump(os.path.join(args.output_dir, 'config.yaml'))
    main(args, config)