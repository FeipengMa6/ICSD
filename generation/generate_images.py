import os
import random
import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL
from diffusers import DPMSolverMultistepScheduler
from tqdm import tqdm
import json
from collections import defaultdict
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.multiprocessing as mp
from torch.utils.data import random_split
from math import floor
class D_(torch.utils.data.Dataset):
    def __init__(self,meta_path,filter_=None):
        with open(meta_path,"r") as f:
            self.meta_data = json.load(f)
        c = 0
        if(filter_ is not None):
            print("total len before filter: ",len(self.meta_data))
            temp_ = []
            for itm in self.meta_data:
                if("coco_id" in itm.keys()):
                    if(str(int(itm["coco_id"])) in filter_):
                        continue
                    else:
                        temp_.append(itm)
                else:
                    c+=1
            self.meta_data = temp_
            print("total len after filter: ",len(self.meta_data))
            print(c)

    def __getitem__(self,idx):
        meta_info = self.meta_data[idx]
        coco_id = str(int(meta_info["coco_id"])) # meta_info["coco_id"]
        image_id = str(int(meta_info["image_id"])) # meta_info["image_id"]
        caption = meta_info["caption"]

        return coco_id,image_id,caption

    def __len__(self):
        return len(self.meta_data)
    
class D_summary(torch.utils.data.Dataset):
    def __init__(self,meta_path,filter_=None):
        with open(meta_path,"r") as f:
            self.meta_data = json.load(f)
        tmp_ = []
        for itm in self.meta_data:
            if(itm.keys() == 'prompt'):
                continue
            else:
                tmp_.append(itm)
        self.meta_data = tmp_

    def __getitem__(self,idx):
        meta_info = self.meta_data[idx]
        # coco_id = meta_info["coco_id"]
        image_id = str(int(meta_info["image_id"]))
        caption = meta_info["caption"]
        return image_id,image_id,caption

    def __len__(self):
        return len(self.meta_data)

def set_seed(seed=1234):
    random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
def main_worker(args,dataset,device,img_root_path,rank=0):
    p = 3 
    set_seed(args.seed+rank+p) 
    diffusion_model_path = args.diffusion_model_path
    step = 20
    height,width = 512,512
    scheduler = DPMSolverMultistepScheduler.from_config(diffusion_model_path,subfolder="scheduler")
    model = StableDiffusionPipeline.from_pretrained(diffusion_model_path,scheduler=scheduler,torch_dtype=torch.float16)
    model.to(device)
    dataloader = torch.utils.data.DataLoader(dataset=dataset,batch_size=args.batch_size,shuffle=False,
                                             pin_memory=True,num_workers=args.num_workers)
    with torch.no_grad():
        for batch in tqdm(dataloader):
            coco_ids,image_ids,captions = batch
            outputs = model(captions,num_inference_steps=step,num_images_per_prompt=1,width=width,height=height)
            imgs = outputs.images
            for j in range(len(imgs)):
                img_save_path = os.path.join(img_root_path,image_ids[j]+"_"+coco_ids[j]+".jpg")
                imgs[j].save(img_save_path)

def main(args):
    set_seed(args.seed)
    ctx = mp.get_context('spawn')
    meta_path = f"../data/{args.data_type}/annotations/{args.data_type}_karpathy_{args.coco_split}_{args.attr}.json"
    img_root_path = args.image_path
    
    os.makedirs(img_root_path,exist_ok=True)
    exist_coco_id = os.listdir(img_root_path)
    exist_coco_id = [i[:-4] for i in exist_coco_id]
    if('summary' in args.attr):
        dataset = D_summary(meta_path=meta_path,filter_=exist_coco_id)
    else:
        dataset = D_(meta_path=meta_path,filter_=exist_coco_id)
    total_len = len(dataset)
    print(total_len)

    process_nums = args.nums_gpu
    lengths  = [1/process_nums]*process_nums
    lengths = [floor(len(dataset) * i) for i in lengths]
    lengths[-1] = len(dataset) - sum(lengths[:-1])
    sub_datasets = torch.utils.data.random_split(dataset,lengths)

    processes = []
    for i in range(process_nums):
        device = torch.device("cuda",i)
        p = ctx.Process(target=main_worker,args=(args,sub_datasets[i],device,img_root_path,i,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_split",type=str,default='train')
    parser.add_argument("--image_path",type=str,default='../data/coco/syn_images')
    parser.add_argument("--diffusion_model_path",type=str,default='/path_to_stable-diffusion-v1-4')
    parser.add_argument("--seed",type=int,default=4567)
    parser.add_argument("--batch_size",type=int,default=16)
    parser.add_argument("--num_workers",type=int,default=16)
    parser.add_argument("--local_rank",type=int,default=-1)
    parser.add_argument("--data_type",type=str,default='coco')
    parser.add_argument("--attr",type=str,default='w_coco_id')
    parser.add_argument("--nums_gpu",type=int,default=8)
    
    args = parser.parse_args()
    os.makedirs(args.image_path,exist_ok=True)

    main(args)