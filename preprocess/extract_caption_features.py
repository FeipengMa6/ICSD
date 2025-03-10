import argparse
import json
import pandas as pd
from pandas import read_parquet
from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer
import random
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import lmdb
import os
import numpy as np
import clip

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank",type=int,default=1)
parser.add_argument("--seed",type=int,default=1234)
parser.add_argument("--meta_path",type=str,default="../data/coco/annotations/captions_train2017.json")
parser.add_argument("--save_path",type=str,default="../data/coco/embeddings")
parser.add_argument("--clip_path",type=str,default="../pretrained_models/ViT-B-16.pt")
parser.add_argument("--diffcse_path",type=str,default="../pretrained_models/diffcse-roberta-base-trans")
parser.add_argument("--bs",type=int,default=128)
parser.add_argument("--clip",action="store_true")
parser.add_argument("--split",type=str,default="train")
args = parser.parse_args()

model_path = args.diffcse_path
tokenizer = AutoTokenizer.from_pretrained(model_path)

def all_gather(local_rank, world_size, **tensors):
    tensors = list(tensors.values())
    _dims = [t.shape[-1] for t in tensors]
    tensors = torch.cat(tensors, dim=-1)
    tensors_all = [torch.zeros_like(tensors) for _ in range(world_size)]
    dist.all_gather(tensors_all, tensors)
    tensors_all[local_rank] = tensors
    tensors_all = torch.cat(tensors_all, dim=0)

    results = list()
    dimStart = 0
    assert sum(_dims) == tensors_all.shape[-1]
    for d in _dims:
        results.append(tensors_all[..., dimStart: dimStart + d])
        dimStart += d

    return tuple(results)

def set_seed(seed=1234):
    random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

class D_CoCo(torch.utils.data.Dataset):
    def __init__(self,meta_path):
        super().__init__()
        with open(meta_path,"r") as f:
            self.meta_data = json.load(f)
        self.meta_annotations = self.meta_data['annotations']
    
    def __getitem__(self,idx):
        meta_info = self.meta_annotations[idx]
        image_id = meta_info['image_id']
        id = meta_info['id']
        caption = meta_info['caption']
        return image_id,id,caption

    def __len__(self):
        return len(self.meta_annotations)
    
    @staticmethod
    def collate_fn(batch):
        image_id,id,caption = zip(*batch)
        tokenizer_outputs = tokenizer(list(caption), padding=True, return_tensors="pt")
        text_ids = tokenizer_outputs['input_ids']
        attn_mask = tokenizer_outputs['attention_mask']
        image_id = torch.tensor(image_id)
        id = torch.tensor(id)
        return image_id,id,text_ids,attn_mask,list(caption)

def main(args):
    set_seed(args.seed)
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda",args.local_rank)
    
    dataset = D_CoCo(args.meta_path)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=args.bs,num_workers=32,
                                    pin_memory=True,prefetch_factor=4,sampler=sampler,collate_fn=dataset.collate_fn)
    if(args.clip):
        model,preprocess = clip.load(args.clip_path)
        model.to(device)
        model.eval()
    else:
        model = AutoModel.from_pretrained(model_path)
        model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.local_rank],output_device=args.local_rank)
        model.eval()
    sample_ids = []
    indexs = []
    text_embs  = []
    with torch.no_grad():
        for i,batch in enumerate(tqdm(dataloader)):
            sample_id,index,text_ids,attn_mask,captions = batch
            text_ids = text_ids.to(device)
            attn_mask = attn_mask.to(device)
            sample_id = sample_id.to(device)
            index = index.to(device)
            if(args.clip):
                text_ids = clip.tokenize(captions)
                text_emb = model.encode_text(text_ids.to(device))
            else:
                text_emb = model(input_ids=text_ids,attention_mask=attn_mask,output_hidden_states=False, return_dict=True).pooler_output
            if(not args.clip):
                sample_id,index,text_emb = all_gather(args.local_rank, int(os.environ["WORLD_SIZE"]), sample_id=sample_id[:,None], index=index[:,None],text_emb=text_emb)
                sample_id = sample_id.squeeze(1)
                index = index.squeeze(1)
                dist.barrier()
                if(args.local_rank==0):
                    sample_ids.append(sample_id.detach().cpu().numpy())
                    indexs.append(index.detach().cpu().numpy())
                    text_embs.append(text_emb.detach().cpu().numpy())
                dist.barrier()
            else:
                sample_ids.append(sample_id.detach().cpu().numpy())
                indexs.append(index.detach().cpu().numpy())
                text_embs.append(text_emb.detach().cpu().numpy())
    
    if(not args.clip):
        if(args.local_rank == 0):
            sample_ids = np.concatenate(sample_ids,axis=0)
            indexs = np.concatenate(indexs,axis=0)
            text_embs = np.concatenate(text_embs,axis=0)
            print(sample_ids.shape)
            np.save(os.path.join(args.save_path,f"sample_id_{args.split}.npy"),sample_ids)
            np.save(os.path.join(args.save_path,f"id_{args.split}.npy"),indexs)
            np.save(os.path.join(args.save_path,f"caption_{args.split}.npy"),text_embs)
    else:
        sample_ids = np.concatenate(sample_ids,axis=0)
        indexs = np.concatenate(indexs,axis=0)
        text_embs = np.concatenate(text_embs,axis=0)
        print(sample_ids.shape,indexs.shape,text_embs.shape)
        np.save(os.path.join(args.save_path,f"clip_sample_id_{args.split}.npy"),sample_ids)
        np.save(os.path.join(args.save_path,f"clip_id_{args.split}.npy"),indexs)
        np.save(os.path.join(args.save_path,f"clip_caption_{args.split}.npy"),text_embs)
if __name__ == '__main__':
    main(args)