import os
import json
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import clip
from data.utils import pre_caption
from tqdm import tqdm
import numpy as np
class coco_karpathy_train(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=30, prompt='',**kwargs):        
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''        
        self.annotation = json.load(open(ann_root,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.prompt = prompt
        for ann in self.annotation:
            if('image' in ann.keys()):
                image_name = ann['image'].split('_')[-1]
            else:
                image_name = f"{int(ann['image_id']):012d}.jpg"
            image_path = os.path.join(self.image_root,image_name)
            ann['path'] = image_path
    def __len__(self):
        return len(self.annotation)
    def __getitem__(self, index):    
        ann = self.annotation[index]
        image_path = ann['path']
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        caption = self.prompt+pre_caption(ann['caption'], self.max_words) 
        return image, caption, 0 
class coco_karpathy_train_w_joint(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=30, prompt='',syn_image_root=None,**kwargs):        
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''        
        filename = 'coco_karpathy_train.json'
        syn_filename = 'coco_karpathy_train_w_coco_id.json'
        self.annotation = json.load(open(os.path.join(ann_root,filename),'r'))
        self.syn_annotation = json.load(open(os.path.join(ann_root,syn_filename),'r'))
        self.transform = transform
        self.image_root = image_root
        self.syn_image_root = syn_image_root
        self.max_words = max_words      
        self.prompt = prompt
        ann_tmp = []
        for ann in self.annotation:
            image_name = ann['image'].split('_')[-1]
            image_path = os.path.join(self.image_root,image_name)        
            ann['path'] = image_path
            ann_tmp.append(ann)
        for ann in self.syn_annotation:
            if('coco_id' not in ann.keys()):
                continue
            image_name = f"{ann['image_id']}_{ann['coco_id']}.jpg"
            image_path = os.path.join(self.syn_image_root,image_name)        
            ann['path'] = image_path
            ann_tmp.append(ann)
        self.annotation = ann_tmp
    def __len__(self):
        return len(self.annotation)
    def __getitem__(self, index):    
        ann = self.annotation[index]
        image_path = ann['path']
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        caption = self.prompt+pre_caption(ann['caption'], self.max_words) 
        return image, caption, self.img_ids[ann['image_id']] 

class coco_karpathy_train_syn_w_group_id(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=30, prompt='',nsfw_images=None,**kwargs):        
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''        
        self.annotation = json.load(open(ann_root,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.prompt = prompt
        self.clip_transform = kwargs['clip_transform'] if 'clip_transform' in kwargs.keys() else None
        if(nsfw_images is not None):
            tmp_ = []
            with open(nsfw_images,"r") as f:
                tmp_ = [i.strip() for i in f]
            nsfw_images = tmp_
        tmp_ann = []
        for ann in tqdm(self.annotation):
            if('coco_id' in ann.keys()):
                image_name = ann["head_id"]+'.jpg'
                image_path = os.path.join(self.image_root,image_name)
                if(nsfw_images is not None and image_name in nsfw_images):
                    continue
                else:
                    ann['path'] = image_path
                    tmp_ann.append(ann)
        self.annotation = tmp_ann
    def __len__(self):
        return len(self.annotation)
    def __getitem__(self, index):    
        return self.__getitem__default(index)
    def __getitem__default(self,index):
        ann = self.annotation[index]
        image_path = ann['path']
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        caption = self.prompt+pre_caption(ann['caption'], self.max_words) 
        weights = 0
        return image, caption, weights
class coco_karpathy_train_syn(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=30, prompt='',nsfw_images=None,**kwargs):        
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''        
        self.annotation = json.load(open(ann_root,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.prompt = prompt
        self.clip_transform = kwargs['clip_transform'] if 'clip_transform' in kwargs.keys() else None
        if(nsfw_images is not None):
            tmp_ = []
            with open(nsfw_images,"r") as f:
                tmp_ = [i.strip() for i in f]
            nsfw_images = tmp_
        tmp_ann = []
        for ann in tqdm(self.annotation):
            if('coco_id' in ann.keys()):
                if('ss1m' in self.image_root):
                    image_name = str(ann['image_id']) + '_' + str(ann['coco_id'])+'.jpg'
                else:
                    image_name = str(ann['image_id']) +'.jpg'
                image_path = os.path.join(self.image_root,image_name)
                if(nsfw_images is not None and image_name in nsfw_images):
                    continue
                else:
                    ann['path'] = image_path
                    tmp_ann.append(ann)
        self.annotation = tmp_ann
    def __len__(self):
        return len(self.annotation)
    def __getitem__(self, index):    
        if(self.clip_transform is not None):
            return self.__getitem__clip(index)
        else:
            return self.__getitem__default(index)
    def __getitem__default(self,index):
        ann = self.annotation[index]
        image_path = ann['path']
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        caption = self.prompt+pre_caption(ann['caption'], self.max_words) 
        return image, caption, 0
    def __getitem__clip(self,index):
        ann = self.annotation[index]
        image_path = ann['path']        
        image_ = Image.open(image_path).convert('RGB')
        image = self.transform(image_)
        clip_image = self.clip_transform(image_)
        caption = self.prompt+pre_caption(ann['caption'], self.max_words) 
        return image, caption, clip_image  
class coco_karpathy_train_syn_summary(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=30, prompt='',nsfw_images=None,**kwargs):        
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''        
        self.annotation = json.load(open(ann_root,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.prompt = prompt
        self.clip_transform = kwargs['clip_transform'] if 'clip_transform' in kwargs.keys() else None
        if(nsfw_images is not None):
            tmp_ = []
            with open(nsfw_images,"r") as f:
                tmp_ = [i.strip() for i in f]
            nsfw_images = tmp_
        tmp_ann = []
        for ann in tqdm(self.annotation):
            if("group_id" in ann.keys()):
                image_name = str(ann['group_id']) + '.jpg'
            else:
                image_name = str(ann['image_id']) + '.jpg'
            image_path = os.path.join(self.image_root,image_name)
            if(nsfw_images is not None and image_name in nsfw_images):
                continue
            elif(not os.path.isfile(image_path)):
                continue
            else:
                ann['path'] = image_path
                tmp_ann.append(ann)
        self.annotation = tmp_ann
    def __len__(self):
        return len(self.annotation)
    def __getitem__(self, index):    
        if(self.clip_transform is not None):
            return self.__getitem__clip(index)
        else:
            return self.__getitem__default(index)
    def __getitem__default(self,index):
        ann = self.annotation[index]
        image_path = ann['path']
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        caption = self.prompt+pre_caption(ann['caption'], self.max_words) 
        return image, caption, 0
    def __getitem__clip(self,index):
        ann = self.annotation[index]
        image_path = ann['path']        
        image_ = Image.open(image_path).convert('RGB')
        image = self.transform(image_)
        clip_image = self.clip_transform(image_)
        caption = self.prompt+pre_caption(ann['caption'], self.max_words) 
        return image, caption, clip_image  

class coco_karpathy_caption_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, split,**kwargs):  
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        self.return_nouns = kwargs['return_nouns'] if 'return_nouns' in kwargs.keys() else None
        self.clip_transform = kwargs['clip_transform'] if 'clip_transform' in kwargs.keys() else None
        if(self.return_nouns):
            filenames = {'val':'coco_karpathy_val_w_nouns.json','test':'coco_karpathy_test_w_nouns.json'}
        else:
            filenames = {'val':'coco_karpathy_val.json','test':'coco_karpathy_test.json'}
        self.annotation = json.load(open(os.path.join(ann_root,filenames[split]),'r'))
        self.transform = transform
        self.image_root = image_root
    def __len__(self):
        return len(self.annotation)
    def __getitem__(self, index):    
        if(self.return_nouns):
            return self.__getitem__nouns(index)
        elif(self.clip_transform):
            return self.__getitem__clip(index)
        else:
            return self.__getitem__default(index)
    def __getitem__default(self,index):
        ann = self.annotation[index]
        image_name = ann['image'].split('_')[-1]
        image_path = os.path.join(self.image_root,image_name)        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)        
        img_id = ann['image'].split('/')[-1].strip('.jpg').split('_')[-1]
        return image, int(img_id)
    def __getitem__clip(self,index):
        ann = self.annotation[index]
        image_name = ann['image'].split('_')[-1]
        image_path = os.path.join(self.image_root,image_name)    
        image_ = Image.open(image_path).convert('RGB')
        image = self.transform(image_)
        clip_image = self.clip_transform(image_)
        img_id = ann['image'].split('/')[-1].strip('.jpg').split('_')[-1]
        return image, int(img_id), clip_image  
    def __getitem__nouns(self,index):
        ann = self.annotation[index]
        image_name = ann['image'].split('_')[-1]
        image_path = os.path.join(self.image_root,image_name)        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)        
        img_id = ann['image'].split('/')[-1].strip('.jpg').split('_')[-1]
        self.clip_prompt = "A picture of "
        self.max_length = 30
        nouns = ann['nouns']
        prompt_sentences = [self.clip_prompt + n for n in nouns]
        prompt_ids = clip.tokenize(prompt_sentences)
        prompt_ids = F.pad(prompt_ids,(0,0,0,self.max_length-len(prompt_ids)))
        return image, int(img_id),prompt_ids
class coco_karpathy_caption_eval_multi(Dataset):
    def __init__(self, transform, clip_transform,image_root, ann_root, split):  
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        filenames = {'val':'coco_karpathy_val.json','test':'coco_karpathy_test.json'}
        self.annotation = json.load(open(os.path.join(ann_root,filenames[split]),'r'))
        self.transform = transform
        self.clip_transform = clip_transform
        self.image_root = image_root
    def __len__(self):
        return len(self.annotation)
    def __getitem__(self, index):    
        ann = self.annotation[index]
        image_name = ann['image'].split('_')[-1]
        image_path = os.path.join(self.image_root,image_name)        
        image_ = Image.open(image_path).convert('RGB')
        image = self.transform(image_)
        clip_image = self.clip_transform(image_)
        img_id = ann['image'].split('/')[-1].strip('.jpg').split('_')[-1]
        return image, int(img_id),clip_image
