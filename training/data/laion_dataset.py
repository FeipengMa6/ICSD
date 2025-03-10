import os
import json
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
import torch
from PIL import Image
from tqdm import tqdm
from data.utils import pre_caption
from pathlib import Path
class laion_train(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=30, prompt='',nsfw_images=None):        
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''        
        self.annotation = json.load(open(ann_root,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.prompt = prompt
        tmp_ann = []
        print("Filter items not exists")
        for i in tqdm(range(len(self.annotation))):
            ann = self.annotation[i]
            img_id = str(ann['coco_id']) + '_' + str(ann['laion_id']).split('.')[0]
            image_name = img_id + '.jpg'
            image_path = os.path.join(self.image_root,image_name)
            ann['path'] = image_path
            if(nsfw_images is not None and image_name in nsfw_images):
                continue
            else:
                if(os.path.isfile(image_path)): 
                    tmp_ann.append(ann)
        self.annotation = tmp_ann
    def __len__(self):
        return len(self.annotation)
    def __getitem__(self, index):    
        ann = self.annotation[index]
        image_path = ann['path']
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        caption = self.prompt+pre_caption(ann['caption'], self.max_words) 
        return image, caption, 0 
class laion_train_large(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=30, prompt=''):        
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''        
        filename='laion_karpathy_train_large2.json'
        self.annotation = json.load(open(os.path.join(ann_root,filename),'r')) 
        self.transform = transform
        self.image_root = image_root 
        self.max_words = max_words      
        self.prompt = prompt
        sub_set = ['00000','00060']
        self.paths = []
        for i,img_path in enumerate(image_root):
            for ann in self.annotation[sub_set[i]]:
                img_id = str(ann['coco_id']) + '_' + str(ann['laion_id']).split('.')[0]
                self.paths.append(os.path.join(img_path,img_id+'.jpg'))
        tmp_ = []
        tmp_.extend(self.annotation[sub_set[0]]) 
        tmp_.extend(self.annotation[sub_set[1]]) 
        self.annotation = tmp_
        self.paths = self.paths
        tmp_ann = []
        tmp_path = []
        print("Filter items not exists")
        for i in tqdm(range(len(self.annotation))):
            ann = self.annotation[i]
            if(os.path.isfile(self.paths[i])): 
                tmp_ann.append(ann)
                tmp_path.append(self.paths[i])
        self.annotation = tmp_ann
        self.paths = tmp_path
        self.img_ids = {}  
        n = 0
        for i in tqdm(range(len(self.annotation))):
            ann = self.annotation[i]
            img_id = str(ann['coco_id']) + '_' + str(ann['laion_id']).split('.')[0]
            ann['image_id'] = img_id
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
    def __len__(self):
        return len(self.annotation) 
    def __getitem__(self, index):    
        ann = self.annotation[index]
        image_path = self.paths[index]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        caption = self.prompt+pre_caption(ann['caption'], self.max_words) 
        return image, caption, self.img_ids[ann['image_id']] 
class laion_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, split):  
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        filenames = {'val':'laion_karpathy_val.json','test':'laion_karpathy_test.json'}
        self.annotation = json.load(open(os.path.join(ann_root,filenames[split]),'r'))
        self.transform = transform
        self.image_root = image_root
        tmp_ann = []
        print("Filter items not exists")
        for i in tqdm(range(len(self.annotation))):
            ann = self.annotation[i]
            img_id = str(ann['coco_id']) + '_' + str(ann['laion_id']).split('.')[0]
            image_path = os.path.join(self.image_root,img_id+'.jpg')
            if(os.path.isfile(image_path)): 
                tmp_ann.append(ann)
        self.annotation = tmp_ann
    def __len__(self):
        return len(self.annotation)
    def __getitem__(self, index):    
        ann = self.annotation[index]
        image_path = os.path.join(self.image_root,str(ann['coco_id']) + '_' + str(ann['laion_id']).split('.')[0]+'.jpg')
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, int(ann['laion_id'])
