import os
import json
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from PIL import Image
from data.utils import pre_caption
class flickr30k_train(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=30, prompt=''):        
        '''
        image_root (string): Root directory of images (e.g. flickr30k/)
        ann_root (string): directory to store the annotation file
        '''        
        url = 'https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_train.json'
        filename = 'flickr30k_train.json'
        download_url(url,ann_root)
        self.annotation = json.load(open(os.path.join(ann_root,filename),'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.prompt = prompt
        self.img_ids = {}  
        n = 0
        for ann in self.annotation:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
    def __len__(self):
        return len(self.annotation)
    def __getitem__(self, index):    
        ann = self.annotation[index]
        image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        caption = self.prompt+pre_caption(ann['caption'], self.max_words) 
        return image, caption, self.img_ids[ann['image_id']] 
class flickr_karpathy_caption_eval(Dataset):
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
            filenames = {'val':'flickr30k_karpathy_val.json','test':'flickr30k_karpathy_test.json'}
        self.annotation = json.load(open(os.path.join(ann_root,filenames[split]),'r'))
        self.transform = transform
        self.image_root = image_root
    def __len__(self):
        return len(self.annotation)
    def __getitem__(self, index):    
        return self.__getitem__default(index)
    def __getitem__default(self,index):
        ann = self.annotation[index]
        image_name = ann['image']+".jpg"
        image_path = os.path.join(self.image_root,image_name)        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)        
        img_id = ann["image"] 
        return image, int(img_id)
class flickr30k_retrieval_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, split, max_words=30):  
        '''
        image_root (string): Root directory of images (e.g. flickr30k/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_val.json',
                'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_test.json'}
        filenames = {'val':'flickr30k_val.json','test':'flickr30k_test.json'}
        download_url(urls[split],ann_root)
        self.annotation = json.load(open(os.path.join(ann_root,filenames[split]),'r'))
        self.transform = transform
        self.image_root = image_root
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption,max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
    def __len__(self):
        return len(self.annotation)
    def __getitem__(self, index):    
        image_path = os.path.join(self.image_root, self.annotation[index]['image'])        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  
        return image, index    