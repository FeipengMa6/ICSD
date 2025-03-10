import os
import json
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from PIL import Image
class nocaps_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, split):   
        filenames = {'out_val':'out_domain_nocaps.json','out_test':'out_domain_nocaps.json',
                     'in_val':'in_domain_nocaps.json','in_test':'in_domain_nocaps.json',
                     'near_val':'near_domain_nocaps.json','near_test':'near_domain_nocaps.json',
                     'overall_val': "all_domain_nocaps.json",'overall_test': 'all_domain_nocaps.json',
                     }
        self.annotation = json.load(open(os.path.join(ann_root,filenames[split]),'r'))
        self.transform = transform
        self.image_root = image_root
    def __len__(self):
        return len(self.annotation)
    def __getitem__(self, index):  
        ann = self.annotation[index]
        image_name = str(ann['image_id'])+".jpg"
        image_path = os.path.join(self.image_root,image_name)     
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)        
        img_id = ann["image_id"] 
        return image, int(img_id)