import argparse
import json
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from multiprocessing import Process,Manager
import torch
from math import floor

parser = argparse.ArgumentParser()
parser.add_argument("--folder_name",type=str,default='syn_images')
parser.add_argument("--data_name",type=str,default='coco')

args = parser.parse_args()

folder_name = args.folder_name
data_name = args.data_name
print(folder_name)
IMAGE_PATH = f"../data/{data_name}/{folder_name}"
OUTPUT_FILE = f"../data/{data_name}/annotations/nsfw_of_{folder_name}.txt"

class D_for_multiprocessing(torch.utils.data.Dataset):
    def __init__(self,list):
        self.list = list
    def __len__(self):
        return len(self.list)
    def __getitem__(self,idx):
        return self.list[idx]
def check_nsfw_images(dataset,shared_list):
    for i in tqdm(range(len(dataset))):
        img_name = dataset[i]
        img_path = os.path.join(IMAGE_PATH,img_name)
        img = Image.open(img_path)
        if(np.sum(img) == 0):
            shared_list.append(img_name)


def find_nsfw_image_list():
    img_names = os.listdir(IMAGE_PATH)
    dataset = D_for_multiprocessing(img_names)
    process_nums = 8
    lengths  = [1/process_nums]*process_nums
    lengths = [floor(len(dataset) * i) for i in lengths]
    lengths[-1] = len(dataset) - sum(lengths[:-1])
    datasets = torch.utils.data.random_split(dataset,lengths)

    with Manager() as manager:
        black_img_list = manager.list()
        process_list = []
        for i in range(process_nums):
            p = Process(target=check_nsfw_images,args=(datasets[i],black_img_list,))
            p.start()
            process_list.append(p)
        for p in process_list:
            p.join()
        results = list(black_img_list)
    print("num of nsfw images is",len(results))
    with open(OUTPUT_FILE,"w") as f:
            for i in results:
                f.write(i+'\n')
if __name__ == '__main__':
    find_nsfw_image_list()
