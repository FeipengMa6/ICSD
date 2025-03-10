import json
import os
from collections import defaultdict
import argparse
from tqdm import tqdm
from multiprocessing import Process
from torch.utils.data import random_split,Dataset
import math
import numpy as np
from collections import defaultdict
import numpy as np
import pandas as pd
import sys


def grouping_caption2coco_format(ann_path,mapping_json,split='train',data_type="coco",attr="w_coco_id",step=30,group_nums=5):
    source_json = json.load(open(os.path.join(ann_path,f'{data_type}_karpathy_{split}_{attr}.json'),"r"))
    mapping_json = json.load(open(mapping_json,"r")) 
    ann_tmp = []
    for ann in source_json:
        if('coco_id' in ann.keys()):
            ann_tmp.append(ann)
    source_json = ann_tmp

    mapping_annotations = []
    for idx in tqdm(range(0,len(mapping_json)//step)):
        ann_list = mapping_json[idx*step:(idx+1)*step]
        head_id = f"{ann_list[0]['image_id']}_{int(ann_list[0]['coco_id'])}"
        for ann in ann_list[:group_nums]: 
            mapping_annotations.append({"image_id":ann["image_id"],"coco_id":ann["coco_id"],
                                        "caption":ann['caption'],"group_id":idx,
                                        "head_id":head_id})
    target_json = mapping_annotations
    print(len(target_json))

    return target_json

def deduplication(coco_format_json):
    mapping_json = coco_format_json
    group2imgcoco = defaultdict(list)
    group2head_id = defaultdict()
    imgcoco2caption = defaultdict()
    for ann in mapping_json:
        group2imgcoco[ann['group_id']].append(f"{ann['image_id']}_{ann['coco_id']}")
        imgcoco2caption[f"{ann['image_id']}_{ann['coco_id']}"] = ann['caption']
        group2head_id[ann["group_id"]] = ann["head_id"]
    print(len(group2imgcoco.keys()))
    from tqdm import tqdm
    def tupleize(lst):
        return tuple(sorted(lst))
    dictionary = group2imgcoco
    unique_dictionary = {}
    seen_values = set()

    for key, value in tqdm(dictionary.items()):
        tuple_value = tupleize(value)
        if tuple_value not in seen_values:
            unique_dictionary[key] = value
            seen_values.add(tuple_value)
    print(len(unique_dictionary.keys()))

    group2imgcoco = unique_dictionary 
    ann_list_w_group_id = []
    for group_id in group2imgcoco.keys():
        imgcoco_ids = group2imgcoco[group_id]
        head_id = group2head_id[group_id]
        for imgcoco_id in imgcoco_ids:
            caption = imgcoco2caption[imgcoco_id]
            image_id,coco_id = imgcoco_id.split("_")
            ann_list_w_group_id.append({"image_id":image_id,"coco_id":coco_id,"caption":caption,"group_id":group_id,"head_id":head_id})
    counter = defaultdict(set)
    for ann in ann_list_w_group_id:
        counter[ann['image_id']+'_'+ann['coco_id']].add(ann['group_id'])
    new_counter = {}
    for k in counter.keys():
        if(isinstance(counter[k],int)):
            new_counter[k] = 1
        else:
            new_counter[k] = len(counter[k])
    imgcoco2grouptimes = new_counter

    ann_list_w_group_id_weighted = []
    for ann in ann_list_w_group_id:
        group_times = imgcoco2grouptimes[ann['image_id']+'_'+ann['coco_id']]
        ann_list_w_group_id_weighted.append({"image_id":ann["image_id"],"coco_id":ann["coco_id"],
                                    "caption":ann['caption'],"group_id":ann["group_id"],"head_id":ann["head_id"],
                                    "group_times":group_times})
    return ann_list_w_group_id_weighted

if __name__ == '__main__':
    data_type = "coco"
    path_dict = {"coco" : "../data/coco/annotations/",
                 "flickr30k": "../data/flickr30k/annotations/",
                 "ss1m": "../data/ss1m/annotations/",
                 }
    ann_path = path_dict[data_type]

    split = "train"
    attr="w_coco_id"
    topk=30
    mapping_json = os.path.join(ann_path,f"group_captions_clip_top{topk}.json")
    group_nums = 5 
    coco_format_json = grouping_caption2coco_format(ann_path,mapping_json,split=split,data_type=data_type,attr=attr,step=topk,group_nums=group_nums)
    ann_list_w_group_id_weighted = deduplication(coco_format_json)
    with open(os.path.join(ann_path,f"{data_type}_karpathy_train_w_group_id.json"),"w") as f:
        json.dump(ann_list_w_group_id_weighted,f)
