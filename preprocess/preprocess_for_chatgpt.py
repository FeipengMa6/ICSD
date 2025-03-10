import json
import os
from collections import defaultdict
import numpy as np
from tqdm import tqdm

data_name = "coco" # "flickr30k", "ss1m"
k = 30  # 20 10

ann_path = f"../data/{data_name}/annotations"
group_captions_path = os.path.join(ann_path,f"group_captions_clip_top{k}.json")
with open(group_captions_path,"r") as f:
    group_captions = json.load(f)
num_samples = len(group_captions) // k
meta_dict = defaultdict(list)
for i in tqdm(range(num_samples)):
    ann_meta = group_captions[i*k:(i+1)*k]
    head_id = f"{ann_meta[0]['image_id']}_{ann_meta[0]['coco_id']}"
    meta_dict[head_id].extend(ann_meta)
key_list = list(meta_dict.keys())
np.random.shuffle(key_list)
shuffle_meta_dict = {k:meta_dict[k] for k in key_list}
with open(f"../data/{data_name}/annotations/{data_name}_chatgpt_train_group_summary.json","w") as f:
    json.dump(shuffle_meta_dict,f)

