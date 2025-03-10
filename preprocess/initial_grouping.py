import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--use_clip",action='store_true')
parser.add_argument("--coco_ann_path",type=str, default="../data/coco")

args = parser.parse_args()

use_clip = args.use_clip
coco_ann_path = args.coco_ann_path #"../data/coco"
coco_train_json = coco_ann_path + "/annotations/coco_karpathy_train_w_coco_id.json"
with open(coco_train_json,"r") as f:
    coco_train = json.load(f)
coco_train_dict = {}
for ann in coco_train:
    if('coco_id' in ann.keys()):
        coco_train_dict[int(int(ann['image_id'])*1e6+int(ann['coco_id']))] = ann
if(use_clip):
    emb_train = np.load(coco_ann_path + "/embeddings/clip_caption_train.npy")
    emb_test = np.load(coco_ann_path + "/embeddings/clip_caption_val.npy")
else:
    emb_train = np.load(coco_ann_path + "/embeddings/caption_train.npy")
    emb_test = np.load(coco_ann_path + "/embeddings/caption_val.npy")
emb_coco = np.concatenate([emb_train,emb_test],axis=0)
print(emb_coco.shape)
if(use_clip):
    image_id_train = np.load(coco_ann_path + "/embeddings/clip_sample_id_train.npy").astype(np.int64)
    image_id_test = np.load(coco_ann_path + "/embeddings/clip_sample_id_val.npy").astype(np.int64)
    coco_id_train = np.load(coco_ann_path + "/embeddings/clip_id_train.npy").astype(np.int64)
    coco_id_test = np.load(coco_ann_path + "/embeddings/clip_id_val.npy").astype(np.int64)
else:
    image_id_train = np.load(coco_ann_path + "/embeddings/sample_id_train.npy").astype(np.int64)
    image_id_test = np.load(coco_ann_path + "/embeddings/sample_id_val.npy").astype(np.int64)
    coco_id_train = np.load(coco_ann_path + "/embeddings/id_train.npy").astype(np.int64)
    coco_id_test = np.load(coco_ann_path + "/embeddings/id_val.npy").astype(np.int64)
image_id_train = image_id_train*1e6 + coco_id_train
image_id_train = image_id_train.astype(np.int64)
image_id_test = image_id_test*1e6 + coco_id_test
image_id_test = image_id_test.astype(np.int64)
id_coco = np.concatenate([image_id_train,image_id_test],axis=0)
print(id_coco.shape)
train_emb = []
test_emb = []
train_karpathy_id = list(coco_train_dict.keys())
emb_shape = None
for i in tqdm(train_karpathy_id):
    idx = np.nonzero(id_coco==i)[0][0]
    if(emb_shape is not None and emb_shape!=emb_coco[idx].shape):
        print(id_coco[idx],emb_shape,emb_coco[idx].shape)
    train_emb.append(emb_coco[idx])
    emb_shape = emb_coco[idx].shape
train_emb = np.stack(train_emb,axis=0).astype(np.float32)
cocoID_emb_dict = {}
for i in tqdm(range(len(id_coco))):
    cocoID_emb_dict[str(id_coco[i])] = emb_coco[i]

import faiss
if(use_clip):
    dim = 512
else:
    dim = 768
faiss.normalize_L2(train_emb)
index_ip = faiss.IndexFlatIP(dim)
index_ip = faiss.IndexIDMap(index_ip)
ngpus = faiss.get_num_gpus()
print("number of GPUs:", ngpus)
res = [faiss.StandardGpuResources() for i in range(ngpus)]
flat_config = []
for i in range(ngpus):
    cfg = faiss.GpuIndexFlatConfig()
    cfg.device = i
    flat_config.append(cfg)
indexes = [faiss.IndexIDMap(faiss.GpuIndexFlatIP(res[i], dim, flat_config[i])) for i in range(ngpus)]
findex = faiss.IndexShards(dim, True, False)
for sub_index in indexes:
    findex.addIndex(sub_index)
findex.add_with_ids(train_emb,train_karpathy_id)
topK = 30
D, I = findex.search(train_emb, topK)
result_list = []
I = np.array(I)
for i,img_coco_id in enumerate(train_karpathy_id):
    for idx in I[i]:
        result_list.append({'image_id':str(int(idx/1e6)),'coco_id':str(int(str(idx)[-6:])),'caption':coco_train_dict[idx]['caption']})
with open(f"../data/coco/annotations/group_captions_clip_top{topK}.json","w") as f:
    json.dump(result_list,f)


