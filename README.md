# ICSD

## Data
Please put the annotations files into `data` folder.

## Preprocess

### 1. Extract COCO caption features
Extract clip features
```
cd preprocess
python extract_captions_features.py \
        --local_rank 1 \
        --meta_path ../data/coco/annotations/captions_train2017.json \
        --save_path ../data/coco/embeddings \
        --clip_path ../pretrained_models/ViT-B-16.pt \
        --diffcse_path ../pretrained_models/diffcse-roberta-base-trans \
        --bs 128 \
        --clip \
        --split train
python extract_captions_features.py \
        --local_rank 1 \
        --meta_path ../data/coco/annotations/captions_train2017.json \
        --save_path ../data/coco/embeddings \
        --clip_path ../pretrained_models/ViT-B-16.pt \
        --diffcse_path ../pretrained_models/diffcse-roberta-base-trans \
        --bs 128 \
        --clip \
        --split val
```
Or you can extract diffcse features
```
cd preprocess
python  -m torch.distributed.launch --nproc_per_node=8 extract_captions_features.py \
        --meta_path ../data/coco/annotations/captions_train2017.json \
        --save_path ../data/coco/embeddings \
        --clip_path ../pretrained_models/ViT-B-16.pt \
        --diffcse_path ../pretrained_models/diffcse-roberta-base-trans \
        --bs 128 \
        --split train
python -m torch.distributed.launch --nproc_per_node=8  extract_captions_features.py \
        --meta_path ../data/coco/annotations/captions_train2017.json \
        --save_path ../data/coco/embeddings \
        --clip_path ../pretrained_models/ViT-B-16.pt \
        --diffcse_path ../pretrained_models/diffcse-roberta-base-trans \
        --bs 128 \
        --split val
```
### 2. Grouping of Captions
Group the most similar captions in the corpus.
Run the following command will get `group_captions_clip_top30.json` and `coco_karpathy_train_w_group_id.json`
```
cd preprocess
python initial_grouping.py --use_clip --coco_ann_path ../data/coco
python remapping.py
```

### 3. Preprocess for Generation
For using ChatGPT, we should preprocess the `group_captions_clip_top30.json` to call api for selection and summarization.
```
cd preprocess
python preprocess_for_chatgpt.py
```
You will get `coco_chatgpt_train_group_summary.json` after running this process.

## Generation

### 1. Selection & Summarization through ChatGPT
```
cd generation
python sele_and_sum.py
```
You will get `coco_group_summary_train.json`
Then run the postprocess function.
```
cd generation
python postprocess.py
```
You will get `coco_karpathy_train_summary_chatgpt.json` and `coco_karpathy_train_w_group_id_chatgpt.json`  for generating images and training, respectively.

### 2. Generating Images

Generate multi-context images
```
python generate_images.py \
        --coco_split train \
        --batch_size 16 \
        --num_workers 16 \
        --diffusion_model_path ../pretrained_models/stable-diffusion-v1-4 \
        --image_path ../data/coco/group_sum_images \
        --data_type coco \
        --attr summary_chatgpt \
        --nums_gpu 8 \
        --seed 23456
```

Filter nsfw images
```
python verify_nsfw_images.py --folder_name group_sum_images --data_name coco
```

## Training
```
cd training
bash run.sh
```

## Citation
If you find this codebase is useful for your research, please cite us:
```
@inproceedings{ma2024image,
  title={Image captioning with multi-context synthetic data},
  author={Ma, Feipeng and Zhou, Yizhou and Rao, Fengyun and Zhang, Yueyi and Sun, Xiaoyan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={5},
  pages={4089--4097},
  year={2024}
}
```