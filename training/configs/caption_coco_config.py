root_dir = "../"
image_root = [
    f'{root_dir}data/coco/syn_images',
]
ann_root = [
    f'{root_dir}data/coco/annotations/ss1m_karpathy_train_w_coco_id_filter.json',
] 
nsfw_images = [
    f'{root_dir}data/coco/annotations/nsfw_of_syn_images.txt',
]
eval_ann_root = f'{root_dir}data/coco/annotations'
eval_image_root = f'{root_dir}data/coco/images'
vocab = f'{root_dir}data/coco/annotations/coco_karpathy_train_vocab.txt'
using_gt_nouns = False
topk = 30
fp16=False
dataset_type = 'unused'
dataset_size = 'base' 
train_type = 'syn_coco_laion'
eval_type = 'out_nocaps' 
max_epoch = 30
num_workers = 16
save_steps = 500
vit = 'clip-base'
vit_grad_ckpt = False
vit_ckpt_layer = 0
batch_size = 64
init_lr = 1e-5 
image_size = 384
max_length = 30  
min_length = 5
num_beams = 3
prompt = 'a picture of '
pretrained = ''
weight_decay = 0.05
min_lr = 0
med_config = 'configs/bert_config.json' 
vit_ckpt_path = f'{root_dir}pretrained_models/deit_base_patch16_224-b5f2ef4d.pth'
bert_ckpt_path = f'{root_dir}pretrained_models/bert-base-uncased'
clip_ckpt_path = f'{root_dir}pretrained_models/clip-vit-base-patch32-384px' 
oa_clip_ckpt_path = f'{root_dir}pretrained_models/ViT-B-16.pt' 
