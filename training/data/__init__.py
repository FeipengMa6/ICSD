import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from data.coco_karpathy_dataset import coco_karpathy_train, coco_karpathy_caption_eval
from data.coco_karpathy_dataset import coco_karpathy_train_syn,coco_karpathy_train_syn_summary
from data.coco_karpathy_dataset import coco_karpathy_train_syn_w_group_id
from data.flickr30k_dataset import flickr_karpathy_caption_eval
from data.nocaps_dataset import nocaps_eval
from data.laion_dataset import laion_train,laion_eval
from data.nocaps_dataset import nocaps_eval
from data.iter_dataloader import make_data_loader
def create_dataset(dataset, config, min_scale=0.5):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    transform_train = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_size'],scale=(min_scale, 1.0),interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])        
    transform_test = transforms.Compose([
        transforms.Resize((config['image_size'],config['image_size']),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])
    clip_transform = transforms.Compose([
        transforms.Resize((224,224),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])
    train_dataset,val_dataset,test_dataset = None,None,None
    if(config['eval_type']=='coco'):
        val_dataset = coco_karpathy_caption_eval(transform_test, config['eval_image_root'], 
                                                config['eval_ann_root'], 'val', 
                                                )
        test_dataset = coco_karpathy_caption_eval(transform_test, config['eval_image_root'], 
                                                config['eval_ann_root'], 'test', 
                                                )
    elif(config['eval_type']=='laion'):
        val_dataset = laion_eval(transform_test, config['eval_image_root'], 
                                config['eval_ann_root'], 'val',
                                return_nouns=config['using_gt_nouns'],
                                )
        test_dataset = laion_eval(transform_test, config['eval_image_root'], 
                                config['eval_ann_root'], 'test',
                                return_nouns=config['using_gt_nouns'],
                                )
    elif(config['eval_type'] == 'flickr'):
        val_dataset = flickr_karpathy_caption_eval(transform_test, config['eval_image_root'], 
                                                config['eval_ann_root'], 'val',
                                                )
        test_dataset = flickr_karpathy_caption_eval(transform_test, config['eval_image_root'], 
                                                config['eval_ann_root'], 'test', 
                                                )
    elif('nocaps' in config['eval_type']):
        if("_" in config['eval_type']):
            domain = config['eval_type'].split("_")[0]
        else:
            domain = "overall"
        val_dataset = nocaps_eval(transform_test, config['eval_image_root'], 
                                                config['eval_ann_root'], domain+'_val',
                                                )
        test_dataset = nocaps_eval(transform_test, config['eval_image_root'], 
                                                config['eval_ann_root'], domain+'_test', 
                                                )
    if(config['train_type'] in ['syn_coco_laion','pretrain']):
        assert len(config['image_root']) == len(config['ann_root'])
        train_dataset_dict = {'coco':coco_karpathy_train_syn,'laion':laion_train,
                              'summary':coco_karpathy_train_syn_summary,
                              'group':coco_karpathy_train_syn_w_group_id}
        subset_type = []
        for ann_r in config['ann_root']:
            if('laion' in ann_r):
                subset_type.append('laion')
            elif('summary' in ann_r):
                subset_type.append('summary')
            elif('group' in ann_r):
                subset_type.append('group')
            else:
                subset_type.append('coco')
        train_subset = [train_dataset_dict[subset_type[i]](transform_train, config['image_root'][i], config['ann_root'][i], prompt=config['prompt'],nsfw_images=config["nsfw_images"][i]) for i in range(len(config['image_root']))]
        train_dataset = train_subset[0]
        for i in range(1,len(config['ann_root'])):
            train_dataset.annotation = train_dataset.annotation + train_subset[i].annotation
    elif(config['train_type'] in ['coco','finetune']):
        assert len(config['image_root']) == len(config['ann_root'])
        train_subset = [coco_karpathy_train(transform_train, config['image_root'][i], config['ann_root'][i], prompt=config['prompt']) for i in range(len(config['image_root']))]
        train_dataset = train_subset[0]
        for i in range(1,len(config['ann_root'])):
            train_dataset.annotation = train_dataset.annotation + train_subset[i].annotation
    elif(config['train_type'] == 'syn_coco'):
        assert len(config['image_root']) == len(config['ann_root'])
        if(len(config['image_root']) == 1):
            train_dataset = coco_karpathy_train_syn(transform_train, config['image_root'][0], config['ann_root'][0], prompt=config['prompt'],nsfw_images=config['nsfw_images'])
        else:
            train_subset = [coco_karpathy_train_syn(transform_train, config['image_root'][i], config['ann_root'][i], prompt=config['prompt'],nsfw_images=config['nsfw_images'][i]) for i in range(len(config['image_root']))]
            train_dataset = train_subset[0]
            for i in range(1,len(config['ann_root'])):
                train_dataset.annotation = train_dataset.annotation + train_subset[i].annotation
    return train_dataset,val_dataset,test_dataset
def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers     
def create_loader(datasets,batch_size,max_epochs,num_workers,is_trains,collate_fns):  
    loaders = []
    for dataset,bs,n_worker,epoch,is_train,collate_fn in zip(datasets,batch_size,max_epochs,num_workers,is_trains,collate_fns):
        loader = None
        loader = make_data_loader(dataset,bs,epoch,n_worker,is_train=is_train,start_iter=0)
        loaders.append(loader)
    return loaders
