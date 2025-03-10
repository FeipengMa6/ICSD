import re
import json
import os
import torch
import torch.distributed as dist
import utils
from src.logger import LOGGER as logger
def pre_caption(caption,max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",       
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
    return caption
def pre_question(question,max_ques_words=50):
    question = re.sub(
        r"([.!\"()*#:;~])",
        '',
        question.lower(),
    ) 
    question = question.rstrip(' ')
    question_words = question.split(' ')
    if len(question_words)>max_ques_words:
        question = ' '.join(question_words[:max_ques_words])
    return question
def save_result(result, result_dir, filename, remove_duplicate=''):
    result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,utils.get_rank()))
    final_result_file = os.path.join(result_dir, '%s.json'%filename)
    json.dump(result,open(result_file,'w'))
    dist.barrier()
    if utils.is_main_process():   
        result = []
        for rank in range(utils.get_world_size()):
            result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,rank))
            res = json.load(open(result_file,'r'))
            result += res
        if remove_duplicate:
            result_new = []
            id_list = []    
            for res in result:
                if res[remove_duplicate] not in id_list:
                    id_list.append(res[remove_duplicate])
                    result_new.append(res)
            result = result_new             
        json.dump(result,open(final_result_file,'w'))            
        logger.info('result file saved to %s'%final_result_file)
    return final_result_file
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from torchvision.datasets.utils import download_url
import os
import sys
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
def coco_caption_eval(coco_gt_root, results_file, split,eval_type):
    if(eval_type == 'coco'):
        filenames = {'val':'coco_karpathy_val_gt.json','test':'coco_karpathy_test_gt.json'}
    elif(eval_type == 'laion'):
        filenames = {'val':'laion_karpathy_val_gt.json','test':'laion_karpathy_test_gt.json'}
    elif(eval_type == 'flickr'):
        filenames = {'val':'flickr30k_karpathy_val_gt.json','test':'flickr30k_karpathy_test_gt.json'}
    elif(eval_type == 'nocaps'):
        filenames = {'val':'all_domain_nocaps_gt.json','test':'all_domain_nocaps_gt.json'}
    elif(eval_type == 'in_nocaps'):
        filenames = {'val':'in_domain_nocaps_gt.json','test':'in_domain_nocaps_gt.json'}
    elif(eval_type == 'near_nocaps'):
        filenames = {'val':'near_domain_nocaps_gt.json','test':'near_domain_nocaps_gt.json'}
    elif(eval_type == 'out_nocaps'):
        filenames = {'val':'out_domain_nocaps_gt.json','test':'out_domain_nocaps_gt.json'}
    annotation_file = os.path.join(coco_gt_root,filenames[split])
    with HiddenPrints():
        coco = COCO(annotation_file)
        coco_result = coco.loadRes(results_file)
        coco_eval = COCOEvalCap(coco, coco_result)
        coco_eval.evaluate()
    for metric, score in coco_eval.eval.items():
        logger.info(f'{metric}: {score:.3f}')
    return coco_eval