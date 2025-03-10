'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import warnings
warnings.filterwarnings("ignore")
from src.vit import VisionTransformer, interpolate_pos_embed
import src.clip_huggingface.modeling_clip as clip
from src.clip_huggingface.utils_clip import interpolate_pos_embed_for_clip
from src.med import BertConfig, BertModel, BertLMHeadModel
from transformers import BertTokenizer
from collections import OrderedDict
from typing import Any, Union, List
import torch
from torch import nn
import torch.nn.functional as F
import os
from urllib.parse import urlparse
from timm.models.hub import download_cached_file
import loralib as lora
from loralib import LoRALayer
class BLIP_Base(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 image_size = 224,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,
                 bert_ckpt_path = None,                 
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer(bert_ckpt_path)   
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)  
    def forward(self, image, caption, mode):
        assert mode in ['image', 'text', 'multimodal'], "mode parameter must be image, text, or multimodal"
        text = self.tokenizer(caption, return_tensors="pt").to(image.device) 
        if mode=='image':    
            image_embeds = self.visual_encoder(image)             
            return image_embeds
        elif mode=='text':
            text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                            return_dict = True, mode = 'text')  
            return text_output.last_hidden_state
        elif mode=='multimodal':
            image_embeds = self.visual_encoder(image)    
            image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)      
            text.input_ids[:,0] = self.tokenizer.enc_token_id
            output = self.text_encoder(text.input_ids,
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,      
                                       return_dict = True,
                                      )              
            return output.last_hidden_state
class BLIP_Decoder(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 image_size = 384,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,
                 prompt = 'a picture of ',
                 bert_ckpt_path = "",
                 vit_ckpt_path = "",
                 clip_ckpt_path = ""
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """            
        super().__init__()
        self.vit = vit
        if("clip" not in vit):
            self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer)
            if('base' in self.vit):
                state_dict = torch.load(vit_ckpt_path,map_location='cpu')['model']
                state_dict['pos_embed'] = interpolate_pos_embed(state_dict['pos_embed'],self.visual_encoder)
                self.visual_encoder.load_state_dict(state_dict,strict=False)
            elif('large' in self.vit):
                from timm.models.helpers import load_custom_pretrained
                from timm.models.vision_transformer import default_cfgs
                vit_cfgs = default_cfgs["vit_large_patch16_224_in21k"]
                vit_cfgs.pop("url")
                vit_cfgs['file'] = vit_ckpt_path
                load_custom_pretrained(self.visual_encoder,vit_cfgs)
        else:
            assert len(clip_ckpt_path) != 0
            self.visual_encoder, vision_width = create_clip(vit,clip_ckpt_path)
        self.tokenizer = init_tokenizer(bert_ckpt_path)   
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        if(bert_ckpt_path):
            self.text_decoder = BertLMHeadModel(config=med_config)
            self.text_decoder.bert = BertModel.from_pretrained(bert_ckpt_path,config=med_config,add_pooling_layer=False)
            self.text_decoder.resize_token_embeddings(len(self.tokenizer)) 
        else:
            self.text_decoder = BertLMHeadModel(config=med_config)
        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids)-1
    def forward(self, image, caption,reduction='mean'):
        if("clip" in self.vit):
            image_embeds = self.visual_encoder(image).last_hidden_state
        else:
            image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        text = self.tokenizer(caption, padding='longest', truncation=True, max_length=40, return_tensors="pt").to(image.device) 
        text.input_ids[:,0] = self.tokenizer.bos_token_id
        decoder_targets = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100)         
        decoder_targets[:,:self.prompt_length] = -100
        decoder_output = self.text_decoder(text.input_ids, 
                                           attention_mask = text.attention_mask, 
                                           encoder_hidden_states = image_embeds,
                                           encoder_attention_mask = image_atts,                  
                                           labels = decoder_targets,
                                           return_dict = True,
                                           reduction = reduction 
                                          )   
        loss_lm = decoder_output.loss
        return loss_lm
    def generate(self, image, sample=False, num_beams=3, max_length=30, min_length=10, top_p=0.9, repetition_penalty=1.0):
        if("clip" in self.vit):
            image_embeds = self.visual_encoder(image).last_hidden_state
        else:
            image_embeds = self.visual_encoder(image)
        if not sample:
            image_embeds = image_embeds.repeat_interleave(num_beams,dim=0)
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask":image_atts}
        prompt = [self.prompt] * image.size(0)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(image.device) 
        input_ids[:,0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1] 
        if sample:
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                  max_length=max_length,
                                                  min_length=min_length,
                                                  do_sample=True,
                                                  top_p=top_p,
                                                  num_return_sequences=1,
                                                  eos_token_id=self.tokenizer.sep_token_id,
                                                  pad_token_id=self.tokenizer.pad_token_id, 
                                                  repetition_penalty=1.1,                                            
                                                  **model_kwargs)
        else:
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                  max_length=max_length,
                                                  min_length=min_length,
                                                  num_beams=num_beams,
                                                  eos_token_id=self.tokenizer.sep_token_id,
                                                  pad_token_id=self.tokenizer.pad_token_id,     
                                                  repetition_penalty=repetition_penalty,
                                                  **model_kwargs)            
        captions = []    
        for output in outputs:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)    
            captions.append(caption[len(self.prompt):])
        return captions
class BLIP_CLIP_Decoder(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 image_size = 384,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,
                 prompt = 'a picture of ',
                 bert_ckpt_path = "",
                 vit_ckpt_path = "",
                 clip_ckpt_path = ""
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """            
        super().__init__()
        self.vit = vit
        if("clip" not in vit):
            self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer)
            if(vit_ckpt_path):
                state_dict = torch.load(vit_ckpt_path,map_location='cpu')['model']
                state_dict['pos_embed'] = interpolate_pos_embed(state_dict['pos_embed'],self.visual_encoder)
                self.visual_encoder.load_state_dict(state_dict,strict=False)
        else:
            assert len(clip_ckpt_path) != 0
            self.visual_encoder, vision_width = create_clip(vit,clip_ckpt_path)
        self.clip_projection = nn.Linear(512,768) 
        self.tokenizer = init_tokenizer(bert_ckpt_path)   
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        if(bert_ckpt_path):
            self.text_decoder = BertLMHeadModel(config=med_config)
            self.text_decoder.bert = BertModel.from_pretrained(bert_ckpt_path,config=med_config,add_pooling_layer=False)
            self.text_decoder.resize_token_embeddings(len(self.tokenizer)) 
        else:
            self.text_decoder = BertLMHeadModel(config=med_config)
        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids)-1
    def forward(self, image, caption, nouns_embeds=None,nouns_mask=None):
        if("clip" in self.vit):
            image_embeds = self.visual_encoder(image).last_hidden_state
        else:
            image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        if(nouns_embeds is not None):
            bs,max_nouns,nouns_dim = nouns_embeds.shape
            nouns_embeds = self.clip_projection(nouns_embeds)
            image_embeds = torch.cat([image_embeds,nouns_embeds],dim=1)
            image_atts = torch.cat([image_atts,nouns_mask],dim=1)
        text = self.tokenizer(caption, padding='longest', truncation=True, max_length=40, return_tensors="pt").to(image.device) 
        text.input_ids[:,0] = self.tokenizer.bos_token_id
        decoder_targets = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100)         
        decoder_targets[:,:self.prompt_length] = -100
        decoder_output = self.text_decoder(text.input_ids, 
                                           attention_mask = text.attention_mask, 
                                           encoder_hidden_states = image_embeds,
                                           encoder_attention_mask = image_atts,                  
                                           labels = decoder_targets,
                                           return_dict = True,   
                                          )   
        loss_lm = decoder_output.loss
        return loss_lm
    def generate(self, image, nouns_embeds=None,sample=False, num_beams=3, max_length=30, min_length=10, top_p=0.9, repetition_penalty=1.0):
        if("clip" in self.vit):
            image_embeds = self.visual_encoder(image).last_hidden_state
        else:
            image_embeds = self.visual_encoder(image)
        if(nouns_embeds is not None):
            bs,max_nouns,nouns_dim = nouns_embeds.shape
            nouns_embeds = self.clip_projection(nouns_embeds)
            image_embeds = torch.cat([image_embeds,nouns_embeds],dim=1)
        if not sample:
            image_embeds = image_embeds.repeat_interleave(num_beams,dim=0)
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask":image_atts}
        prompt = [self.prompt] * image.size(0)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(image.device) 
        input_ids[:,0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1] 
        if sample:
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                  max_length=max_length,
                                                  min_length=min_length,
                                                  do_sample=True,
                                                  top_p=top_p,
                                                  num_return_sequences=1,
                                                  eos_token_id=self.tokenizer.sep_token_id,
                                                  pad_token_id=self.tokenizer.pad_token_id, 
                                                  repetition_penalty=1.1,                                            
                                                  **model_kwargs)
        else:
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                  max_length=max_length,
                                                  min_length=min_length,
                                                  num_beams=num_beams,
                                                  eos_token_id=self.tokenizer.sep_token_id,
                                                  pad_token_id=self.tokenizer.pad_token_id,     
                                                  repetition_penalty=repetition_penalty,
                                                  **model_kwargs)            
        captions = []    
        for output in outputs:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)    
            captions.append(caption[len(self.prompt):])
        return captions
def blip_decoder(pretrained='',**kwargs):
    model = BLIP_Decoder(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        assert(len(msg.missing_keys)==0)
    return model
def clip_encoder_blip_decoder(pretrained='',**kwargs):
    model = BLIP_CLIP_Decoder(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        assert(len(msg.missing_keys)==0)
    return model    
def blip_feature_extractor(pretrained='',**kwargs):
    model = BLIP_Base(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        assert(len(msg.missing_keys)==0)
    return model        
def init_tokenizer(bert_ckpt_path):
    tokenizer = BertTokenizer.from_pretrained(bert_ckpt_path)
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]  
    return tokenizer
def create_vit(vit, image_size, use_grad_checkpointing=False, ckpt_layer=0, drop_path_rate=0):
    assert vit in ['base', 'large'], "vit parameter must be base or large"
    if vit=='base':
        vision_width = 768
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=12, 
                                           num_heads=12, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0 or drop_path_rate
                                          )   
    elif vit=='large':
        vision_width = 1024
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=24, 
                                           num_heads=16, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0.1 or drop_path_rate
                                          )   
    return visual_encoder, vision_width
def mark_only_lora_as_trainable_(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
    if bias == 'none':
        return
    elif bias == 'pe':
        for n, p in model.named_parameters():
            if 'position_embedding' in n:
                p.requires_grad = True
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and \
                hasattr(m, 'bias') and \
                m.bias is not None:
                    m.bias.requires_grad = True
    else:
        raise NotImplementedError
def mark_only_lora_as_untrainable_(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'lora_' in n:
            p.requires_grad = False
    if bias == 'none':
        return
def create_clip(vit,clip_ckpt_path):
    if vit=='clip-base':
        vision_width = 768
    elif vit=='clip-large':
        vision_width = 1024
    visual_encoder = clip.CLIPVisionModel.from_pretrained(clip_ckpt_path)
    return visual_encoder, vision_width
def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")
def load_vit_checkpoint(model,url_or_filename):
    if is_url(url_or_filename):
        cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu') 
    elif os.path.isfile(url_or_filename):        
        checkpoint = torch.load(url_or_filename, map_location='cpu') 
    else:
        raise RuntimeError('checkpoint url or path is invalid')
    state_dict = checkpoint['model']
    new_state_dict = OrderedDict()
    for k,v in state_dict.items():
        new_state_dict['visual_encoder.'+k] = v
    del state_dict
    state_dict = new_state_dict
    state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder) 
    if 'visual_encoder_m.pos_embed' in model.state_dict().keys():
        state_dict['visual_encoder_m.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                                         model.visual_encoder_m)    
    for key in model.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape!=model.state_dict()[key].shape:
                del state_dict[key]
    msg = model.load_state_dict(state_dict,strict=False)
    print('load checkpoint from %s'%url_or_filename)  
def load_checkpoint(model,url_or_filename):
    if is_url(url_or_filename):
        cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu') 
    elif os.path.isfile(url_or_filename):        
        checkpoint = torch.load(url_or_filename, map_location='cpu') 
    else:
        print(url_or_filename)
        raise RuntimeError('checkpoint url or path is invalid')
    state_dict = checkpoint['model']
    if('visual_encoder.pos_embed' in state_dict.keys()):
        state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder) 
        if 'visual_encoder_m.pos_embed' in model.state_dict().keys():
            state_dict['visual_encoder_m.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                                            model.visual_encoder_m)    
        for key in model.state_dict().keys():
            if key in state_dict.keys():
                if state_dict[key].shape!=model.state_dict()[key].shape:
                    del state_dict[key]
        msg = model.load_state_dict(state_dict,strict=False)
        print('load checkpoint from %s'%url_or_filename)  
        return model,msg
    else:
        target_num_patches = 576
        state_dict['visual_encoder.vision_model.embeddings.position_embedding.weight'] = interpolate_pos_embed_for_clip(state_dict['visual_encoder.vision_model.embeddings.position_embedding.weight'],target_num_patches)
        state_dict['visual_encoder.vision_model.embeddings.position_ids'] =  torch.arange(target_num_patches+1).expand((1, -1))
        msg = model.load_state_dict(state_dict,strict=False)
        print('load checkpoint from %s'%url_or_filename)  
        return model,msg
