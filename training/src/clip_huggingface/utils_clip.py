import torch
def interpolate_pos_embed_for_clip(pos_embed_checkpoint, target_num_patches):        
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = target_num_patches 
    num_extra_tokens = target_num_patches+1 - num_patches 
    orig_size = int((pos_embed_checkpoint.shape[-2]-num_extra_tokens) ** 0.5)
    new_size = int(num_patches ** 0.5)
    if orig_size!=new_size:
        extra_tokens = pos_embed_checkpoint[:num_extra_tokens]
        pos_tokens = pos_embed_checkpoint[num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens.unsqueeze(0), pos_tokens), dim=1)
        print('reshape position embedding from %d to %d'%(orig_size ** 2,new_size ** 2))
        return new_pos_embed.squeeze(0)
    else:
        return pos_embed_checkpoint