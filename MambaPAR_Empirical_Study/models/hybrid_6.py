import math
import os
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
from models.vit import *
from timm.models import create_model
from .Vim import VisionMamba

class HybridClassifier(nn.Module):
    def __init__(self,args, attr_num, words, dim=768):
        super().__init__()
        self.args = args
        self.attr_num = attr_num
        self.dim = dim

        # 创建Vim
        self.vim = self.create_Vim(args)
        # 图片特征维度映射器
        self.vim_proj = nn.Linear(in_features=384,out_features=dim)
        self.pos_embed_vim = nn.Parameter(torch.zeros(1, args.conv_dim-1, 384))
        self.patch_embed_vim = self.vim.patch_embed
        self.pos_drop_vim = self.vim.pos_drop
        self.vim_layers = self.vim.layers


        # 创建Vit
        base_pretrain_path='checkpoints/jx_vit_base_p16_224-80ecf9dd.pth'
        self.vit = vit_base()
        self.vit.load_param(base_pretrain_path)

        self.pos_embed_vit = nn.Parameter(torch.zeros(1, args.conv_dim-1, dim))
        self.patch_embed_vit = self.vit.patch_embed
        self.pos_drop_vit = self.vit.pos_drop
        self.vit_blocks = self.vit.blocks

        
        self.adpeters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1152,32),
                nn.Linear(32,32),
                nn.Linear(32,1152)
            ) for i in range(12)
        ])


        self.up_final = nn.Linear(384,768)

        # 一维卷积进行通道数降维
        self.conv_pool = nn.Conv1d(in_channels=args.conv_dim-1,out_channels = attr_num, kernel_size = 1)

        self.weight_layer = nn.ModuleList([nn.Linear(768, 1) for i in range(self.attr_num)])
        self.bn = nn.BatchNorm1d(self.attr_num)
    
    def forward_once(self,i,x_vit,x_vim,residual = None):
        x_vit = self.vit_blocks[i](x_vit)
        x_roi = x_vit
        
        x_vim,residual = self.vim_layers[2*i](x_vim,residual)
        x_vim,residual = self.vim_layers[2*i+1](x_vim,residual)

        x_roim = x_vim
        
        x_mix = torch.cat([x_vit,x_vim],dim=-1)
        x_mix = self.adpeters[i](x_mix)

        x_vit = x_mix[:,:,:768]+x_roi
        x_vim = x_mix[:,:,768:]

        x_vim = x_vim + x_roim
        # breakpoint()

        return x_vit,x_vim,residual

        

    def forward(self, imgs, text, label=None):

        x_vit = self.patch_embed_vit(imgs)
        x_vim =  self.patch_embed_vim(imgs)

        x_vit = x_vit + self.pos_embed_vit
        x_vim = x_vim + self.pos_embed_vim

        x_vit = self.pos_drop_vit(x_vit)
        x_vim = self.pos_drop_vim(x_vim)


        residual = None
        for i in range(len(self.vit_blocks)):
            x_vit,x_vim,residual = self.forward_once(i,x_vit,x_vim,residual)


        x = x_vit + self.up_final(x_vim)

        x = self.conv_pool(x)
            
        logits = torch.cat([self.weight_layer[i](x[:, i, :]) for i in range(self.attr_num)], dim=1)
        logits = self.bn(logits)
        return logits   
    
    def create_Vim(self,args):
        model =create_model(
                    args.Vim,
                    pretrained=False,
                    num_classes=args.nb_classes,
                    drop_rate=args.drop,
                    drop_path_rate=args.drop_path,
                    drop_block_rate=None,
                    img_size=args.input_size
                )
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed

        model.load_state_dict(checkpoint_model, strict=False)

        return model
    

    