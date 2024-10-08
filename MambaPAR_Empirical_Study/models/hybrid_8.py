import math
import os
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
from models.vit import *
from timm.models import create_model
from .Vim import VisionMamba
from torch.nn.functional import mse_loss
import copy

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
        self.cls_token_vim = self.vim.cls_token
        self.pos_embed_vim = self.vim.pos_embed
        self.patch_embed_vim = self.vim.patch_embed
        self.pos_drop_vim = self.vim.pos_drop
        self.vim_layers = self.vim.layers

        self.vim_mid = copy.deepcopy(self.vim_layers[:-2:2])
        # breakpoint()


        # 创建Vit
        self.vit = vit_base()

        self.cls_token = self.vit.cls_token
        self.pos_embed = self.vit.pos_embed
        self.patch_embed_vit = self.vit.patch_embed
        self.pos_drop_vit = self.vit.pos_drop
        self.vit_blocks = self.vit.blocks

        
        self.down_adpeters = nn.ModuleList([
            nn.Linear(768,384) for i in range(12)
        ])

        self.up_adpeters = nn.ModuleList([
            nn.Linear(384,768) for i in range(12)
        ])

        # 一维卷积进行通道数降维
        self.conv_vis = nn.Conv1d(in_channels=args.conv_dim,out_channels = attr_num, kernel_size = 1)
        self.weight_layer = nn.ModuleList([nn.Linear(768, 1) for i in range(self.attr_num)])
        self.bn = nn.BatchNorm1d(self.attr_num)

        self.conv_pool_vim = nn.Conv1d(in_channels=args.conv_dim,out_channels = attr_num, kernel_size = 1)
        self.weight_layer_vim = nn.ModuleList([nn.Linear(384, 1) for i in range(self.attr_num)])
        self.bn_vim = nn.BatchNorm1d(self.attr_num)
    
    def forward_once(self,i,x_vit,x_vim,residual = None):
        x_vit = self.vit_blocks[i](x_vit)
        
        # breakpoint()
        x_vim,residual = self.vim_layers[2*i](x_vim,residual)
        x_vim,residual = self.vim_layers[2*i+1](x_vim,residual)

        x_vim_kd = self.vim_mid[i](x_vim,residual)
        x_vim_kd = self.up_adpeters[i](x_vim)

        kdloss = mse_loss(x_vit[:,1:,:],x_vim_kd[:,1:,:])
        # breakpoint()

        x_vim = x_vim + self.down_adpeters[i](x_vim_kd)

        return x_vit,x_vim,residual,kdloss

        

    def forward(self, imgs, text, label=None):

        B = imgs.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_tokens_vim = self.cls_token_vim.expand(B, -1, -1)

        x_vit = self.patch_embed_vit(imgs)
        x_vim =  self.patch_embed_vim(imgs)

        x_vit = torch.cat((cls_tokens, x_vit), dim=1) + self.pos_embed
        x_vim = torch.cat((cls_tokens_vim, x_vim), dim=1) + self.pos_embed_vim

        x_vit = self.pos_drop_vit(x_vit)
        x_vim = self.pos_drop_vim(x_vim)

        residual = None
        KD_loss = []
        for i in range(len(self.vit_blocks) - 1):
            x_vit,x_vim,residual,kdloss = self.forward_once(i,x_vit,x_vim,residual)
            KD_loss.append(kdloss)
            # print(i)
            # breakpoint()

        # x_vim,residual = self.vim_layers[11*2](x_vim,residual)
        # x_vim,residual = self.vim_layers[11*2+1](x_vim,residual)

        x_vit = self.conv_vis(x_vit)
        x_vim = self.conv_pool_vim(x_vim)

            
        logits_vit = torch.cat([self.weight_layer[i](x_vit[:, i, :]) for i in range(self.attr_num)], dim=1)
        logits_vit = self.bn(logits_vit)

        logits_vim = torch.cat([self.weight_layer_vim[i](x_vim[:, i, :]) for i in range(self.attr_num)], dim=1)
        logits_vim = self.bn_vim(logits_vim)

        # breakpoint()
        if KD_loss == []:
            KD_loss = 0.0
        else:
            KD_loss = sum(KD_loss)/len(KD_loss)
        return logits_vit,logits_vim,KD_loss
    
    def load_param(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu')
        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dicts' in param_dict:
            param_dict = param_dict['state_dicts']
        for k, v in param_dict.items():
            if 'head' in k or 'dist' in k:
                continue

            if "vis_embed" in k:
                continue

            if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                # For old models that I trained prior to conv based patchification
                O, I, H, W = self.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            elif k == 'pos_embed' and v.shape != self.pos_embed.shape:
                v = resize_pos_embed(v, self.pos_embed, self.patch_embed.num_y, self.patch_embed.num_x)
            try:
                # breakpoint()
                self.state_dict()[k].copy_(v)
                
            except:
                print('===========================ERROR=========================')
                print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape, self.state_dict()[k].shape))


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
    

    