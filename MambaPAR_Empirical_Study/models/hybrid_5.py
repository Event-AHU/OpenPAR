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

        self.vis_embed = nn.Parameter(torch.zeros(1, 1, dim))

        # 创建Vit
        base_pretrain_path='checkpoints/jx_vit_base_p16_224-80ecf9dd.pth'
        self.vit = vit_base()
        self.vit.load_param(base_pretrain_path)

        self.cls_token = self.vit.cls_token
        self.pos_embed = self.vit.pos_embed
        self.patch_embed = self.vit.patch_embed
        self.pos_drop = self.vit.pos_drop
        self.vit_blocks = self.vit.blocks
        self.dim_proj = nn.Linear(768,384)
        self.pools = nn.ModuleList([
            nn.Conv1d(in_channels=args.conv_dim,out_channels = args.conv_dim//4, kernel_size = 1) for i in range(12)
        ])

        self.final_pool = nn.Sequential(
            nn.Conv1d(in_channels=args.conv_dim *11 //4 -2,out_channels = args.conv_dim, kernel_size = 1),
        )

        self.norm = self.vit.norm

        # 一维卷积进行通道数降维
        self.conv_vis = nn.Conv1d(in_channels=args.conv_dim,out_channels = attr_num, kernel_size = 1)

        self.weight_layer = nn.ModuleList([nn.Linear(768, 1) for i in range(self.attr_num)])
        self.bn = nn.BatchNorm1d(self.attr_num)

        self.conv_pool_vim = nn.Conv1d(in_channels=args.conv_dim,out_channels = attr_num, kernel_size = 1)
        self.weight_layer_vim = nn.ModuleList([nn.Linear(384, 1) for i in range(self.attr_num)])
        self.bn_vim = nn.BatchNorm1d(self.attr_num)
        

    def forward(self, imgs, text, label=None):

        vis_features = self.patch_embed(imgs)

        B = vis_features.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        vis_features =  torch.cat((cls_tokens, vis_features), dim=1) + self.pos_embed

        x = self.pos_drop(vis_features)
        out_x = None
        for i in range(len(self.vit_blocks) - 1):
            x = self.vit_blocks[i](x)
            if out_x ==None:
                out_x = self.pools[i](x)
            else:
                out_x = torch.cat([out_x,self.pools[i](x)],dim=1)
            # breakpoint()

        out_x = self.dim_proj(out_x)
        # breakpoint()
        x_vim = self.final_pool(self.vim.forward_by_layers(out_x))
        # breakpoint()
        x_vim = self.conv_pool_vim(x_vim)

        x = self.conv_vis(x)
            
        logits = torch.cat([self.weight_layer[i](x[:, i, :]) for i in range(self.attr_num)], dim=1)
        logits = self.bn(logits)

        logits_vim = torch.cat([self.weight_layer_vim[i](x_vim[:, i, :]) for i in range(self.attr_num)], dim=1)
        logits_vim = self.bn_vim(logits_vim)

        logits = (logits + logits_vim)/2
        # breakpoint()
        return logits   
    
    def load_param(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu')
        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dicts' in param_dict:
            param_dict = param_dict['state_dicts']
        for k, v in param_dict.items():
            if 'head' in k or 'dist' in k:
                continue

            # if "conv_vis" in k:
            #     continue

            # if "weight_layer." in k:
            #     continue
            
            # if "bn." in k:
            #     continue

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
    

    