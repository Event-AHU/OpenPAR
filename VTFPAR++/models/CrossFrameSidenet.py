import logging
from functools import partial
from typing import Dict, List, Tuple

import torch
from timm import create_model
from timm.models.vision_transformer import VisionTransformer
from torch import nn
from torch.nn import functional as F

from models.layers import MLP, build_fusion_layer
from models.timm_wrapper import PatchEmbed
from models.sidenet_vit import *

class CrossFrameSideNetwork(nn.Module):
    def __init__(self, model_name, fusion_type, fusion_map, depth=8):
        super().__init__()
        
        vit_model= create_model(
            model_name,#vit_w240n6d8_patch16 vit_base_patch16_224
            False,#False
            img_size=224,#640-->224
            drop_path_rate=0.0,#0.0
            fc_norm=False,
            num_classes=0,
            embed_layer=PatchEmbed,
        )

        if vit_model.cls_token is not None:
            vit_model.pos_embed = nn.Parameter(vit_model.pos_embed[:, 1:, ...])
        del vit_model.cls_token
        vit_model.cls_token = None

        del vit_model.norm
        vit_model.norm = nn.Identity()

        self.vit_model = vit_model
        self.num_features = vit_model.num_features
        self.cls_embed = nn.Parameter(torch.zeros(1, 1, self.num_features))
        self.cls_pos_embed = nn.Parameter(
            torch.zeros(1, 1, self.num_features)
        )
        nn.init.normal_(self.cls_embed, std=0.02)
        nn.init.normal_(self.cls_pos_embed, std=0.02)
        self.fusion_map = {int(j): int(i) for i, j in [x.split("->") for x in fusion_map]}
        fusion_type: str = 'cross1d'
        self.frist_linear=nn.Linear(768,self.num_features)
        fusion_layers = nn.ModuleDict(
            {
                f"layer_{tgt_idx}": build_fusion_layer(
                    fusion_type, 768, vit_model.num_features
                )   
                for tgt_idx in range(len(self.fusion_map.keys()))
            }
        )
        self.fusion_layers = fusion_layers
        
        self.projection_layers = nn.ModuleList([nn.Linear(self.num_features, self.num_features) for _ in range(len(self.fusion_map.keys()))])
        

    def forward(self, clip_features):
        x = clip_features[0]
        B,L,D=clip_features[0][0].shape
        pos_embed = self.vit_model.pos_embed 
        pos_embed = torch.cat(
            [self.cls_pos_embed.expand(pos_embed.shape[0], -1, -1), pos_embed], dim=1
        )

        x = torch.empty(len(clip_features[0]), B, L + 1, self.num_features).cuda()
        
        for frame_idx in clip_features.keys():
            feat_list = clip_features[frame_idx]
            blk = self.vit_model.blocks[frame_idx]
            if frame_idx == 0:
                for select_layer in range(len(feat_list)):
                    feat_list[select_layer] = torch.cat([self.cls_embed.expand(B, -1, -1), 
                                                        self.frist_linear(feat_list[select_layer].float())], dim=1) + pos_embed
                    feat_list[select_layer] = self.vit_model.norm_pre(feat_list[select_layer])
                    x[select_layer] = blk(feat_list[select_layer])
            else: 
                updated_x = x.clone()  
                for select_layer in range(len(feat_list)):
                    updated_x[select_layer] = blk(x[select_layer])
                    updated_x[select_layer] = self.fuse(select_layer, updated_x[select_layer], feat_list[select_layer])
                x = updated_x  
            
        return x

    def fuse(self,layer_idx,x,select_feat) :
        fusion_features = self.fusion_layers[f"layer_{layer_idx}"](x[:, 1:, ...], select_feat)
        x = torch.cat(
            [
                x[:, :1, ...],
                fusion_features,
            ],
            dim=1,
        )
        return x
