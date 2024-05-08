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

class SideAdapterNetwork(nn.Module):
    def __init__(self, model_name, fusion_type, fusion_map, deep_supervision_idxs, depth=8):
        super().__init__()
        
        vit_model= create_model(
            model_name,
            False,
            img_size=224,
            drop_path_rate=0.0,
            fc_norm=False,
            num_classes=0,
            embed_layer=PatchEmbed,
        )
        self.depth = depth
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
        
        fusion_type: str = 'add'
        fusion_layers = nn.ModuleDict(
            {
                f"layer_{tgt_idx}": build_fusion_layer(
                    fusion_type, 768, vit_model.num_features
                )
                for tgt_idx, src_idx in self.fusion_map.items()
            }
        )
        self.fusion_layers = fusion_layers
        self.deep_supervision_idxs = deep_supervision_idxs

    def forward(
        self, image, clip_features):
        features, hydra_features= self.forward_features(image, clip_features)
        return features


    def forward_features(self, image, clip_features) :
        x, (h, w) = self.vit_model.patch_embed(image) 
        L = x.shape[1]  
        pos_embed = self.vit_model.pos_embed 
        ori_h, ori_w = self.vit_model.patch_embed.grid_size
        if pos_embed.shape[1] != L: 
            pos_embed = (
                F.interpolate(
                    pos_embed.reshape(1, ori_h, ori_w, -1).permute(0, 3, 1, 2),
                    size=[h, w],
                    mode="bicubic",
                    align_corners=False,
                )
                .flatten(2)
                .permute(0, 2, 1)
            )
        pos_embed = torch.cat(
            [self.cls_pos_embed.expand(pos_embed.shape[0], -1, -1), pos_embed], dim=1
        )
        x = torch.cat(
            [self.cls_embed.expand(x.shape[0], -1, -1), x],
            dim=1,
        )  
        x = x + pos_embed
        x = self.vit_model.norm_pre(x)
        hydra_x=[]
        x = self.fuse(0, x, clip_features, (h, w))
        hydra_x.append(x)
        outs = []
        
        for i, blk in enumerate(self.vit_model.blocks[:self.depth-1], start=1):
            x = blk(x)
            x = self.fuse(i, x, clip_features, (h, w))
            if i in self.fusion_map:
                hydra_x.append(x)
            if i in self.deep_supervision_idxs:
                outs = x
            if i < len(self.vit_model.blocks):
                x = x + pos_embed
        return outs, torch.stack(hydra_x)

    def fuse(self,block_idx,x,clip_features,spatial_shape,) :
        
        if block_idx in self.fusion_map: 
            src_idx = self.fusion_map[block_idx] 
            fusion_features = self.fusion_layers[f"layer_{block_idx}"](x[:, 1:, ...], clip_features[src_idx], spatial_shape)
            x = torch.cat(
                [
                    x[:, :1, ...],
                    fusion_features,
                ],
                dim=1,
            )
        return x
