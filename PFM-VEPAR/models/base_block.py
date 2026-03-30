import math

import torch
import torch.nn as nn
import torch.nn.init as init

import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from models.registry import CLASSIFIER
from tools.dct_utils import apply_frequency_filter
from models.adaptive_cross_attention import AdaptiveCrossAttentionFusion
from models.hopfield.hopfield_layer import HopfieldLayerEnhanced
from models.hopfield.hopfield_layer import ExternalMemoryHopfield


def build_norm_layer(num_features, norm_layer, data_format_in='channels_first', data_format_out='channels_first'):
    """构建归一化层"""
    if norm_layer == 'BN':
        return nn.BatchNorm2d(num_features)
    elif norm_layer == 'LN':
        return nn.LayerNorm(num_features)
    else:
        return nn.Identity()


def build_act_layer(act_layer):

    if act_layer == 'GELU':
        return nn.GELU()
    elif act_layer == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_layer == 'SiLU':
        return nn.SiLU(inplace=True)
    else:
        return nn.Identity()


class StemLayer(nn.Module):
    """
    Stem layer for event stream processing
    Args:
        in_chans (int): number of input channels
        out_chans (int): number of output channels  
        act_layer (str): activation layer
        norm_layer (str): normalization layer
    """
    def __init__(self, in_chans=3, out_chans=96, act_layer='GELU', norm_layer='BN'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans // 2, kernel_size=3, stride=2, padding=1)
        self.norm1 = build_norm_layer(out_chans // 2, norm_layer, 'channels_first', 'channels_first')
        self.act = build_act_layer(act_layer)
        self.conv2 = nn.Conv2d(out_chans // 2, out_chans, kernel_size=3, stride=2, padding=1)
        self.norm2 = build_norm_layer(out_chans, norm_layer, 'channels_first', 'channels_first')

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x


class DCTProcessor(nn.Module):
    def __init__(self, enable_dct=True, apply_to_rgb=False, apply_to_event=True):
        super(DCTProcessor, self).__init__()
        self.enable_dct = enable_dct
        self.apply_to_rgb = apply_to_rgb
        self.apply_to_event = apply_to_event
    
    def forward(self, rgb_data, event_data):
        if not self.enable_dct:
            return rgb_data, event_data
        if self.apply_to_rgb and rgb_data is not None:
            rgb_data = apply_frequency_filter(rgb_data)
        if self.apply_to_event and event_data is not None:
            event_data = apply_frequency_filter(event_data)
        
        return rgb_data, event_data


class BaseClassifier(nn.Module):

    def fresh_params(self, bn_wd):
        if bn_wd:
            return self.parameters()
        else:
            return self.named_parameters()

@CLASSIFIER.register("linear")
class LinearClassifier(BaseClassifier):
    def __init__(self, nattr, c_in, bn=False, pool='avg', scale=1):
        super().__init__()

        self.pool = pool
        if pool == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif pool == 'max':
            self.pool = nn.AdaptiveMaxPool2d(1)

        self.logits = nn.Sequential(
            nn.Linear(c_in, nattr),
            nn.BatchNorm1d(nattr) if bn else nn.Identity()
        )


    def forward(self, feature, label=None):

        if len(feature.shape) == 3:  # for vit (bt, nattr, c)

            bt, hw, c = feature.shape
            # NOTE ONLY USED FOR INPUT SIZE (256, 192)
            h = 16
            w = 12
            feature = feature.reshape(bt, h, w, c).permute(0, 3, 1, 2)

        feat = self.pool(feature).view(feature.size(0), -1)
        x = self.logits(feat)

        return [x], feature



@CLASSIFIER.register("cosine")
class NormClassifier(BaseClassifier):
    def __init__(self, nattr, c_in, bn=False, pool='avg', scale=30):
        super().__init__()

        self.logits = nn.Parameter(torch.FloatTensor(nattr, c_in))

        stdv = 1. / math.sqrt(self.logits.data.size(1))
        self.logits.data.uniform_(-stdv, stdv)

        self.pool = pool
        if pool == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif pool == 'max':
            self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, feature, label=None):
        feat = self.pool(feature).view(feature.size(0), -1)
        feat_n = F.normalize(feat, dim=1)
        weight_n = F.normalize(self.logits, dim=1)
        x = torch.matmul(feat_n, weight_n.t())
        return [x], feat_n


def initialize_weights(module):
    for m in module.children():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, _BatchNorm):
            m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.uniform_(-stdv, stdv)


class FeatClassifier(nn.Module):

    def __init__(self, backbone, classifier, bn_wd=True, enable_dct=True, dct_on_rgb=False, dct_on_event=True, 
                 enable_stem=True, stem_out_chans=96, enable_cross_attention=False, cross_attn_layers=2, cross_attn_heads=8, 
                 fusion_type='adaptive',  enable_hopfield=False, hopfield_apply_to_rgb=True, hopfield_apply_to_event=True,
                 hopfield_n_prototype=1000, hopfield_dropout=0.1, hopfield_temperature=None, cfg=None):
        super(FeatClassifier, self).__init__()

        self.backbone = backbone
        self.classifier = classifier
        self.bn_wd = bn_wd
        self.enable_stem = enable_stem
        self.enable_cross_attention = enable_cross_attention
        self.fusion_type = fusion_type
        self.enable_hopfield = enable_hopfield
        if self.enable_hopfield:
            self.hopfield_apply_to_rgb = hopfield_apply_to_rgb
            self.hopfield_apply_to_event = hopfield_apply_to_event
            if hopfield_apply_to_rgb:
                self.hop_rgb = HopfieldLayerEnhanced(
                    dim=768, n_prototype=hopfield_n_prototype, dropout=hopfield_dropout,
                    temperature=hopfield_temperature
                )
            if hopfield_apply_to_event:
                self.hop_event = HopfieldLayerEnhanced(
                    dim=768, n_prototype=hopfield_n_prototype, dropout=hopfield_dropout,
                    temperature=hopfield_temperature
                )

        if self.enable_stem:
            self.event_stem = StemLayer(
                in_chans=3, 
                out_chans=stem_out_chans,
                act_layer='GELU',
                norm_layer='BN'
            )

            self.stem_dct_processor = DCTProcessor(
                enable_dct=enable_dct,
                apply_to_rgb=False,
                apply_to_event=dct_on_event
            )

            self.stem_projection = nn.Conv2d(stem_out_chans, 3, kernel_size=1, stride=1, padding=0)

        if self.enable_cross_attention:
            if fusion_type == 'adaptive':
                self.cross_attention_fusion = AdaptiveCrossAttentionFusion(
                    d_model=768,
                    nhead=cross_attn_heads,
                    num_layers=cross_attn_layers,
                    dim_feedforward=2048,
                    dropout=0.1
                )

        self.enable_post_fusion_hopfield = getattr(cfg, 'POST_FUSION_HOPFIELD', {}).get('ENABLE', False) if cfg is not None else False
        if self.enable_post_fusion_hopfield:
            if self.enable_cross_attention and self.fusion_type == 'adaptive':
                post_fusion_dim = 768
            elif self.enable_cross_attention:
                post_fusion_dim = 768 * 2
            else:
                post_fusion_dim = 768 * 2
            
            self.post_fusion_hopfield = HopfieldLayerEnhanced(
                dim=post_fusion_dim,
                n_prototype=getattr(cfg, 'POST_FUSION_HOPFIELD', {}).get('N_PROTOTYPE', 1000),
                dropout=getattr(cfg, 'POST_FUSION_HOPFIELD', {}).get('DROPOUT', 0.1),
                temperature=getattr(cfg, 'POST_FUSION_HOPFIELD', {}).get('TEMPERATURE', None)
            )

        self.enable_mem_hop = getattr(cfg, 'MEMORY_HOPFIELD', {}).get('ENABLE', False) if cfg is not None else False
        if self.enable_mem_hop:
            mem_cfg = getattr(cfg, 'MEMORY_HOPFIELD', {})
            self.mem_hop_fusion = mem_cfg.get('FUSION', 'add')
            mem_temp = mem_cfg.get('TEMPERATURE', None)
            mem_dropout = mem_cfg.get('DROPOUT', 0.1)
            mem_freeze = mem_cfg.get('FREEZE', True)

            if mem_cfg.get('APPLY_TO_RGB', True):
                self.hop_rgb_mem = ExternalMemoryHopfield(
                    dim=768, dropout=mem_dropout, temperature=mem_temp,
                    proj_qkv=True, freeze_memory=mem_freeze
                )
                rgb_mem_path = mem_cfg.get('RGB_BANK_PATH', '')
                if rgb_mem_path:
                    self.hop_rgb_mem.load_memory(rgb_mem_path)

            if mem_cfg.get('APPLY_TO_EVENT', True):
                self.hop_event_mem = ExternalMemoryHopfield(
                    dim=768, dropout=mem_dropout, temperature=mem_temp,
                    proj_qkv=True, freeze_memory=mem_freeze
                )
                event_mem_path = mem_cfg.get('EVENT_BANK_PATH', '')
                if event_mem_path:
                    self.hop_event_mem.load_memory(event_mem_path)

            if self.mem_hop_fusion == 'concat':
                self.mem_fuse_proj_rgb = nn.Linear(768 * 2, 768)
                self.mem_fuse_proj_event = nn.Linear(768 * 2, 768)

            self.enable_cross_modal_memory = mem_cfg.get('CROSS_MODAL_INTERACTION', False)
            if self.enable_cross_modal_memory:
                self.consistency_module = nn.Sequential(
                    nn.Linear(768 * 2, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1),
                    nn.Sigmoid()
                )
                self.cross_modal_weight = nn.Parameter(torch.tensor(0.5))

    

    def fresh_params(self):
        params = list(self.classifier.fresh_params(self.bn_wd))
        
        if self.enable_stem:
            if self.bn_wd:
                params.extend(list(self.event_stem.parameters()))
                params.extend(list(self.stem_dct_processor.parameters()))
                params.extend(list(self.stem_projection.parameters()))
            else:
                params.extend(list(self.event_stem.named_parameters()))
                params.extend(list(self.stem_dct_processor.named_parameters()))
                params.extend(list(self.stem_projection.named_parameters()))

        elif self.enable_cross_attention:
            if self.bn_wd:
                params.extend(list(self.cross_attention_fusion.parameters()))
            else:
                params.extend(list(self.cross_attention_fusion.named_parameters()))

        if hasattr(self, 'final_modal_fusion') and self.final_modal_fusion is not None:
            if self.bn_wd:
                params.extend(list(self.final_modal_fusion.parameters()))
            else:
                params.extend(list(self.final_modal_fusion.named_parameters()))

        if self.enable_post_fusion_hopfield:
            if self.bn_wd:
                params.extend(list(self.post_fusion_hopfield.parameters()))
            else:
                params.extend(list(self.post_fusion_hopfield.named_parameters()))
            
        return params

    def finetune_params(self):
        """返回预训练backbone的参数"""
        if self.bn_wd:
            return self.backbone.parameters()
        else:
            return self.backbone.named_parameters()

    def forward(self, image_rgb, image_event, label=None):
        image_rgb = image_rgb.squeeze(1)
        batch_size, num_frames, channels, height, width = image_event.size()

        image_event = image_event.view(-1, channels, height, width)  # (B*F, C, H, W)
        if self.enable_stem:
            event_stem_features = self.event_stem(image_event)  # [B*F, stem_out_chans, H//4, W//4]

            _, event_stem_features = self.stem_dct_processor(None, event_stem_features)

            event_processed = self.stem_projection(event_stem_features)  # [B*F, 3, H//4, W//4]
            event_processed = F.interpolate(
                event_processed, 
                size=(height, width), 
                mode='bilinear', 
                align_corners=False
            )
        else:
            event_processed = image_event

        use_prompt_path = (
            hasattr(self.backbone, 'forward_with_prompts') and hasattr(self.backbone, 'patch_embed') and
            hasattr(self, 'cfg') and getattr(getattr(self.cfg, 'PROMPT', {}), 'ENABLE', False)
        )
        if use_prompt_path:
            event_patches = self.backbone.patch_embed(event_processed)  # [B*F, Np, D]
            _, Np, Demb = event_patches.shape
            event_patches = event_patches.view(batch_size, num_frames, Np, Demb).mean(dim=1)  # [B, Np, D]

            cfg_obj = getattr(self, 'cfg', None)
            P = getattr(getattr(cfg_obj, 'PROMPT', {}), 'NUM_TOKENS', 12)
            insert_layers = getattr(getattr(cfg_obj, 'PROMPT', {}), 'INSERT_LAYERS', [3, 6, 9])
            remove_after_block = getattr(getattr(cfg_obj, 'PROMPT', {}), 'REMOVE_AFTER_BLOCK', True)
            self.prompt_num_tokens = getattr(self, 'prompt_num_tokens', P)
            self.prompt_insert_layers = getattr(self, 'prompt_insert_layers', insert_layers)
            if not hasattr(self, 'prompt_reduce'):
                self.prompt_reduce = nn.Linear(Np, self.prompt_num_tokens)
            event_prompts = self.prompt_reduce(event_patches.transpose(1, 2)).transpose(1, 2)  # [B, P, D]
            fused_map = self.backbone.forward_with_prompts(
                image_rgb,
                prompt_tokens=event_prompts,
                insert_layers=self.prompt_insert_layers,
                remove_after_block=remove_after_block,
            )  # [B, N, D]
            if self.enable_hopfield and hasattr(self, 'hop_rgb'):
                fused_map = self.hop_rgb(fused_map)
            mem_rgb = self.hop_rgb_mem(fused_map) if hasattr(self, 'hop_rgb_mem') else None
            mem_event = self.hop_event_mem(fused_map) if hasattr(self, 'hop_event_mem') else None
            if self.enable_cross_attention and (mem_rgb is not None) and (mem_event is not None):
                if self.fusion_type == 'adaptive':
                    feat_map = self.cross_attention_fusion(mem_rgb, mem_event)  # [B, N, D]
                # else:
                #     feat_map = self.cross_attention_fusion(mem_rgb, mem_event)  # [B, N, 2*D]
            else:
                feat_map = fused_map
        else:
            rgb_feat_map = self.backbone(image_rgb)         # [B, N, D]
            event_feat_map = self.backbone(event_processed) # [B*F, N, D]
            _, N, D = rgb_feat_map.shape
            event_features = event_feat_map.view(batch_size, num_frames, N, D).mean(dim=1)

            if self.enable_hopfield:
                if hasattr(self, 'hop_rgb'):
                    rgb_feat_map = self.hop_rgb(rgb_feat_map)
                if hasattr(self, 'hop_event'):
                    event_features = self.hop_event(event_features)

            if hasattr(self, 'enable_mem_hop') and self.enable_mem_hop:
                if hasattr(self, 'hop_rgb_mem') and hasattr(self, 'hop_event_mem'):
                    if getattr(self, 'enable_cross_modal_memory', False):
                        mem_rgb, rgb_attn = self.hop_rgb_mem.forward_with_attention(rgb_feat_map)
                        mem_event, event_attn = self.hop_event_mem.forward_with_attention(event_features)
                        rgb_attn_mean = rgb_attn.mean(dim=1)
                        event_attn_mean = event_attn.mean(dim=1)
                        attn_similarity = torch.cosine_similarity(rgb_attn_mean, event_attn_mean, dim=-1)
                        consistency_weight = self.consistency_module(
                            torch.cat([mem_rgb.mean(dim=1), mem_event.mean(dim=1)], dim=-1)
                        ).squeeze(-1)
                        adaptive_weight = self.cross_modal_weight * consistency_weight.unsqueeze(-1).unsqueeze(-1)
                        rgb_feat_map = rgb_feat_map + adaptive_weight * mem_rgb + (1 - adaptive_weight) * mem_event
                        event_features = event_features + (1 - adaptive_weight) * mem_rgb + adaptive_weight * mem_event
                    else:
                        mem_rgb = self.hop_rgb_mem(rgb_feat_map)
                        mem_event = self.hop_event_mem(event_features)
                        if getattr(self, 'mem_hop_fusion', 'add') == 'add':
                            rgb_feat_map = rgb_feat_map + mem_rgb
                            event_features = event_features + mem_event
                        else:
                            if not hasattr(self, 'mem_fuse_proj_rgb'):
                                self.mem_fuse_proj_rgb = nn.Linear(768 * 2, 768)
                            if not hasattr(self, 'mem_fuse_proj_event'):
                                self.mem_fuse_proj_event = nn.Linear(768 * 2, 768)
                            rgb_feat_map = self.mem_fuse_proj_rgb(torch.cat([rgb_feat_map, mem_rgb], dim=-1))
                            event_features = self.mem_fuse_proj_event(torch.cat([event_features, mem_event], dim=-1))
                else:
                    if hasattr(self, 'hop_rgb_mem'):
                        mem_rgb = self.hop_rgb_mem(rgb_feat_map)
                        if getattr(self, 'mem_hop_fusion', 'add') == 'add':
                            rgb_feat_map = rgb_feat_map + mem_rgb
                        else:
                            if not hasattr(self, 'mem_fuse_proj_rgb'):
                                self.mem_fuse_proj_rgb = nn.Linear(768 * 2, 768)
                            rgb_feat_map = self.mem_fuse_proj_rgb(torch.cat([rgb_feat_map, mem_rgb], dim=-1))
                    if hasattr(self, 'hop_event_mem'):
                        mem_event = self.hop_event_mem(event_features)
                        if getattr(self, 'mem_hop_fusion', 'add') == 'add':
                            event_features = event_features + mem_event
                        else:
                            if not hasattr(self, 'mem_fuse_proj_event'):
                                self.mem_fuse_proj_event = nn.Linear(768 * 2, 768)
                            event_features = self.mem_fuse_proj_event(torch.cat([event_features, mem_event], dim=-1))

            if self.enable_cross_attention:
                if self.fusion_type == 'adaptive':
                    feat_map = self.cross_attention_fusion(rgb_feat_map, event_features)  # [B, N, D]
                # else:
                #     feat_map = self.cross_attention_fusion(rgb_feat_map, event_features)  # [B, N, 2*D]
            else:
                feat_map = torch.cat([rgb_feat_map, event_features], dim=-1)  # [B, N, 2*D]

        if self.enable_hopfield:
            if hasattr(self, 'hop_rgb'):
                rgb_feat_map = self.hop_rgb(rgb_feat_map)
            if hasattr(self, 'hop_event'):
                event_features = self.hop_event(event_features)

        if hasattr(self, 'enable_mem_hop') and self.enable_mem_hop:
            if hasattr(self, 'hop_rgb_mem') and hasattr(self, 'hop_event_mem'):
                if getattr(self, 'enable_cross_modal_memory', False):
                    mem_rgb, rgb_attn = self.hop_rgb_mem.forward_with_attention(rgb_feat_map)
                    mem_event, event_attn = self.hop_event_mem.forward_with_attention(event_features)

                    rgb_attn_mean = rgb_attn.mean(dim=1)  # [B, M]
                    event_attn_mean = event_attn.mean(dim=1)  # [B, M]
                    attn_similarity = torch.cosine_similarity(rgb_attn_mean, event_attn_mean, dim=-1)  # [B]
                    consistency_weight = self.consistency_module(
                        torch.cat([mem_rgb.mean(dim=1), mem_event.mean(dim=1)], dim=-1)
                    ).squeeze(-1)  # [B]
                    adaptive_weight = self.cross_modal_weight * consistency_weight.unsqueeze(-1).unsqueeze(-1)
                    rgb_feat_map = rgb_feat_map + adaptive_weight * mem_rgb + (1 - adaptive_weight) * mem_event
                    event_features = event_features + (1 - adaptive_weight) * mem_rgb + adaptive_weight * mem_event
                else:
                    mem_rgb = self.hop_rgb_mem(rgb_feat_map)
                    mem_event = self.hop_event_mem(event_features)
                    if getattr(self, 'mem_hop_fusion', 'add') == 'add':
                        rgb_feat_map = rgb_feat_map + mem_rgb
                        event_features = event_features + mem_event
                    else:
                        if not hasattr(self, 'mem_fuse_proj_rgb'):
                            self.mem_fuse_proj_rgb = nn.Linear(768 * 2, 768)
                        if not hasattr(self, 'mem_fuse_proj_event'):
                            self.mem_fuse_proj_event = nn.Linear(768 * 2, 768)
                        rgb_feat_map = self.mem_fuse_proj_rgb(torch.cat([rgb_feat_map, mem_rgb], dim=-1))
                        event_features = self.mem_fuse_proj_event(torch.cat([event_features, mem_event], dim=-1))
            else:
                if hasattr(self, 'hop_rgb_mem'):
                    mem_rgb = self.hop_rgb_mem(rgb_feat_map)
                    if getattr(self, 'mem_hop_fusion', 'add') == 'add':
                        rgb_feat_map = rgb_feat_map + mem_rgb
                    else:
                        if not hasattr(self, 'mem_fuse_proj_rgb'):
                            self.mem_fuse_proj_rgb = nn.Linear(768 * 2, 768)
                        rgb_feat_map = self.mem_fuse_proj_rgb(torch.cat([rgb_feat_map, mem_rgb], dim=-1))

                if hasattr(self, 'hop_event_mem'):
                    mem_event = self.hop_event_mem(event_features)
                    if getattr(self, 'mem_hop_fusion', 'add') == 'add':
                        event_features = event_features + mem_event
                    else:
                        if not hasattr(self, 'mem_fuse_proj_event'):
                            self.mem_fuse_proj_event = nn.Linear(768 * 2, 768)
                        event_features = self.mem_fuse_proj_event(torch.cat([event_features, mem_event], dim=-1))

        if self.enable_cross_attention:
            if self.fusion_type == 'adaptive':
                feat_map = self.cross_attention_fusion(rgb_feat_map, event_features)  # [B, N, D]
            # else:
            #     feat_map = self.cross_attention_fusion(rgb_feat_map, event_features)  # [B, N, 2*D]
        else:
            feat_map = torch.cat([rgb_feat_map, event_features], dim=-1)  # [B, N, 2*D]

        if self.enable_post_fusion_hopfield:
            feat_map = self.post_fusion_hopfield(feat_map)
        
        logits, feat = self.classifier(feat_map, label)
        return logits, feat
