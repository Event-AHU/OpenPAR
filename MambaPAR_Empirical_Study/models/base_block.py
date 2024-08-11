import math
import os
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
from models.vit import *
from timm.models import create_model

from models.vmamba import VSSM
from .Vim import VisionMamba
from mamba_ssm import MambaLMHeadModel
from transformers import AutoTokenizer
import copy
from mamba_ssm.ops.triton.layernorm import RMSNorm

class MambaClassifier(nn.Module):
    def __init__(self,args, attr_num, words, dim=768, pretrain_path='checkpoints/jx_vit_base_p16_224-80ecf9dd.pth'):
        super().__init__()
        self.args = args
        self.attr_num = attr_num
        self.dim = dim
        
        if self.args.use_Vis_model == "Vit":
            base_pretrain_path='checkpoints/jx_vit_base_p16_224-80ecf9dd.pth'
            self.vit = vit_base()
            self.vit.load_param(base_pretrain_path)
        elif self.args.use_Vis_model == "Vim":
            # 创建Vim
            self.vim = self.create_Vim(args)
            # 图片特征维度映射器
            self.vim_proj = nn.Linear(in_features=384,out_features=dim)
            # 图片位置编码
        elif self.args.use_Vis_model == "VMamba":
            self.vmamba = self.build_vssm_models(cfg = "vssm_base")
            self.vm_proj = nn.Linear(in_features=1024,out_features=dim)
            self.flat = nn.Flatten(1,2)

        self.vis_embed = nn.Parameter(torch.zeros(1, 1, dim))

        if not args.only_img:
            # 储存标签文本
            self.words = words
            # 文本特征维度映射器
            self.word_proj = nn.Linear(args.proj_text_dim, dim)
            # 文本位置编码
            self.tex_embed = nn.Parameter(torch.zeros(1, 1, dim))

            if not args.Text_not_train:
                # 创建mamba
                self.mamba = MambaLMHeadModel.from_pretrained("./checkpoints/mamba-130M/", dtype=torch.float16, device="cuda")

                # 创建tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained("./checkpoints/bert-base-cased")

            if not args.no_VSF:
                # 深度复制生成mamba块，用作VSF mamba
                if args.Text_not_train:
                    self.mamba = MambaLMHeadModel.from_pretrained("./checkpoints/mamba-130M/", dtype=torch.float16, device="cuda")
                self.VSF_mamba = copy.deepcopy(self.mamba.backbone.layers[-2:])
                self.VSF_mamba = self.VSF_mamba.float()
                self.norm = RMSNorm(hidden_size= dim)
            
            else:
                # 一维卷积进行特征融合
                self.conv_mm = nn.Conv1d(in_channels=args.conv_dim+attr_num,out_channels = attr_num, kernel_size = 1)
        else:
            # 一维卷积对图片特征进行映射
            self.conv_vis = nn.Conv1d(in_channels=args.conv_dim,out_channels = attr_num, kernel_size = 1)

        self.weight_layer = nn.ModuleList([nn.Linear(dim, 1) for i in range(self.attr_num)])
        self.bn = nn.BatchNorm1d(self.attr_num)

        

    def get_label_embeds(self):
        tokens = self.tokenizer(self.words,return_tensors="pt",padding='max_length',max_length=10)
        input_ids = tokens.input_ids.to(device="cuda")
        embeddings = self.mamba.backbone(input_ids, inference_params=None)
        return embeddings

    def forward(self, imgs, text, label=None):
        # 提取视觉特征
        if self.args.use_Vis_model == "Vit_bs":
            vis_features = self.vit(imgs) # (B,129,768)
        elif self.args.use_Vis_model == "Vim":
            vis_features = self.vim.forward_features(imgs) # (B,197,384)
            vis_features = self.vim_proj(vis_features)
        elif self.args.use_Vis_model == "VMamba":
            vis_features = self.vmamba(imgs)
            vis_features = self.vm_proj(self.flat(vis_features))

        # 加入位置编码
        vis_embed = vis_features + self.vis_embed    

        if not self.args.only_img:
            
            if not self.args.Text_not_train:
                # 提取文本特征
                word_vec = self.get_label_embeds()
                word_vec = word_vec.mean(1)# (num_class, 768)
            else:
                word_vec = text
            word_embed = self.word_proj(word_vec.float()).expand(vis_features.shape[0], word_vec.shape[0], vis_features.shape[-1])
            # 加入位置编码
            tex_embed = word_embed + self.tex_embed
            

            # 视觉文本特征拼接
            text_vis = torch.cat([vis_embed, tex_embed], dim=1)
            
            if not self.args.no_VSF:
                # 特征对齐
                x = self.VSF_mamba[0](text_vis)[0] + text_vis
                x = self.VSF_mamba[1](x)[0] + x
                x = self.norm(x)
            else:
                x = self.conv_mm(text_vis)
            
        else:
            x = self.conv_vis(vis_embed)

        x = x[:,-self.attr_num:,:]
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
    

    def build_vssm_models(self,cfg="vssm_base", ckpt=True, only_backbone=False, with_norm=True,
    CFGS = dict(
        vssm_tiny=dict(
            model=dict(
                depths=[2, 2, 9, 2], 
                dims=96, 
                d_state=16, 
                dt_rank="auto", 
                ssm_ratio=2.0, 
                attn_drop_rate=0., 
                drop_rate=0., 
                drop_path_rate=0.1, 
                mlp_ratio=0.0,
                downsample_version="v1",
            ), 
            ckpt=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../ckpts/classification/vssm/vssmtiny/ckpt_epoch_292.pth"),
        ),
        vssm_small=dict(
            model=dict(
                depths=[2, 2, 27, 2], 
                dims=96, 
                d_state=16, 
                dt_rank="auto", 
                ssm_ratio=2.0, 
                attn_drop_rate=0., 
                drop_rate=0., 
                drop_path_rate=0.3, 
                mlp_ratio=0.0,
                downsample_version="v1",
            ), 
            ckpt=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../ckpts/classification/vssm/vssmsmall/ema_ckpt_epoch_238.pth"),
        ),
        vssm_base=dict(
            model=dict(
                depths=[2, 2, 27, 2], 
                dims=128, 
                d_state=16, 
                dt_rank="auto", 
                ssm_ratio=2.0, 
                attn_drop_rate=0., 
                drop_rate=0., 
                drop_path_rate=0.6, 
                mlp_ratio=0.0,
                downsample_version="v1",
            ),  
            ckpt=os.path.join(os.path.dirname(os.path.abspath(__file__)), "/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c0/DATA/kongweizhe/PARMamba/VTB-main/checkpoints/vssmbase_dp06_ckpt_epoch_241.pth"),
        ),
    ),
        ckpt_key="model",**kwargs):
        if cfg not in CFGS:
            return None
        
        model_params = CFGS[cfg]["model"]
        model_ckpt = CFGS[cfg]["ckpt"]

        model = VSSM(**model_params)
        if only_backbone:
            if with_norm:
                def forward(self: VSSM, x: torch.Tensor):
                    x = self.patch_embed(x)
                    for layer in self.layers:
                        x = layer(x)
                    x = self.classifier.norm(x)
                    x = x.permute(0, 3, 1, 2).contiguous()
                    return x
                model.forward = partial(forward, model)
                del model.classifier.norm
                del model.classifier.head
                del model.classifier.avgpool
            else:
                def forward(self: VSSM, x: torch.Tensor):
                    x = self.patch_embed(x)
                    for layer in self.layers:
                        x = layer(x)
                    x = x.permute(0, 3, 1, 2).contiguous()
                    return x
                model.forward = partial(forward, model)
                del model.classifier

        if ckpt:
            ckpt = model_ckpt
            try:
                _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
                print(f"Successfully load ckpt {ckpt}")
                incompatibleKeys = model.load_state_dict(_ckpt[ckpt_key], strict=False)
                print(incompatibleKeys)        
            except Exception as e:
                print(f"Failed loading checkpoint form {ckpt}: {e}")

        return model