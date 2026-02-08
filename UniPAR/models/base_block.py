import math
import torch.nn.functional as F
import torch.nn as nn
import torch
from models.vit import *
from models.PatchEmbedding import MultiModalPatchEmbedding

class ClassifierHead(nn.Module):
    # def __init__(self, dim):
    #     super().__init__()
    #     self.weight_layer_PA100k = nn.ModuleList([nn.Linear(dim, 1) for _ in range(26)])
    #     self.weight_layer_DUKE = nn.ModuleList([nn.Linear(dim, 1) for _ in range(36)])
    #     self.weight_layer_EventPAR = nn.ModuleList([nn.Linear(dim, 1) for _ in range(50)])
    #     self.bn_PA100k = nn.BatchNorm1d(26)
    #     self.bn_DUKE = nn.BatchNorm1d(36)
    #     self.bn_EventPAR = nn.BatchNorm1d(50)

    def __init__(self, dim):
        super().__init__()
        self.weight_layer_PA100k = nn.ModuleList([nn.Linear(dim, 1) for _ in range(26)])
        self.weight_layer_DUKE = nn.ModuleList([nn.Linear(dim, 1) for _ in range(36)])
        self.weight_layer_EventPAR = nn.ModuleList([nn.Linear(dim, 1) for _ in range(50)])
        self.weight_layer_MSP60k = nn.ModuleList([nn.Linear(dim, 1) for _ in range(57)])
        self.bn_PA100k = nn.BatchNorm1d(26)
        self.bn_DUKE = nn.BatchNorm1d(36)
        self.bn_EventPAR = nn.BatchNorm1d(50)
        self.bn_MSP60k = nn.BatchNorm1d(57)
    
    def forward(self, x, word_vec):
        if word_vec.shape[0] == 26:
            logits = torch.cat([self.weight_layer_PA100k[i](x[:, i, :]) for i in range(26)], dim=1)
            logits = self.bn_PA100k(logits)
        elif word_vec.shape[0] == 57:
            logits = torch.cat([self.weight_layer_MSP60k[i](x[:, i, :]) for i in range(57)], dim=1)
            logits = self.bn_MSP60k(logits)
        elif word_vec.shape[0] == 36:
            logits = torch.cat([self.weight_layer_DUKE[i](x[:, i, :]) for i in range(36)], dim=1)
            logits = self.bn_DUKE(logits)
        elif word_vec.shape[0] == 50:
            logits = torch.cat([self.weight_layer_EventPAR[i](x[:, i, :]) for i in range(50)], dim=1)
            logits = self.bn_EventPAR(logits)
        return logits

class TransformerClassifier(nn.Module):
    def __init__(self, dim=768, pretrain_path="/wangx/DATA/Code/xujiarui/jx_vit_base_p16_224-80ecf9dd.pth", args=None):
        super().__init__()
        self.word_embed = nn.Linear(768, dim)

        self.vit = vit_base()
        self.vit.load_param(pretrain_path)
        self.patch_embed = MultiModalPatchEmbedding(args=args) # args=args
        if args.PE_load:
            self.patch_embed.load_param(pretrain_path)
        self.encoder = self.vit.blocks
        self.blocks = self.vit.blocks[-1:]
        self.norm = self.vit.norm
        self.decoder = ClassifierHead(dim)

        self.vis_embed = nn.Parameter(torch.zeros(1, 1, dim))
        self.tex_embed = nn.Parameter(torch.zeros(1, 1, dim))

    #  Vison+Text
    def forward(self, imgs, events, word_vec, label=None):

        feature = self.patch_embed(imgs, events)
        for blk in self.encoder[:-1]:
            feature = blk(feature)
        
        word_embed = self.word_embed(word_vec).expand(feature.shape[0], word_vec.shape[0], feature.shape[-1])
        
        tex_embed = word_embed + self.tex_embed
        vis_embed = feature + self.vis_embed

        x = torch.cat([tex_embed, vis_embed], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        logits = self.decoder(x, word_vec)

        return logits

    
    def load_param_add(self, pretrain_path):
        param_dict = torch.load(pretrain_path, map_location='cpu',weights_only=False)
        # print("Keys in param_dict:", list(param_dict.keys()))  # 打印顶级键
        if 'state_dicts' in param_dict:
            param_dict = param_dict['state_dicts']
            # print("Keys in param_dict:", list(param_dict.keys()))  # 打印顶级键
        for k, v in param_dict.items():
            if k.startswith('vit.') or k.startswith('patch_embed.') or k.startswith('encoder.'):
                # print(k)
                continue
            # elif k.startswith('decoder.'):
            #     k = k[8:]
            #     self.state_dict()[k].copy_(v)
            #     print(k,'lord succsess')
            else:
                self.state_dict()[k].copy_(v)
                print(k,'lord succsess')


class VisualOnlyTransformerClassifier(nn.Module):
    def __init__(self, dim=768, pretrain_path="/wangx/DATA/Code/xujiarui/jx_vit_base_p16_224-80ecf9dd.pth", args=None):
        super().__init__()

        # 视觉组件保持不变
        self.vit = vit_base()
        self.vit.load_param(pretrain_path)
        self.patch_embed = MultiModalPatchEmbedding(args=args)
        if args.PE_load:
            self.patch_embed.load_param(pretrain_path)

        # 使用所有encoder blocks
        self.encoder = self.vit.blocks
        self.norm = self.vit.norm

        # 修改ClassifierHead以适应只有视觉输入
        self.decoder = VisualOnlyClassifierHead(dim)


    def forward(self, imgs, events, word_vec, label=None):
        # 提取视觉特征（MultiModalPatchEmbedding会自动添加CLS token）
        feature = self.patch_embed(imgs, events)

        # 通过所有Transformer层
        for blk in self.encoder:
            feature = blk(feature)

        # LayerNorm
        x = self.norm(feature)

        # 分类预测（使用CLS token，它是第一个token）
        logits = self.decoder(x, word_vec)

        return logits


class VisualOnlyClassifierHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 保持原有的分类器结构
        self.weight_layer_PA100k = nn.ModuleList([nn.Linear(dim, 1) for _ in range(26)])
        self.weight_layer_DUKE = nn.ModuleList([nn.Linear(dim, 1) for _ in range(36)])
        self.weight_layer_EventPAR = nn.ModuleList([nn.Linear(dim, 1) for _ in range(50)])
        self.weight_layer_MSP60k = nn.ModuleList([nn.Linear(dim, 1) for _ in range(57)])

        self.bn_PA100k = nn.BatchNorm1d(26)
        self.bn_DUKE = nn.BatchNorm1d(36)
        self.bn_EventPAR = nn.BatchNorm1d(50)
        self.bn_MSP60k = nn.BatchNorm1d(57)

    def forward(self, x, word_vec):
        # 使用CLS token（第一个token）
        # x的形状: [batch, num_tokens, dim]
        # 第一个token是CLS token（来自MultiModalPatchEmbedding）
        cls_token = x[:, 0, :]  # [batch, dim]
        cls_token_expanded = cls_token.unsqueeze(1)  # [batch, 1, dim]
        cls_token_expanded = cls_token_expanded.expand(-1, word_vec.shape[0], -1)  # [batch, num_attr, dim]

        if word_vec.shape[0] == 26:
            logits = torch.cat([self.weight_layer_PA100k[i](cls_token_expanded[:, i, :]) for i in range(26)], dim=1)
            logits = self.bn_PA100k(logits)
        elif word_vec.shape[0] == 57:
            logits = torch.cat([self.weight_layer_MSP60k[i](cls_token_expanded[:, i, :]) for i in range(57)], dim=1)
            logits = self.bn_MSP60k(logits)
        elif word_vec.shape[0] == 36:
            logits = torch.cat([self.weight_layer_DUKE[i](cls_token_expanded[:, i, :]) for i in range(36)], dim=1)
            logits = self.bn_DUKE(logits)
        elif word_vec.shape[0] == 50:
            logits = torch.cat([self.weight_layer_EventPAR[i](cls_token_expanded[:, i, :]) for i in range(50)], dim=1)
            logits = self.bn_EventPAR(logits)

        return logits