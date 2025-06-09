# ### VT
# import torch.nn.functional as F
# import torch.nn as nn
# import torch
# from models.vit import *
# from models.vrwkv import *
# from clip import clip
# class TransformerClassifier(nn.Module):
#     def __init__(self, attr_num, dim=768, pretrain_path='/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c0/DATA/songhaoyu/PAR/VRWKV_PAR/checkpoints/jx_vit_base_p16_224-80ecf9dd.pth'):
#         super().__init__()
#         self.attr_num = attr_num
        
#         self.dim = dim
#         vit = vit_base()
#         vit.load_param(pretrain_path)
#         self.vrwkv = VRWKV(args.rwkv_config,args.rwkv_checkpoint)
#         # clip_model, ViT_preprocess = clip.load("/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c0/DATA/songhaoyu/PAR/VRWKV_PAR/checkpoints/ViT-L-14.pt", device="cuda",download_root='/checkpoints') 
#         # self.clip_model = clip_model
#         # self.text_vec = clip.tokenize(attributes).to("cuda")
#         self.word_embed = nn.Linear(768, dim)
#         self.blocks = vit.blocks[-1:]
#         # self.norm = self.vit.norm
#         self.norm = nn.LayerNorm(dim)
#         self.weight_layer = nn.ModuleList([nn.Linear(dim, 1) for i in range(self.attr_num)])
#         self.bn = nn.BatchNorm1d(self.attr_num)

#         self.vis_embed = nn.Parameter(torch.zeros(1, 1, dim))
#         self.tex_embed = nn.Parameter(torch.zeros(1, 1, dim))

#     def forward(self, imgs, word_vec):
#         # features = self.vit(imgs)
#         features = self.vrwkv.forward(imgs).to("cuda").float()
#         # print(features.shape)
#         batch_size = features.shape[0]
#         word_embed = self.word_embed(word_vec).expand(batch_size, self.attr_num, self.dim).to("cuda").float()
#         # word_embed = self.clip_model.encode_text(self.text_vec).expand(batch_size, self.attr_num, self.dim).to("cuda").float()
#         tex_embed = word_embed + self.tex_embed
#         vis_embed = features + self.vis_embed
#         x = torch.cat([tex_embed, vis_embed], dim=1)

#         for blk in self.blocks:
#             x = blk(x)
#         x = self.norm(x)

#         logits = torch.cat([self.weight_layer[i](x[:, i, :]) for i in range(self.attr_num)], dim=1)
#         logits = self.bn(logits)

#         return logits



## 纯视觉模型
import torchvision.models as torch_models
import torch.nn.functional as F
import torch.nn as nn
import torch
from models.vit import *
from models.vrwkv import *
from clip import clip
from mmcls.models import build_backbone, build_classifier
from mmcls_custom.models.backbones.otn_rwkv import OTN_RWKV
from mmcls_custom.models.backbones.GCN import * 
from config import argument_parser
parser = argument_parser()
args = parser.parse_args()
class TransformerClassifier(nn.Module):
    def __init__(self, attr_num, dim=768, pretrain_path='/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c02/DATA/wanghaiyang/baseline/VRWKV_PAR_Backbone/checkpoints/jx_vit_base_p16_224-80ecf9dd.pth'):
        super().__init__()
        self.attr_num = attr_num
        self.dim = dim
        self.mode=args.fusion_mode
        self.backbones=args.backbones

        if self.backbones=='rwkv':
            self.vit = vit_base()
            self.vit.load_param(pretrain_path)

            self.vrwkv_rgb = VRWKV(args.rwkv_config,args.rwkv_checkpoint)
            self.vrwkv_event = VRWKV(args.rwkv_config,args.rwkv_checkpoint)
            self.blocks =self.vit.blocks[-1:]
        elif self.backbones=='vit':
            self.vit_rgb = vit_base()
            self.vit_event = vit_base()
            self.vit_rgb .load_param(pretrain_path)
            self.vit_event.load_param(pretrain_path)
            self.blocks =self.vit_rgb.blocks[-1:]
            
        elif self.backbones=='resnet50':
            self.vit = vit_base()
            self.vit.load_param(pretrain_path)
            self.resnet50 = torch_models.resnet50(pretrained=True)
            self.resnet50 = self.resnet50.eval() 
            self.resnet50_feature_extractor = nn.Sequential(*list(self.resnet50.children())[:-3])
            self.rgb_proj = nn.Linear(1024, 768)
            self.event_proj = nn.Linear(1024, 768)
            self.blocks =self.vit.blocks[-1:]


        self.norm = nn.LayerNorm(dim)
        self.weight_layer = nn.ModuleList([nn.Linear(dim, 1) for i in range(self.attr_num)])
        self.bn = nn.BatchNorm1d(self.attr_num)

        ####################
        self.otn_rwkv=OTN_RWKV(n_embd=768, n_head=12, n_layer=12, layer_id=0, shift_mode='q_shift_multihead',
                               shift_pixel=1, hidden_rate=4, init_mode='fancy', key_norm=False, with_cls_token=False, with_cp=False,patch_resolution=(16,8))
        
       
        self.adaptive_pool = nn.AdaptiveAvgPool1d(output_size=128)

        self.gcn = TokenGCN(num_layers=3, use_knn=True)
       
    def forward(self, imgs, word_vec):

        if self.backbones=='rwkv':
            features_rgb = self.vrwkv_rgb.forward(imgs[0]).to("cuda").float()
            features_event_5 = self.vrwkv_event.forward(imgs[1]).to("cuda").float()
            features_rgb = features_rgb.squeeze(1)

            #similarity
            B,T,N,D=features_event_5.shape
            features_event_5=features_event_5.reshape(B, T * N, D)
            features_event=self.filter_similar_tokens_vectorized_2(features_event_5)
            assert features_event.shape[1]>=128, f'filter features_event token less 128'

            if features_event.shape[1] > 128:
                features_event=features_event.transpose(1, 2)
                features_event=self.adaptive_pool(features_event)
                features_event=features_event.transpose(1, 2)


        elif self.backbones=='resnet50':
            rgb=imgs[0]
            event=imgs[1]
            batch_size, num_frames, channels, height, width = event.size()

            rgb = rgb.squeeze(1)
            event=event.view(-1, channels, height, width)#(DF, C, H, W)

            features_rgb =self.resnet50_feature_extractor(rgb).permute(0, 2, 3, 1)
            features_event = self.resnet50_feature_extractor(event).permute(0, 2, 3, 1)

            DF, H,W, D = features_event.shape
            features_rgb = features_rgb.reshape(batch_size, H * W, D)
            features_event = features_event.reshape(batch_size, num_frames * H * W, D)
            features_rgb=self.rgb_proj(features_rgb)
            features_event=self.event_proj(features_event)


        elif self.backbones=='vit':
            rgb=imgs[0]
            event=imgs[1]
            batch_size, num_frames, channels, height, width = event.size()

            rgb = rgb.squeeze(1)
            event=event.view(-1, channels, height, width)#(DF, C, H, W)

            features_rgb = self.vit_rgb(rgb)[:,1:,:]
            features_event = self.vit_event(event)[:,1:,:]
            
            DF, N, D = features_event.shape
            features_event = features_event.reshape(batch_size,num_frames,  N, D)
            features_event=features_event.reshape(batch_size, num_frames* N, D)

        x=self.otn_rwkv.forward(features_rgb,features_event)

        #通用
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        logits = torch.cat([self.weight_layer[i](x[:, i, :]) for i in range(self.attr_num)], dim=1)
        logits = self.bn(logits)

        return logits
    





    def filter_similar_tokens_vectorized_2(self, x, threshold=0.75, min_tokens=128):
        B, TN, D = x.shape

        # 归一化 token 向量，计算余弦相似度
        x_norm = x / torch.norm(x, dim=-1, keepdim=True)  
        sim_matrix = torch.bmm(x_norm, x_norm.transpose(1, 2))  # 计算余弦相似度矩阵

        # 初始化掩码（全 True）
        mask = torch.ones((B, TN), dtype=torch.bool, device=x.device)

        # 找到相似度高于阈值的 token 对，并去除对角线
        high_sim = sim_matrix > threshold
        high_sim &= ~torch.eye(TN, dtype=torch.bool, device=x.device).unsqueeze(0)

        # 标记需要去除的 token（如果一个 token 与任何其他 token 过于相似，则去除）
        mask &= ~high_sim.any(dim=2)  

        # **确保至少保留 min_tokens 个 token**
        for i in range(B):
            if mask[i].sum() < min_tokens:
                # 计算每个 token 的信息量（通过 L2 范数衡量）
                scores = torch.norm(x[i], dim=-1)  # shape: (TN,)
                # 按信息量排序，并选取最重要的 min_tokens 个
                topk_indices = scores.topk(min_tokens, largest=True).indices
                # 只保留这些 token
                mask[i] = False  # 先全部设为 False
                mask[i][topk_indices] = True  # 只保留最重要的 token

        # 重新筛选 token
        filtered_tokens = [x[i][mask[i]] for i in range(B)]
        filtered_tokens = torch.nn.utils.rnn.pad_sequence(filtered_tokens, batch_first=True)

        return filtered_tokens

    
