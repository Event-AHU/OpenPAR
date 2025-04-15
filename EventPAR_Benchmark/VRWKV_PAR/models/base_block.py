import torch.nn.functional as F
import torch.nn as nn
import torch
from models.vit import *
from models.vrwkv import *
from clip import clip
from mmcls.models import build_backbone, build_classifier
from mmcls_custom.models.backbones.otn_rwkv import OTN_RWKV

from config import argument_parser
parser = argument_parser()
args = parser.parse_args()
class TransformerClassifier(nn.Module):
    def __init__(self, attr_num, dim=768, pretrain_path='/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c01/DATA/wanghaiyang/baseline/VRWKV_PAR_fusion/checkpoints/jx_vit_base_p16_224-80ecf9dd.pth'):
        super().__init__()
        self.attr_num = attr_num
        self.dim = dim
        self.mode=args.fusion_mode

        vit = vit_base()
        vit.load_param(pretrain_path)
        self.vrwkv_rgb = VRWKV(args.rwkv_config,args.rwkv_checkpoint)
        self.vrwkv_event = VRWKV(args.rwkv_config,args.rwkv_checkpoint)
        self.blocks = vit.blocks[-1:]
        self.norm = nn.LayerNorm(dim)
        self.weight_layer = nn.ModuleList([nn.Linear(dim, 1) for i in range(self.attr_num)])
        self.bn = nn.BatchNorm1d(self.attr_num)

        self.otn_rwkv=OTN_RWKV(n_embd=768, n_head=12, n_layer=12, layer_id=0, shift_mode='q_shift_multihead',
                               shift_pixel=1, hidden_rate=4, init_mode='fancy', key_norm=False, with_cls_token=False, with_cp=False,patch_resolution=(16,8))
        self.adaptive_pool = nn.AdaptiveAvgPool1d(output_size=128)

    def forward(self, imgs, word_vec):

    
        features_rgb = self.vrwkv_rgb.forward(imgs[0]).to("cuda").float()
        features_event = self.vrwkv_event.forward(imgs[1]).to("cuda").float()
        features_rgb=features_rgb.squeeze(1)

        B,T,N,D=features_event.shape
        features_event=features_event.reshape(B, T * N, D)
        features_event=self.filter_similar_tokens_vectorized(features_event)
        assert features_event.shape[1]>=128, f'filter features_event token less 128'

        if features_event.shape[1] > 128:
            features_event=features_event.transpose(1, 2)
            features_event=self.adaptive_pool(features_event)
            features_event=features_event.transpose(1, 2)
   
        if self.mode=='otn_rwkv':
            x=self.otn_rwkv.forward(features_rgb,features_event)
    
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        logits = torch.cat([self.weight_layer[i](x[:, i, :]) for i in range(self.attr_num)], dim=1)
        logits = self.bn(logits)

        return logits
    



    def filter_similar_tokens_vectorized(self, x, threshold=0.75, min_tokens=128):
        B, TN, D = x.shape
        x_norm = x / torch.norm(x, dim=-1, keepdim=True)  
        sim_matrix = torch.bmm(x_norm, x_norm.transpose(1, 2)) 
       
        mask = torch.ones((B, TN), dtype=torch.bool, device=x.device)

        high_sim = sim_matrix > threshold
        high_sim &= ~torch.eye(TN, dtype=torch.bool, device=x.device).unsqueeze(0)

        
        mask &= ~high_sim.any(dim=2)  

        
        for i in range(B):
            if mask[i].sum() < min_tokens:
                scores = torch.norm(x[i], dim=-1)           
                topk_indices = scores.topk(min_tokens, largest=True).indices            
                mask[i] = False  
                mask[i][topk_indices] = True  
        filtered_tokens = [x[i][mask[i]] for i in range(B)]
        filtered_tokens = torch.nn.utils.rnn.pad_sequence(filtered_tokens, batch_first=True)

        return filtered_tokens
  