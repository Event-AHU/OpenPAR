import math
import torch.nn.functional as F
import torch.nn as nn
import torch
from models.vit import *
from models.vrwkv import *
from hflayers import Hopfield
from config import argument_parser
parser = argument_parser()
args = parser.parse_args()
class TransformerClassifier(nn.Module):
    def __init__(self, attr_num, dim=768, pretrain_path='/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c0/DATA/songhaoyu/PAR/VRWKV_PAR/checkpoints/jx_vit_base_p16_224-80ecf9dd.pth'):
        super().__init__()
        self.attr_num = attr_num
        self.word_embed = nn.Linear(768, dim)

        # self.vit = vit_base()
        # self.vit.load_param(pretrain_path)
        # self.blocks = self.vit.blocks[-1:]
        self.vrwkv = VRWKV(args.rwkv_config,args.rwkv_checkpoint)
        vit = vit_base()
        vit.load_param(pretrain_path)
        self.blocks = vit.blocks[-1:]
        self.norm = nn.LayerNorm(dim)
        self.weight_layer = nn.ModuleList([nn.Linear(dim, 1) for i in range(self.attr_num)])
        self.bn = nn.BatchNorm1d(self.attr_num)

        self.vis_embed = nn.Parameter(torch.zeros(1, 1, dim))
        self.tex_embed = nn.Parameter(torch.zeros(1, 1, dim))

        #hop
        self.hopfield = Hopfield(input_size=768,
                                  scaling=22.3,
                                  normalize_hopfield_space=False,
                                  normalize_hopfield_space_affine=False,
                                  normalize_pattern_projection=False,
                                  normalize_pattern_projection_affine=False,
                                  normalize_state_pattern=False,
                                  normalize_state_pattern_affine=False,
                                  normalize_stored_pattern=False,
                                  normalize_stored_pattern_affine=False,
                                  state_pattern_as_static=True,
                                  pattern_projection_as_static=True,
                                  stored_pattern_as_static=True,
                                  disable_out_projection=True,
                                  num_heads=1,
                                  dropout=False)

    def forward(self, imgs,img_temps,word_vec, label=None):
        device = next(self.parameters()).device
        imgs = imgs.to(device)
        img_temps = img_temps.to(device) # [16, 5, 3, 224, 224]
        word_vec = word_vec.to(device)
        if label is not None:
            label = label.to(device)
        b_s=imgs.size(0)
        # features = self.vit(imgs)
        features = self.vrwkv.forward(imgs).to(device).float()
        # stored_pattern=[self.vit(img_temps[i,:,:]) for i in range(b_s)]
        stored_pattern=[self.vrwkv.forward(img_temps[i,:,:]) for i in range(b_s)]
        stored_pattern=torch.stack(stored_pattern)[:,:,0,:]  #[16, 5, 768]
        word_embed = self.word_embed(word_vec).expand(features.shape[0], word_vec.shape[0], features.shape[-1])
        
        tex_embed = word_embed + self.tex_embed#[bs,35,768]
        vis_embed = features + self.vis_embed
        vis_embed=vis_embed[:,0,:]#[bs,768]
        vis_embed_hop=self.hopfield((stored_pattern,vis_embed.unsqueeze(1).expand(-1,stored_pattern.size(1),-1),vis_embed.unsqueeze(1).expand(-1,stored_pattern.size(1),-1)))
        vis_embed_hop=torch.cat([vis_embed_hop,features],dim=1)
        tex_embed_hop1 = self.hopfield((tex_embed[:, 0:8, :], 
                                        vis_embed.unsqueeze(1).expand(-1, tex_embed[:, 0:8, :].size(1), -1), 
                                        vis_embed.unsqueeze(1).expand(-1, tex_embed[:, 0:8, :].size(1), -1)))

        tex_embed_hop2 = self.hopfield((tex_embed[:, 8:16, :], 
                                        vis_embed.unsqueeze(1).expand(-1, tex_embed[:, 8:16, :].size(1), -1), 
                                        vis_embed.unsqueeze(1).expand(-1, tex_embed[:, 8:16, :].size(1), -1)))

        tex_embed_hop3 = self.hopfield((tex_embed[:, 16:24, :], 
                                        vis_embed.unsqueeze(1).expand(-1, tex_embed[:, 16:24, :].size(1), -1), 
                                        vis_embed.unsqueeze(1).expand(-1, tex_embed[:, 16:24, :].size(1), -1)))

        tex_embed_hop4 = self.hopfield((tex_embed[:, 24:35, :], 
                                        vis_embed.unsqueeze(1).expand(-1, tex_embed[:, 24:35, :].size(1), -1), 
                                        vis_embed.unsqueeze(1).expand(-1, tex_embed[:, 24:35, :].size(1), -1)))
        tex_embed_hop=torch.cat([tex_embed_hop1,tex_embed_hop2,tex_embed_hop3,tex_embed_hop4],dim=1)
        #tex_embed_hop=torch.cat([tex_embed_hop,tex_embed],dim=1)
        x = torch.cat([tex_embed_hop, vis_embed_hop], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        logits = torch.cat([self.weight_layer[i](x[:, i, :]) for i in range(self.attr_num)], dim=1)
        logits = self.bn(logits)

        return logits