import torch.nn as nn
import torch
from CLIP.clip import clip
from models.vit import *
from models.visual import *
from models.sidenet import *
from models.sidenet_vit import *
from models.temporal_sidenet import *
from models.CrossFrameSidenet import *
from config import argument_parser

parser = argument_parser()
args = parser.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
model_register = {144: 'vit_w144n6d8_patch16', 240: 'vit_w240n6d8_patch16', 256: 'vit_w256n8d8_patch16',
                336: 'vit_w336n6d8_patch16', 432: 'vit_w432n6d8_patch16', 512: 'vit_w512n8d8_patch16',
                768: 'vit_w768n8d8_patch16'}


class TransformerClassifier(nn.Module):
    def __init__(self, ViT_model, attr_num,attr_words, dim=args.dim):
        super().__init__()
        super().__init__()
        self.attr_num = attr_num
        model_name = model_register[args.dim]
        self.clip_visual_extractor = FeatureExtractor(ViT_model.visual, last_layer_idx=-1, frozen_exclude=["positional_embedding"],)
        self.spatial_sidenet = SideAdapterNetwork(model_name,args.fusion_type, args.fusion_map, [7, 8])
        self.crossframe_sidenet = CrossFrameSideNetwork(model_name,args.fusion_type, args.fusion_map, [7, 8])
        self.img_type = ViT_model.dtype
        self.word_embed = nn.Linear(512, dim)
        self.visual_embed= nn.Linear(768, dim)
       
        vit_model= create_model(
            model_name,
            False,
            img_size=224,
            drop_path_rate=0.0,
            fc_norm=False,
            num_classes=0,
            embed_layer=PatchEmbed,
        )

        self.blocks = vit_model.blocks[-1:]
        self.norm = vit_model.norm
        self.weight_layer = nn.ModuleList([nn.Linear(dim, 1) for i in range(self.attr_num)])
        self.bn = nn.BatchNorm1d(self.attr_num)
        self.text = clip.tokenize(attr_words).to(device)
        self.fusion_linear = nn.Linear(197*2, 197) 
        self.w_temporal = nn.Parameter(torch.ones(1))
        input_size = 197 * dim
        self.spatial_lstm = nn.GRU(input_size=dim, hidden_size=dim, num_layers=1, batch_first=True)
        self.temporal_lstm = nn.GRU(input_size=dim, hidden_size=dim, num_layers=1, batch_first=True)

    def forward(self, videos,ViT_model):
        ViT_features=[]
        if len(videos.size())<5 :
            videos.unsqueeze(1) 

        batch_size, num_frames, channels, height, width = videos.size()
        imgs=videos.view(-1, channels, height, width)
        clip_image_features, _ = self.clip_visual_extractor(imgs.type(self.img_type))
        spatial_features = self.spatial_sidenet(imgs,clip_image_features)
        _,L,side_D = spatial_features.size()
        spatial_features = spatial_features.view(batch_size,num_frames,L,side_D)
        fuse_spatial_feat = spatial_features.mean(dim=1)
        frame_select_feat={k:[] for k in range(num_frames)}

        for layer in self.spatial_sidenet.fusion_map.values():
            selected_layer_feat = clip_image_features[layer]
            BF,D,H,W = selected_layer_feat.shape
            reshape_feat=selected_layer_feat.view(BF,D,-1).permute(0,2,1).view(batch_size,num_frames,H*W,D).permute(1,0,2,3)
            for fidx,frame_feat in enumerate(reshape_feat) :
                frame_select_feat[fidx].append(frame_feat.cuda().float())
        
        temporal_features = self.crossframe_sidenet(frame_select_feat)
        fuse_temporal_feat = temporal_features.mean(dim=0)
        fusion_side_feat = fuse_spatial_feat + fuse_temporal_feat
        text_features = ViT_model.encode_text(self.text).to(device).float()
        textual_features = self.word_embed(text_features).expand(batch_size, text_features.shape[0],side_D)  
        x = torch.cat([textual_features, fusion_side_feat], dim=1)
        for b_c, blk in enumerate(self.blocks):
            x,attn_map = blk(x)
        x = self.norm(x)

        logits = torch.cat([self.weight_layer[i](x[:, i, :]) for i in range(self.attr_num)], dim=1)
        logits = self.bn(logits)
        
        return logits
    
