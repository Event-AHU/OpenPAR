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

interact_dict={
'default':["0->0", "3->1", "6->2", "9->3","11->4"],
'0':["0->0", "1->1", "2->2", "3->3","4->4"], 
'1':["7->0", "8->1", "9->2", "10->3","11->4"],
'2':["0->0", "3->1", "6->2", "9->3","11->4"], 
'3':["3->1", "6->2", "9->3","11->4"],
'4':["6->2", "9->3","11->4"], 
'5':["9->3","11->4"], '6':["11->4"],
'7':["0->0", "1->1", "2->2", "4->3","6->4","8->5", "10->6", "11->7"]
}

class TransformerClassifier(nn.Module):
    def __init__(self, ViT_model, attr_num,attr_words, dim=args.mmformer_dim, spatial_dim=args.spatial_dim, 
    temporal_dim=args.temporal_dim, pretrain_path='./jx_vit_base_p16_224-80ecf9dd.pth'):
        super().__init__()
        super().__init__()

        self.attr_num = attr_num
        self.dim = dim
        self.using_tem = not args.without_temporal 
        self.using_spa = not args.without_spatial

        if args.dataset == 'MARS':
            self.attr_len = [2,2,2,2,2,2,2,2,2,6,5,9,10,4]
            group = "top length, bottom type, shoulder bag, backpack, hat, hand bag, hair, gender, bottom length, pose, motion, top color, bottom color, age"
        else:
            self.attr_len = [2,2,2,2,2,2,2,6,5,9,8]
            group = "backpack, shoulder bag, hand bag, boots, gender, hat, shoes, top length, pose, motion, top color, bottom color"

        print('Using Temporal Side Net' if self.using_tem else 'Without Temporal Side Net')
        print('Using Spatial Side Net' if self.using_spa else 'Without Spatial Side Net')

        self.spatial_interact_map = interact_dict[args.spatial_interact_map]

        print('-' * 60)
        print(f"The interaction pattern of the spatial side net is {self.spatial_interact_map}")

        self.temporal_interact_map =interact_dict[args.temporal_interact_map]

        print(f"The interaction pattern of the temporal side net is {self.temporal_interact_map}")
        print('-' * 60)
        print(f"The dimension of the spatial/temporal side net is [{args.spatial_dim}] / [{args.temporal_dim}]")
        
        spa_model_name = model_register[args.spatial_dim]
        tem_model_name = model_register[args.temporal_dim]

        print(f"The model of the spatial/temporal side net is [{spa_model_name}] / [{tem_model_name}]")

        model_name = model_register[args.mmformer_dim]

        print(f"The model of the mmformer is {model_name}")

        self.clip_visual_extractor = FeatureExtractor(ViT_model.visual, last_layer_idx=-1, frozen_exclude=["positional_embedding"])
        if self.using_spa:
            self.spatial_sidenet = SideAdapterNetwork(spa_model_name, args.fusion_type, self.spatial_interact_map, [7, 8])

        if self.using_tem:
            self.crossframe_sidenet = CrossFrameSideNetwork(tem_model_name, args.fusion_type, self.temporal_interact_map, [7, 8])

        self.img_type = ViT_model.dtype
        self.word_embed = nn.Linear(512, dim)
        self.visual_embed = nn.Linear(768, dim)
        self.spa_embed = nn.Linear(spatial_dim, dim)
        self.tem_embed = nn.Linear(temporal_dim, dim)
        # mmformer
        vit = vit_base()
        vit.load_param(pretrain_path)
        
        vit_model= create_model(
            model_name,#vit_w240n6d8_patch16 vit_base_patch16_224
            False,#False
            img_size=224,#640-->224
            drop_path_rate=0.1,
            fc_norm=False,
            num_classes=0,
            embed_layer=PatchEmbed,
        )
        
        self.blocks = vit_model.blocks[-1:]
        self.norm = vit_model.norm
        self.weight_layer = nn.ModuleList([nn.Linear(dim, l) for l in self.attr_len])
        self.bn = nn.BatchNorm1d(sum(self.attr_len))
        self.text = clip.tokenize(attr_words).to(device)
        self.w_temporal = nn.Parameter(torch.ones(1))
        input_size = 197 * dim
        print(f"The features aggregation of the spatial/temporal side net is [{args.spatial_feat_aggregation}] / [{args.temporal_feat_aggregation}]")

        if args.spatial_feat_aggregation == 'LSTM':
            self.spatial_feat_aggregation = nn.LSTM(input_size=dim, hidden_size=dim, num_layers=1, batch_first=True)
        elif args.spatial_feat_aggregation == 'GRU':
            self.spatial_feat_aggregation = nn.GRU(input_size=dim, hidden_size=dim, num_layers=1, batch_first=True)
        elif args.spatial_feat_aggregation =='MLP':
            self.spatial_feat_aggregation = nn.Conv1d(args.frames, 1, kernel_size=1)
        else:
            self.spatial_feat_aggregation = nn.AdaptiveAvgPool2d((1,240))

        if args.temporal_feat_aggregation == 'LSTM':
            self.temporal_feat_aggregation = nn.LSTM(input_size=dim, hidden_size=dim, num_layers=1, batch_first=True)
        elif args.temporal_feat_aggregation == 'GRU':
            self.temporal_feat_aggregation = nn.GRU(input_size=dim, hidden_size=dim, num_layers=1, batch_first=True)
        elif args.temporal_feat_aggregation =='MLP':
            self.temporal_feat_aggregation = nn.Conv1d(len(self.temporal_interact_map), 1, kernel_size=1)
        else:
            self.temporal_feat_aggregation = nn.AdaptiveAvgPool2d((1,240))


    def forward(self, videos,ViT_model):
        ViT_features=[]
        if len(videos.size())<5 :
            videos.unsqueeze(1) 

        batch_size, num_frames, channels, height, width = videos.size()
        imgs=videos.view(-1, channels, height, width)
        
        clip_image_features, _ = self.clip_visual_extractor(imgs.type(self.img_type))
        if self.using_spa:
            
            spatial_features = self.spatial_sidenet(imgs,clip_image_features)
            _,L,side_D = spatial_features.size()
            spatial_features = spatial_features.view(batch_size,num_frames,L,side_D)

            fuse_spatial_feat= torch.empty(batch_size, L, side_D).cuda()
            spatial_features = spatial_features.permute(0,2,1,3)
            for bb, batch_spatial_feat in enumerate(spatial_features):
                output = self.spatial_feat_aggregation(batch_spatial_feat)
                fuse_spatial_feat[bb] = output[0][:,-1,:] if isinstance(output, list) else output[:,-1,:]

        if self.using_tem:
            frame_select_feat={k:[] for k in range(num_frames)}
            for layer in self.crossframe_sidenet.fusion_map.values():
                selected_layer_feat = clip_image_features[layer]
                BF,D,H,W = selected_layer_feat.shape
                reshape_feat=selected_layer_feat.view(BF,D,-1).permute(0,2,1).view(batch_size,num_frames,H*W,D).permute(1,0,2,3)
                for fidx,frame_feat in enumerate(reshape_feat) :
                    frame_select_feat[fidx].append(frame_feat.cuda().float())
            temporal_features = self.crossframe_sidenet(frame_select_feat)
            num_select,B,L,side_D = temporal_features.size()
            temporal_features = temporal_features.permute(1,2,0,3)
            fuse_temporal_feat= torch.empty(batch_size, L , side_D).cuda()
            for bb, batch_temporal_feat in enumerate(temporal_features):
                output= self.temporal_feat_aggregation(batch_temporal_feat)
                fuse_temporal_feat[bb] = output[0][:,-1,:] if isinstance(output, list) else output[:,-1,:]
            
        if self.using_tem and self.using_spa:
            fusion_side_feat = fuse_spatial_feat + fuse_temporal_feat
        elif self.using_spa :
            fusion_side_feat = fuse_spatial_feat 
        elif self.using_tem :
            fusion_side_feat = fuse_temporal_feat

        text_features = ViT_model.encode_text(self.text).to(device).float()
        textual_features = self.word_embed(text_features).expand(batch_size, text_features.shape[0],self.dim)  
        x = torch.cat([textual_features, fusion_side_feat], dim=1)

        for b_c, blk in enumerate(self.blocks):
            x = blk(x)
        x = self.norm(x)
        logits = torch.cat([self.weight_layer[i](x[:, i, :]) for i in range(len(self.attr_len))], dim=1)
        logits = self.bn(logits)
        
        return logits
    
