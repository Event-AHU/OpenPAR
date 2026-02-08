# import math
# import torch.nn.functional as F
# import torch.nn as nn
# import torch
# from einops import rearrange


# class MultiModalPatchEmbedding(nn.Module):
#     def __init__(self, img_size=(256, 128), patch_size=(16, 16), in_channels=3, 
#                  embed_dim=768, num_frames=5, args=None):
#         super().__init__()
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.num_frames = num_frames
#         self.embed_dim = embed_dim
        
#         # 计算patch网格尺寸和数量
#         self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
#         self.num_patches = self.grid_size[0] * self.grid_size[1]
        
#         # 图像/视频的patch embedding
#         self.img_patch_embed = nn.Conv2d(
#             in_channels=in_channels,
#             out_channels=embed_dim,
#             kernel_size=patch_size,
#             stride=patch_size
#         )
        
#         # 事件流的patch embedding (如果使用)
#         self.event_patch_embed = nn.Conv2d(
#             in_channels=in_channels,
#             out_channels=embed_dim,
#             kernel_size=patch_size,
#             stride=patch_size
#         )
#         self.modal_type_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
#         # 空间位置嵌入 (所有帧共享)
#         self.spatial_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
#         # 时间位置嵌入 (每个时间步一个嵌入)
#         if num_frames > 1:
#             self.temp_pos_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        
#         # CLS token
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
#         # 可学习的位置类型嵌入 (用于区分CLS token)
#         self.pos_type_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
#         if args.union_token:
#             self.time_adapters = nn.ModuleDict({
#                 'video': self._build_time_adapter(num_frames),  # 视频数据6帧
#                 'event': self._build_time_adapter(num_frames)  # 视频+事件5帧
#             })
#         else:
#             self.time_adapters = None
#         # self.norm = nn.LayerNorm(embed_dim)
#         self.init_weights()
    
#     def _build_time_adapter(self, num_frames):
#     # 构建时间维度适配器，将任意帧数转换为1帧
#         return nn.Sequential(
#                 nn.Linear(num_frames * self.embed_dim, self.embed_dim),
#                 nn.GELU(),
#                 nn.Linear(self.embed_dim, self.embed_dim)
#                 )

#     def init_weights(self):
#         nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)
#         if hasattr(self, 'temp_pos_embed'):
#             nn.init.trunc_normal_(self.temp_pos_embed, std=0.02)
#         nn.init.trunc_normal_(self.cls_token, std=0.02)
#         nn.init.trunc_normal_(self.pos_type_embed, std=0.02)
        

#     def forward(self, x, event_x=None):
#         B = x.shape[0]
        
#         # ========== 处理主输入 (图像/视频) ==========
#         if len(x.shape) == 4:  # 图像数据 [B, C, H, W]
#             # Patch嵌入
#             x = self.img_patch_embed(x)  # [B, embed_dim, grid_h, grid_w]
#             x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
            
#             # 添加空间位置嵌入
#             x = x + self.spatial_pos_embed
            
#         elif len(x.shape) == 5:  # 视频数据 [B, T, C, H, W]
#             T = x.shape[1]
#             # Patch嵌入
#             x = rearrange(x, 'b t c h w -> (b t) c h w')
#             x = self.img_patch_embed(x)  # [B*T, embed_dim, grid_h, grid_w]
#             x = x.flatten(2).transpose(1, 2)  # [B*T, num_patches, embed_dim]

#             if self.time_adapters:
#                 x = rearrange(x, '(b t) n d -> b t n d', b=B, t=T)
#                 x = rearrange(x, 'b t n d -> b n (t d)')
#                 x = self.time_adapters['video'](x)
#                 x = x + self.spatial_pos_embed
#             else:
#                 x = rearrange(x, '(b t) n d -> b (t n) d', b=B, t=T)  # [B, T*num_patches, embed_dim]

#                 # 准备空间和时间位置嵌入
#                 spatial_pos = self.spatial_pos_embed.unsqueeze(1)  # [1, 1, num_patches, d]
#                 spatial_pos = spatial_pos.repeat(1, T, 1, 1)  # [1, T, num_patches, d]
#                 spatial_pos = rearrange(spatial_pos, '1 t n d -> 1 (t n) d')
                
#                 temp_pos = self.temp_pos_embed.unsqueeze(2)  # [1, T, 1, d]
#                 temp_pos = temp_pos.repeat(1, 1, self.num_patches, 1)  # [1, T, num_patches, d]
#                 temp_pos = rearrange(temp_pos, '1 t n d -> 1 (t n) d')
                
#                 # 合并位置嵌入
#                 x = x + spatial_pos + temp_pos
        
#         # ========== 添加CLS token ==========
#         cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, d]
#         cls_tokens = cls_tokens + self.pos_type_embed  # 为CLS token添加特殊位置编码
#         x = torch.cat((cls_tokens, x), dim=1)  # [B, 1+T*num_patches, d]
        
#         # ========== 处理事件流数据 (如果存在) ==========
#         if event_x is not None:
#             T = event_x.shape[1]
#             # Patch嵌入
#             event_x = rearrange(event_x, 'b t c h w -> (b t) c h w')
#             event_x = self.event_patch_embed(event_x)
#             event_x = event_x.flatten(2).transpose(1, 2)
            
#             # 添加位置嵌入 (与主输入相同的方式)
#             if T > 1:  # 视频事件流
#                 if self.time_adapters:
#                     event_x = rearrange(event_x, '(b t) n d -> b t n d', b=B, t=T)
#                     event_x = rearrange(event_x, 'b t n d -> b n (t d)')
#                     event_x = self.time_adapters['event'](event_x)
#                     event_x = event_x + self.spatial_pos_embed
#                 else:
#                     event_x = rearrange(event_x, '(b t) n d -> b (t n) d', b=B, t=T)

#                     spatial_pos = self.spatial_pos_embed.unsqueeze(1)
#                     spatial_pos = spatial_pos.repeat(1, T, 1, 1)
#                     spatial_pos = rearrange(spatial_pos, '1 t n d -> 1 (t n) d')
                    
#                     temp_pos = self.temp_pos_embed.unsqueeze(2)
#                     temp_pos = temp_pos.repeat(1, 1, self.num_patches, 1)
#                     temp_pos = rearrange(temp_pos, '1 t n d -> 1 (t n) d')
                    
#                     event_x = event_x + spatial_pos + temp_pos
#             else:  # 图像事件流
#                 event_x = event_x + self.spatial_pos_embed
            
#             # 添加模态类型嵌入
#             event_x = event_x + self.modal_type_embed
            
#             # 为事件流添加CLS token (与主输入共享)
#             event_cls = self.cls_token.expand(B, -1, -1) + self.pos_type_embed
#             event_x = torch.cat((event_cls, event_x), dim=1)
            

#             x = torch.cat([x, event_x], dim=1)

#         # x = self.norm(x)
#         return x
    
#     def load_param(self, pretrain_path):
#         param_dict = torch.load(pretrain_path, map_location='cpu')
#         for k, v in param_dict.items():
#             if k == 'patch_embed.proj.bias':
#                 # print(k, v.shape)
#                 self.state_dict()['img_patch_embed.bias'].copy_(v)
#                 self.state_dict()['event_patch_embed.bias'].copy_(v)
#             elif k == 'patch_embed.proj.weight':
#                 # print(k, v.shape)
#                 self.state_dict()['img_patch_embed.weight'].copy_(v)
#                 self.state_dict()['event_patch_embed.weight'].copy_(v)
#             elif k == 'cls_token':
#                 # print(k, v.shape)
#                 self.state_dict()['cls_token'].copy_(v)
#             elif k == 'pos_embed':
#                 # print(k, v.shape)
#                 v = resize_pos_embed(v, self.spatial_pos_embed.shape[1] + 1, self.grid_size[0], self.grid_size[1])
#                 self.state_dict()['pos_type_embed'].copy_(v[:, :1, :])
#                 self.state_dict()['spatial_pos_embed'].copy_(v[:, 1:, :])
    
# def resize_pos_embed(posemb, ntok_new, hight, width):
#     # Rescale the grid of position embeddings when loading from state_dict. Adapted from
#     # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224

#     posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]
#     ntok_new -= 1

#     gs_old = int(math.sqrt(len(posemb_grid)))
#     posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
#     posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear', align_corners=False)
#     posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
#     posemb = torch.cat([posemb_token, posemb_grid], dim=1)
#     return posemb



import math
import torch.nn.functional as F
import torch.nn as nn
import torch
from einops import rearrange


class MultiModalPatchEmbedding(nn.Module):
    def __init__(self, img_size=(256, 128), patch_size=(16, 16), in_channels=3, 
                 embed_dim=768, num_frames=5, args=None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.target_frames = 2
        
        # 计算patch网格尺寸和数量
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        # 图像/视频的patch embedding
        self.img_patch_embed = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # 事件流的patch embedding (如果使用)
        self.event_patch_embed = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.modal_type_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 空间位置嵌入 (所有帧共享)
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # 时间位置嵌入 (每个时间步一个嵌入)
        if num_frames > 1:
            self.temp_pos_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 可学习的位置类型嵌入 (用于区分CLS token)
        self.pos_type_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        if args.union_token:
            self.time_adapters = nn.ModuleDict({
                'video1': self._build_time_adapter(num_frames, self.target_frames),  # 视频数据5帧
                'video2': self._build_time_adapter(num_frames),  #事件的视频
                'event': self._build_time_adapter(num_frames)  # 视频+事件5帧
            })
        else:
            self.time_adapters = None
        # self.norm = nn.LayerNorm(embed_dim)
        self.init_weights()
    
    def _build_time_adapter(self, num_frames, target_frames=1):
    # 构建时间维度适配器，将任意帧数转换为1帧
        return nn.Sequential(
                nn.Linear(num_frames * self.embed_dim, self.embed_dim),
                nn.GELU(),
                nn.Linear(self.embed_dim, target_frames * self.embed_dim)
                )

    def init_weights(self):
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)
        if hasattr(self, 'temp_pos_embed'):
            nn.init.trunc_normal_(self.temp_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_type_embed, std=0.02)
        

    def forward(self, x, event_x=None):
        B = x.shape[0]
        
        # ========== 处理主输入 (图像/视频) ==========
        if len(x.shape) == 4:  # 图像数据 [B, C, H, W]
            # Patch嵌入
            x = self.img_patch_embed(x)  # [B, embed_dim, grid_h, grid_w]
            x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
            
            # 添加空间位置嵌入
            x = x + self.spatial_pos_embed
            
        elif len(x.shape) == 5:  # 视频数据 [B, T, C, H, W]
            T = x.shape[1]
            # Patch嵌入
            x = rearrange(x, 'b t c h w -> (b t) c h w')
            x = self.img_patch_embed(x)  # [B*T, embed_dim, grid_h, grid_w]
            x = x.flatten(2).transpose(1, 2)  # [B*T, num_patches, embed_dim]
            num_patches = x.shape[1]

            x = rearrange(x, '(b t) n d -> b (t n) d', b=B, t=T)  # [B, T*num_patches, embed_dim]

            # 准备空间和时间位置嵌入
            spatial_pos = self.spatial_pos_embed.unsqueeze(1)  # [1, 1, num_patches, d]
            spatial_pos = spatial_pos.repeat(1, T, 1, 1)  # [1, T, num_patches, d]
            spatial_pos = rearrange(spatial_pos, '1 t n d -> 1 (t n) d')
            
            temp_pos = self.temp_pos_embed.unsqueeze(2)  # [1, T, 1, d]
            temp_pos = temp_pos.repeat(1, 1, self.num_patches, 1)  # [1, T, num_patches, d]
            temp_pos = rearrange(temp_pos, '1 t n d -> 1 (t n) d')
            
            # 合并位置嵌入
            x = x + spatial_pos + temp_pos

            if self.time_adapters:
                x = rearrange(x, 'b (t n) d -> b n (t d)', t=T, n=num_patches)
                if event_x is not None:
                    x = self.time_adapters['video2'](x)
                else:
                    x = self.time_adapters['video1'](x)
                    x = x.reshape(B, num_patches, self.target_frames, self.embed_dim)
                    x = x.reshape(B, self.target_frames * num_patches, self.embed_dim)

        
        # ========== 添加CLS token ==========
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, d]
        cls_tokens = cls_tokens + self.pos_type_embed  # 为CLS token添加特殊位置编码
        x = torch.cat((cls_tokens, x), dim=1)  # [B, 1+T*num_patches, d]
        
        # ========== 处理事件流数据 (如果存在) ==========
        if event_x is not None:
            T = event_x.shape[1]
            # Patch嵌入
            event_x = rearrange(event_x, 'b t c h w -> (b t) c h w')
            event_x = self.event_patch_embed(event_x)
            event_x = event_x.flatten(2).transpose(1, 2)
            
            # 添加位置嵌入 (与主输入相同的方式)
            if T > 1:  # 视频事件流
                if self.time_adapters:
                    event_x = rearrange(event_x, '(b t) n d -> b t n d', b=B, t=T)
                    event_x = rearrange(event_x, 'b t n d -> b n (t d)')
                    event_x = self.time_adapters['event'](event_x)
                    event_x = event_x + self.spatial_pos_embed
                else:
                    event_x = rearrange(event_x, '(b t) n d -> b (t n) d', b=B, t=T)

                    spatial_pos = self.spatial_pos_embed.unsqueeze(1)
                    spatial_pos = spatial_pos.repeat(1, T, 1, 1)
                    spatial_pos = rearrange(spatial_pos, '1 t n d -> 1 (t n) d')
                    
                    temp_pos = self.temp_pos_embed.unsqueeze(2)
                    temp_pos = temp_pos.repeat(1, 1, self.num_patches, 1)
                    temp_pos = rearrange(temp_pos, '1 t n d -> 1 (t n) d')
                    
                    event_x = event_x + spatial_pos + temp_pos
            else:  # 图像事件流
                event_x = event_x + self.spatial_pos_embed
            
            # 添加模态类型嵌入
            event_x = event_x + self.modal_type_embed
            
            # 为事件流添加CLS token (与主输入共享)
            event_cls = self.cls_token.expand(B, -1, -1) + self.pos_type_embed
            event_x = torch.cat((event_cls, event_x), dim=1)
            

            x = torch.cat([x, event_x], dim=1)

        # x = self.norm(x)
        return x
    
    def load_param(self, pretrain_path):
        param_dict = torch.load(pretrain_path, map_location='cpu',weights_only=False)
        # print("Keys in param_dict:", list(param_dict.keys()))  # 打印顶级键
        if 'state_dicts' in param_dict:
            param_dict = param_dict['state_dicts']
            # print("Keys in param_dict:", list(param_dict.keys()))  # 打印顶级键
        for k, v in param_dict.items():
            # 'patch_embed.modal_type_embed', 
            # 'patch_embed.spatial_pos_embed', 
            # 'patch_embed.temp_pos_embed', 
            # 'patch_embed.cls_token', 
            # 'patch_embed.pos_type_embed', 
            # 'patch_embed.img_patch_embed.weight', 
            # 'patch_embed.img_patch_embed.bias', 
            # 'patch_embed.event_patch_embed.weight', 
            # 'patch_embed.event_patch_embed.bias',
            if k.startswith('vit.'):
                # print(k)
                continue
            if k.startswith('patch_embed.'):
                k = k[12:]
                # self.state_dict()[k].copy_(v)
                # print(k,'lord succsess')
                if k == 'img_patch_embed.bias':
                    self.state_dict()['img_patch_embed.bias'].copy_(v)
                    # print(k,"suc")
                elif k == 'event_patch_embed.bias':
                    self.state_dict()['event_patch_embed.bias'].copy_(v)
                    # print(k,"suc")
                elif k == 'img_patch_embed.weight':
                    self.state_dict()['img_patch_embed.weight'].copy_(v)
                    # print(k,"suc")
                elif k == 'event_patch_embed.weight':
                    self.state_dict()['event_patch_embed.weight'].copy_(v)
                    # print(k,"suc")
                elif k == 'cls_token':
                    self.state_dict()['cls_token'].copy_(v)
                    # print(k,"suc")
                elif k == 'pos_type_embed':
                    self.state_dict()['pos_type_embed'].copy_(v)
                    # print(k,"suc")
                elif k == 'spatial_pos_embed':
                    self.state_dict()['spatial_pos_embed'].copy_(v)
                    # print(k,"suc")
            # if k == 'patch_embed.proj.bias':
            #     # print(k, v.shape)
            #     self.state_dict()['img_patch_embed.bias'].copy_(v)
            #     self.state_dict()['event_patch_embed.bias'].copy_(v)
            # elif k == 'patch_embed.proj.weight':
            #     # print(k, v.shape)
            #     self.state_dict()['img_patch_embed.weight'].copy_(v)
            #     self.state_dict()['event_patch_embed.weight'].copy_(v)
            # elif k == 'cls_token':
            #     # print(k, v.shape)
            #     self.state_dict()['cls_token'].copy_(v)
            # elif k == 'pos_embed':
            #     # print(k, v.shape)
            #     # v = resize_pos_embed(v, self.spatial_pos_embed.shape[1] + 1, self.grid_size[0], self.grid_size[1])
            #     self.state_dict()['pos_type_embed'].copy_(v[:, :1, :])
            #     self.state_dict()['spatial_pos_embed'].copy_(v[:, 1:, :])
    
def resize_pos_embed(posemb, ntok_new, hight, width):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224

    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]
    ntok_new -= 1

    gs_old = int(math.sqrt(len(posemb_grid)))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear', align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb