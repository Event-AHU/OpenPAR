import torch
import torch.nn as nn
from timm.models.vision_transformer import Mlp, Block

class SidePatchEmbed(nn.Module):
    def __init__(self, img_size=224, in_chans=3, embed_dim=240):
        super(SidePatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=16, stride=16)
        self.norm = nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x

def create_vit_block(dim, num_heads, mlp_dim, qkv_bias=False, drop=0.0, attn_drop=0.0, drop_path=0.0):
    block = nn.Sequential(
        nn.LayerNorm(dim),
        Block(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_dim / dim,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            drop_path=drop_path
        ),
        nn.Identity(),  
        nn.Identity()  
    )
    return block

class SideViT(nn.Module):
    def __init__(self, img_size, dim, num_heads, mlp_dim, depth, qkv_bias=False,
                 drop=0.0, attn_drop=0.0, drop_path=0.0,embed_layer=None):
        super(SideViT, self).__init__()
        self.num_features = dim
        if embed_layer :
            self.patch_embed = embed_layer(img_size=img_size, patch_size=16, in_chans=3, embed_dim=dim)
        else:
            self.patch_embed = SidePatchEmbed(img_size=img_size, in_chans=3, embed_dim=dim)
        self.blocks = nn.ModuleList([
            create_vit_block(dim, num_heads, mlp_dim, qkv_bias, drop, attn_drop, drop_path)
            for _ in range(depth)  
        ])

        self.head = nn.Sequential(
            nn.Identity(),  
            nn.Dropout(p=0.0, inplace=False),  
            nn.Identity()  
        )

    def forward(self, x):
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.head(x)

        return x