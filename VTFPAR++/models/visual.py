from typing import List
import torch
from torch import nn
from torch.nn import functional as F
from open_clip.transformer import VisionTransformer
from models.attn_helper import cross_attn_layer, downsample2d, resize_pos_embed2d

class ClipOutput(dict):
    def __init__(self, spacial_shape, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spacial_shape = spacial_shape

    def save(self, idx: int, clip_feat: torch.Tensor):
        l, n, c = clip_feat.shape
        self[idx] = (
            clip_feat[1:].permute(1, 2, 0).reshape(n, c, *self.spacial_shape)
        )  # n, c, h, w
        self[f"{idx}_cls_token"] = clip_feat[0:1]  # 1, n, c


class FeatureExtractor(nn.Module):
    
    def __init__(
        self,
        visual_encoder: VisionTransformer,
        last_layer_idx: int = -1,
        frozen_exclude=[],
    ):
        super().__init__()
        self.image_size = visual_encoder.input_resolution
        self.patch_size = visual_encoder.patch_size
        self.grid_size = self.image_size // self.patch_size
        self.num_features = visual_encoder.ln_pre.normalized_shape[0]#768
        self.conv1 = visual_encoder.conv1
        self.class_embedding = visual_encoder.class_embedding
        self.positional_embedding = visual_encoder.positional_embedding
        self.ln_pre = visual_encoder.ln_pre
        if last_layer_idx == -1:
            self.resblocks = visual_encoder.transformer.resblocks
            self.last_output_idx = len(self.resblocks) + 1
        else:
            self.resblocks = visual_encoder.transformer.resblocks[:last_layer_idx]
            self.last_output_idx = last_layer_idx + 1
        self.frozen_exclude = frozen_exclude
        self._freeze(self.frozen_exclude)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  
        _, _, h, w = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1)  
        x = x.permute(0, 2, 1)  
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        ) 
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND

        outputs = ClipOutput(spacial_shape=(h, w))
        outputs.save(0, x)
        
        for i, resblock in enumerate(self.resblocks, start=1):
            x = resblock(x)
            outputs.save(i, x)
        return outputs,x

    def _freeze(self, frozen_exclude):
        if "all" in frozen_exclude:
            return
        for name, param in self.named_parameters():
            if not any([exclude in name for exclude in frozen_exclude]):
                param.requires_grad = False

    @property
    def size_divisibility(self):
        return self.patch_size[0]