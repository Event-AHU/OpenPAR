from collections import OrderedDict
from turtle import Turtle
from typing import Tuple, Union
import math
from functools import reduce
from operator import mul
from config import argument_parser
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Conv2d, Dropout
parser = argument_parser()
args = parser.parse_args()
assert args.dataset in ['PA100k', 'RAPV1','RAPV2','PETA','WIDER','PETAzs','RAPzs','UPAR','YCJC',"RAPV1Expand",'MSPAR'], \
    f'dataset name {args.dataset} is not exist,The legal name is PA100k,RAPV1,RAPV2,PETA,WIDER,PETAzs,RAPzs,RAPV1Expand'
datasets_attrnum={'PA100k':26,'RAPV1':51,'PETA':35,'PETAzs':35,'UPAR':40,'RAPzs':53,'RAPV2':54,'WIDER':14,"RAPV1Expand":51,'MSPAR':57}
attr_num=datasets_attrnum[args.dataset]
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module): #ResNet
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x
        
        x = x.type(self.conv1.weight.dtype)#torch.Size([1, 3, 224, 224])
        x = stem(x)#torch.Size([1, 64, 56, 56])
        x = self.layer1(x)#torch.Size([1, 256, 56, 56])
        x = self.layer2(x)#torch.Size([1, 512, 28, 28])
        x = self.layer3(x)#torch.Size([1, 1024, 14, 14])
        x = self.layer4(x)#torch.Size([1, 2048, 7, 7])
        x = self.attnpool(x)#torch.Size([1, 1024])
        
        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
    def attention(self, x: torch.Tensor,visual_mask: torch.Tensor = None):

        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        if visual_mask is not None:
            self.attn_mask = visual_mask.to(dtype=x.dtype, device=x.device) 
        return self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask)
    
    def forward(self, x: torch.Tensor, visual_mask: torch.Tensor = None):
        
        attn_output, attn_output_weights = self.attention(self.ln_1(x),visual_mask)
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))

        return x,attn_output_weights


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None,
                 VorT:bool=False, prompt_num:int=25, part_num:list=None, row_patch_num:int=0):
        super().__init__()
        self.VorT=VorT
        self.width = width
        self.layers = layers
        self.prompt_num = prompt_num
        self.part_num = part_num
        self.div_prompt_num = int(prompt_num/(args.div_num+1))
        self.vis_len= prompt_num + 5 + row_patch_num**2
        self.prefix_len = prompt_num + 5 
        if self.VorT:
            val = math.sqrt(6. / float(3 * reduce(mul, (14,14), 1) + width))
            self.prompt_deep=nn.Parameter(torch.zeros(args.vis_depth,prompt_num,1,width))
            nn.init.uniform_(self.prompt_deep.data, -val,val)
        else:
            val = math.sqrt(6. / float(3 * reduce(mul, (14,14), 1) + width))
            self.prompt_text_deep=nn.Parameter(torch.zeros(layers,prompt_num,attr_num,width))
            nn.init.uniform_(self.prompt_text_deep.data, -val,val)

        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

        if self.VorT==True and args.use_div and args.use_vismask:
            print('bulid visual mask')
            self.visual_mask = self.build_visual_mask()
        else:
            self.visual_mask = None

    def forward(self, x: torch.Tensor):
        bs = x.shape[1]
        attnmap=[]
        if self.VorT==True:
            for layer,block in enumerate(self.resblocks):
                if args.use_div :
                    if layer < args.vis_depth:
                        for div in range(args.div_num + 1):
                            start_idx = 1 if div == 0 else self.div_prompt_num * div + div + 1
                            prompts = self.prompt_deep[layer, self.div_prompt_num * div: self.div_prompt_num * (div + 1)]
                            x = torch.cat([x[:start_idx], prompts.repeat(1, x.shape[1], 1).to(x.device).to(x.dtype), x[start_idx if layer == 0 else start_idx + self.div_prompt_num:]], dim=0)
                        assert x.size() == (self.vis_len, bs, self.width), f'The Shape of image features is {x.size()}'
                else:
                    if layer < args.vis_depth:
                        x = torch.cat([x[:1], self.prompt_deep[layer].repeat(1, x.shape[1], 1).to(x.device).to(x.dtype), x[1 if layer == 0 else self.prompt_num + 1:]], dim=0)
                x,attn_output_weights = block(x,self.visual_mask)   
            all_class = None
            if args.use_div:
                all_class = torch.stack([x[1 if idx == 0 else self.div_prompt_num * idx + idx + 1 - 1] for idx in range(args.div_num + 1)]).permute(1, 0, 2)
            return x, all_class,attn_output_weights
        else:
            for layer,block in enumerate(self.resblocks):
                if args.use_textprompt:
                    if layer == 0 :
                        # print("x shape:", x.shape)
                        # print("prompt shape:", self.prompt_text_deep[0].shape)
                        x=torch.cat([x,self.prompt_text_deep[0].to(x.device).to(x.dtype)],dim=0) 
                    x=torch.cat([x[:77,:,:],self.prompt_text_deep[layer].to(x.dtype).to(x.device)],dim=0)
                x,_ = block(x)
            return x
    def build_visual_mask(self):
        visual_mask = []
        length = self.vis_len
        insert_part_num = list(self.part_num)
        insert_part_num.insert(0,0)
        every_prefix_len = 1+self.div_prompt_num
        mask_part_start = [self.prefix_len,self.prefix_len + 48,self.prefix_len + 112,self.prefix_len + 176]
        masked_row = [idx for idx in range(1+self.div_prompt_num,self.prefix_len)]                    
        for row in range(length):
            attn_mask = torch.zeros(length, dtype=torch.float)
            if row in masked_row :
                part_idx = int(row/every_prefix_len)
                mask_class_per = [per_idx for per_idx in range(part_idx*every_prefix_len)]
                mask_class_post = [post_idx for post_idx in range((part_idx+1)*every_prefix_len,self.prefix_len)]
                all_patch = [per_idx for per_idx in range(self.prefix_len,length)] 
                no_mask = [per_id for per_id in range(mask_part_start[part_idx-1],mask_part_start[part_idx-1]+self.part_num[part_idx-1])]
                mask_patch = [item for item in all_patch if item not in no_mask]   
                mask_indices = mask_class_per + mask_class_post + mask_patch
                attn_mask[mask_indices] = float("-inf")
                visual_mask.append(attn_mask)
            else:
                visual_mask.append(attn_mask)
        visual_mask = torch.stack(visual_mask)
        return visual_mask

class VisionTransformer(nn.Module):#ViTB-32 32,768,12,12,512
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, prompt_num:int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        row_patch_num = (input_resolution // patch_size)
        
        base_part_row_num = int(row_patch_num // args.div_num) + int(args.overlap_row // 2)#假设div_num为4 则part_row_num = [5,6,6,5] 0-4 3-8 7-12 11-15
        self.part_row_num = [base_part_row_num+int(args.overlap_row//2) if row!=0 and row!=args.div_num-1 else base_part_row_num for row in range(args.div_num)]
        self.part_num = [part_row * row_patch_num for part_row in self.part_row_num] # 80个token
        if args.use_div :
            self.part_class_embedding = nn.Parameter(scale * torch.randn((args.div_num,width)))
        self.transformer = Transformer(width, layers, heads,VorT=True,prompt_num=prompt_num,part_num=self.part_num,row_patch_num=row_patch_num)#24层
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):

        x = self.conv1(x)  # shape = [*, width, grid, grid] 
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2] 
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width] 
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        x = x + self.positional_embedding.to(x.dtype)

        if args.use_div :
            x = torch.cat([x[:,:1],self.part_class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
                                ,x[:,1:]],dim=1)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x,all_class,attnmap = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD   
        if self.proj is not None:
            x = x @ self.proj
            if all_class is not None:
                all_class = all_class @ self.proj
        return x,all_class,attnmap
    
class SoftmaxWithTemperature(nn.Module):
    def __init__(self, initial_temperature=1.0):
        super(SoftmaxWithTemperature, self).__init__()
        self.temperature = nn.Parameter(torch.tensor(initial_temperature))

    def forward(self, logits):
        scaled_logits = logits / self.temperature
        softmax_output = torch.softmax(scaled_logits, dim=-1)
        return softmax_output


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 ):

        super().__init__()
        parser = argument_parser()
        self.args = parser.parse_args()
        self.vis_prompt_len=self.args.vis_prompt
        self.text_prompt_len=self.args.text_prompt
        self.context_length = context_length
        if args.use_GL :
            self.softmax_model = SoftmaxWithTemperature()
            self.agg_bn = nn.BatchNorm1d(attr_num)
        
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                prompt_num=self.vis_prompt_len
            )
        
        self.text_prompt_len=self.text_prompt_len if args.use_textprompt else 0
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            prompt_num=self.text_prompt_len
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))#torch.empty()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length+self.text_prompt_len, self.context_length+self.text_prompt_len)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model] [text_len,77,512]
        x = x + self.positional_embedding.type(self.dtype)
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)#(77,5,512)
        
        x = x.permute(1, 0, 2)  # LND -> NLD#torch.Size([5, 77, 512])
        x = self.ln_final(x).type(self.dtype)#torch.Size([5, 77, 512])
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection#torch.Size([5, 512])
        return x
    
    def forward_aggregate(self, image, text):
        all_class = (image / image.norm(dim=-1, keepdim=True)).float()
        text_features = (text / text.norm(dim=-1, keepdim=True)).float()
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * all_class @ text_features.t()
        
        similarity = self.softmax_model(logits_per_image)
        global_similarity = similarity[:,0]
        local_similarity = similarity[:,1:]                
        for logits_local in local_similarity:
            max_values, _ = torch.max(logits_local, dim=0)#max_values.detach().numpy()
            min_values, _ = torch.min(logits_local, dim=0)
            gama=max_values > args.ag_threshold
            similarity_aggregate = gama.float() * max_values + (1 - gama.float()) * min_values    

        final_similarity = (similarity_aggregate + global_similarity) / 2   

        return self.agg_bn(final_similarity),logits_per_image
        
    
    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        #return image_features,text_features
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict
    # breakpoint()
    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]#768
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size#224
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
    )
    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]
    convert_weights(model)
    model.load_state_dict(state_dict,strict=False)
    return model.eval()