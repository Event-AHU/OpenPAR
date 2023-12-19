import os.path as osp
import torch
import torch.nn as nn
from torch.nn import functional as F
from CLIP.clip import clip
from CLIP.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer.float()#
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):#tokenized_prompts
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

class PromptLearner(nn.Module):#重点
    def __init__(self, template_prompt_length ,classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)# 100
        n_ctx = template_prompt_length # number of context tokens
        #ctx_init = "A pedestrian whose"
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        """
        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")#将_替换为空格
            n_ctx = len(ctx_init.split(" "))#init长度
            prompt = clip.tokenize(ctx_init) #token化 这里是有初始值的不是随机的
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt.to("cuda")).type(dtype).to("cuda")#embedding layer 
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]#context token 向量
            prompt_prefix = ctx_init#将init作为prefix
        else:
        """
        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02) #从给定均值和标准差的正态分布N(mean, std)中生成值, torch.nn.init.normal(tensor, mean=0, std=1) 
        prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')#"X X X X X X X X X X X X X X X X"
        print(f"Number of context words (tokens): {n_ctx}")#16 超参数

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized 将ctx_vectors变为可学习参数
        
        classnames = [name.replace("_", " ") for name in classnames] #将classname中的_变为空格 有些词中间会有下划线
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]#token化后的name长度
        prompts = [prompt_prefix + " " + name + "." for name in classnames]#组装prompt # X X X X X X X X X X X X X X X X classname 为什么这么设置
        
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])#token化 [class_len,77]

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts.to("cuda")).to("cuda")#embedding layer 

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS    句子起始标识符  缓冲区存储前缀 
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS  句子结束标识符   缓冲区存储后缀
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor 
        self.name_lens = name_lens
        
    def forward(self):
        
        ctx = self.ctx
        if ctx.dim() == 2:#如果ctx有2个维度[,] 则升维
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)#在expand中的-1表示取当前所在维度的尺寸，也就是表示当前维度不变。
        
        prefix = self.token_prefix#这里应该是从缓冲区取出
        suffix = self.token_suffix
        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        return prompts