import torch
import torch.nn as nn
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

class PromptLearner(nn.Module):

    def __init__(self, args, classnames, clip_model, dataset, mode='attribute', device='cuda'):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = args.context_len
        dtype = torch.float32
        ctx_dim = clip_model.ln_final.weight.shape[0]
        prompt_prefix = ' '.join(['X'] * n_ctx)
        print(f'Initial context: "{prompt_prefix}"')
        print(f'Number of context words (tokens): {n_ctx}')
        print(f'Random init prompts!!!')
        prompts = clip.tokenize(prompt_prefix)
        with torch.no_grad():
            embeddings = clip_model.token_embedding(prompts.to(device)).type(dtype)
        ctx_vectors = embeddings[0, 1:1 + n_ctx, :]
        self.ctx = nn.Parameter(ctx_vectors.to(device))
        classnames = [name.replace('_', ' ') for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + ' ' + name + '.' for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts.to(device)).type(dtype)
        self.register_buffer('token_prefix', embedding[:, :1, :])
        self.register_buffer('token_suffix', embedding[:, 1 + n_ctx:, :])
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts.to(device)
        self.name_lens = name_lens
        self.class_token_position = 'end'

    def forward(self, attr_idx=None):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts

class PromptLearner2(nn.Module):

    def __init__(self, args, clip_model, device='cuda'):
        super().__init__()
        n_ctx = args.context_len if not args.no_sen_prompt else 0
        dtype = torch.float32
        self.prompt_prefix = ' '.join(['X'] * n_ctx)
        ctx_dim = clip_model.ln_final.weight.shape[0]
        print(f'Initial context: "{self.prompt_prefix}"')
        print(f'Number of context words (tokens): {n_ctx}')
        print(f'Random init prompts!!!')
        prompts = clip.tokenize(self.prompt_prefix)
        with torch.no_grad():
            embeddings = clip_model.token_embedding(prompts.to(device)).type(dtype)
        ctx_vectors = embeddings[0, 1:1 + n_ctx, :]
        self.ctx = nn.Parameter(ctx_vectors.to(device))
        self.n_ctx = n_ctx
        self.device = device

    def forward(self, classnames, clip_model):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(len(classnames), -1, -1)
        classnames = [name.replace('_', ' ') for name in classnames]
        prompts = [self.prompt_prefix + ' ' + name + '.' for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p, truncate=True) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts.to(self.device)).to(self.device)
        prefix = embedding[:, :1, :]
        suffix = embedding[:, 1 + self.n_ctx:, :]
        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return (prompts, tokenized_prompts)

class TextEncoder(nn.Module):

    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x
