import torch
import math


def get_cos_map(N, device, dtype=torch.float):

    weight_x = (torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(1, -1) + 0.5) / N
    weight_n = torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(-1, 1)
    
    # 计算余弦基
    weight = torch.cos(weight_n * weight_x * torch.pi) * math.sqrt(2 / N)
    weight[0, :] /= math.sqrt(2)
    return weight

def dct_2d(x):

    B, C, H, W = x.shape
    device = x.device
    dtype = x.dtype

    weight_h = get_cos_map(H, device, dtype)  # size: [H, H]
    weight_w = get_cos_map(W, device, dtype)  # size: [W, W]

    x_dct_h = weight_h @ x
    x_dct = x_dct_h @ weight_w.T
    
    return x_dct

def idct_2d(x_dct):

    B, C, H, W = x_dct.shape
    device = x_dct.device
    dtype = x_dct.dtype

    weight_h = get_cos_map(H, device, dtype) # size: [H, H]
    weight_w = get_cos_map(W, device, dtype) # size: [W, W]

    x_idct_w = x_dct @ weight_w

    x_idct = weight_h.T @ x_idct_w
    
    return x_idct


def get_decay_map(H, W, device, dtype=torch.float):

    weight_n = torch.pow(torch.linspace(0, H - 1, H, device=device, dtype=dtype), 2).view(-1, 1)
    weight_m = torch.pow(torch.linspace(0, W - 1, W, device=device, dtype=dtype), 2).view(1, -1)

    decay_core = (weight_n * (torch.pi / H) ** 2 + weight_m * (torch.pi / W) ** 2)
    weight_exp = torch.exp(-decay_core)
    
    return weight_exp # size: [H, W]

def apply_frequency_filter(x):
    B, C, H, W = x.shape
    x_dct = dct_2d(x)

    weight_exp = get_decay_map(H, W, device=x.device, dtype=x.dtype)
    x_dct_filtered = x_dct * weight_exp
    x_filtered_spatial = idct_2d(x_dct_filtered)
    
    return x_filtered_spatial


def apply_feature_frequency_filter(x):

    B, N, D = x.shape
    H_patches = 16  # 256 // 16
    W_patches = 12  # 192 // 16

    assert N == H_patches * W_patches, f"Expected {H_patches * W_patches} patches, got {N}"
    x_spatial = x.transpose(1, 2).reshape(B, D, H_patches, W_patches)
    x_filtered = apply_frequency_filter(x_spatial)
    x_tokens = x_filtered.reshape(B, D, N).transpose(1, 2)
    
    return x_tokens


if __name__ == '__main__':
    input_tensor = torch.randn(1, 3, 32, 32)

    print(f"input: {input_tensor.shape}")
    output_tensor = apply_frequency_filter(input_tensor)
    print(f"output: {output_tensor.shape}")



