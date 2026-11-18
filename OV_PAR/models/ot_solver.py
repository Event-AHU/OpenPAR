import torch

def Sinkhorn(a, b, M, reg, max_iter=100, thresh=0.001):
    K = torch.exp(-M / reg)
    r = torch.ones_like(a)
    c = torch.ones_like(b)
    thresh = 0.001
    for i in range(max_iter):
        r0 = r
        r = a / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
        c = b / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
        err = (r - r0).abs().mean(dim=1)
        if torch.all(err < thresh):
            break
    T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K
    return T
