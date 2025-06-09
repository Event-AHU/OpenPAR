import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, knn_graph,knn

class GCNLayer(nn.Module):
    """单层GCN封装"""
    def __init__(self, in_dim, out_dim):
        super(GCNLayer, self).__init__()
        self.gcn = GCNConv(in_dim, out_dim)

    def forward(self, x, edge_index):
        return self.gcn(x, edge_index).relu()

class TokenGCN(nn.Module):
    """GCN 网络，支持全连接和 KNN"""
    def __init__(self, in_dim=768, hidden_dim=512, out_dim=768, num_layers=3, k=5, use_knn=False):
        super(TokenGCN, self).__init__()
        self.use_knn = use_knn
        self.k = k
        self.num_layers = num_layers

        # 定义 GCN 层
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNLayer(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.gcn_layers.append(GCNLayer(hidden_dim, hidden_dim))
        self.gcn_layers.append(GCNLayer(hidden_dim, out_dim))

    def forward(self, x):
        """
        x: (B, N, 768) 输入 token 特征
        返回: (B, 128, 768)
        """
        B, N, C = x.shape  # (B, N, 768)
        out_list = []

        for i in range(B):
            xi = x[i]  # (N, 768) 单个样本

            # 1. 构建邻接矩阵
            if self.use_knn:
                edge_index = knn( xi,xi[:128],k=self.k)  # KNN 邻接矩阵+
            else:
                edge_index = self.get_fully_connected_edges(N, x.device)  # 全连接图
            
            # 2. 依次通过多层 GCN
            for j, layer in enumerate(self.gcn_layers):
                xi = layer(xi, edge_index)
                
            
            out_list.append(xi[:128])  # 取前 128 个 token

        return torch.stack(out_list, dim=0)  # (B, 128, 768)

    def get_fully_connected_edges(self, num_nodes, device):
        """构造完全图的邻接边"""
        row = torch.arange(num_nodes, dtype=torch.long, device=device).repeat(num_nodes)
        col = torch.arange(num_nodes, dtype=torch.long, device=device).repeat_interleave(num_nodes)
        mask = row != col
        #edge_index = torch.stack([row, col], dim=0)
        edge_index = torch.stack([row[mask], col[mask]], dim=0)
        return edge_index