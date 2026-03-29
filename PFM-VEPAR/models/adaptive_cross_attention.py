"""
简化的自适应权重交叉注意力融合模块
专注于adaptive_weighted融合策略的完整实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding2D(nn.Module):
    """2D位置编码，适配ViT的patch特征"""
    
    def __init__(self, d_model, max_len=200):
        super().__init__()
        self.d_model = d_model
        
        # 创建位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]
        
    def forward(self, x):
        """
        Args:
            x: [B, N, D] 输入特征
        Returns:
            位置编码 [B, N, D]
        """
        B, N, D = x.shape
        return self.pe[:, :N, :].expand(B, -1, -1)


class CrossAttentionLayer(nn.Module):
    """单层交叉注意力模块"""
    
    def __init__(self, d_model=768, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        
        # RGB → Event 交叉注意力
        self.rgb_to_event_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # Event → RGB 交叉注意力
        self.event_to_rgb_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
        # MLP层
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding2D(d_model)
        
    def forward(self, rgb_features, event_features):
        """
        Args:
            rgb_features: [B, N, D] RGB特征
            event_features: [B, N, D] Event特征
        
        Returns:
            enhanced_rgb: [B, N, D] 增强的RGB特征
            enhanced_event: [B, N, D] 增强的Event特征
        """
        # 添加位置编码
        rgb_pos = self.pos_encoding(rgb_features)
        event_pos = self.pos_encoding(event_features)
        
        # RGB → Event 交叉注意力
        rgb_query = rgb_features + rgb_pos
        event_kv = event_features + event_pos
        
        rgb_attn_out, _ = self.rgb_to_event_attn(rgb_query, event_kv, event_features)
        rgb_enhanced = self.norm1(rgb_features + self.dropout1(rgb_attn_out))
        
        # Event → RGB 交叉注意力
        event_query = event_features + event_pos
        rgb_kv = rgb_enhanced + rgb_pos
        
        event_attn_out, _ = self.event_to_rgb_attn(event_query, rgb_kv, rgb_enhanced)
        event_enhanced = self.norm2(event_features + self.dropout2(event_attn_out))
        
        # MLP处理
        rgb_mlp_out = self.mlp(rgb_enhanced)
        rgb_final = self.norm3(rgb_enhanced + self.dropout3(rgb_mlp_out))
        
        event_mlp_out = self.mlp(event_enhanced)
        event_final = self.norm4(event_enhanced + self.dropout4(event_mlp_out))
        
        return rgb_final, event_final


class AdaptiveWeightedFusion(nn.Module):
    """自适应权重融合模块"""
    
    def __init__(self, d_model=768, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # 全局权重网络：基于整个序列的平均信息
        self.global_weight_net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # [B, 2*D, N] -> [B, 2*D, 1]
            nn.Flatten(),             # [B, 2*D, 1] -> [B, 2*D]
            nn.Linear(d_model * 2, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
            nn.Softmax(dim=-1)
        )
        
        # 局部权重网络：基于每个位置的特征
        self.local_weight_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
            nn.Softmax(dim=-1)
        )
        
        # 特征增强网络：对融合后的特征进行进一步处理
        self.feature_enhance = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        # 残差连接的权重
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
        # 全局和局部权重的平衡因子
        self.register_parameter('alpha', nn.Parameter(torch.tensor(0.6)))
        
    def forward(self, rgb_features, event_features):
        """
        Args:
            rgb_features: [B, N, D] RGB特征
            event_features: [B, N, D] Event特征
        
        Returns:
            fused_features: [B, N, D] 融合后的特征
        """
        B, N, D = rgb_features.shape
        
        # 拼接特征
        concat_features = torch.cat([rgb_features, event_features], dim=-1)  # [B, N, 2*D]
        
        # 计算全局权重
        global_input = concat_features.transpose(1, 2)  # [B, 2*D, N] for AdaptiveAvgPool1d
        global_weights = self.global_weight_net(global_input)  # [B, 2]
        global_weights = global_weights.unsqueeze(1).expand(-1, N, -1)  # [B, N, 2]
        
        # 计算局部权重
        local_weights = self.local_weight_net(concat_features)  # [B, N, 2]
        
        # 自适应组合全局和局部权重
        alpha = torch.sigmoid(self.alpha)  # 确保在[0,1]范围内
        combined_weights = alpha * global_weights + (1 - alpha) * local_weights
        
        # 提取RGB和Event权重
        rgb_weight = combined_weights[:, :, 0:1]    # [B, N, 1]
        event_weight = combined_weights[:, :, 1:2]  # [B, N, 1]
        
        # 加权融合
        weighted_features = rgb_features * rgb_weight + event_features * event_weight
        
        # 特征增强
        enhanced_features = self.feature_enhance(weighted_features)
        
        # 残差连接：保留原始RGB特征的信息
        residual_weight = torch.sigmoid(self.residual_weight)
        fused_features = enhanced_features + residual_weight * rgb_features
        
        return fused_features


class AdaptiveCrossAttentionFusion(nn.Module):
    """
    完整的自适应交叉注意力融合模块
    结合交叉注意力和自适应权重融合
    """
    
    def __init__(self, d_model=768, nhead=8, num_layers=2, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        self.num_layers = num_layers
        
        # 多层交叉注意力
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # 自适应权重融合
        self.adaptive_fusion = AdaptiveWeightedFusion(d_model, dropout)
        
        # 最终归一化
        self.final_norm = nn.LayerNorm(d_model)
        
    def forward(self, rgb_features, event_features):
        """
        Args:
            rgb_features: [B, N, D] RGB特征
            event_features: [B, N, D] Event特征
        
        Returns:
            fused_features: [B, N, D] 融合后的特征
        """
        # 多层交叉注意力处理
        for layer in self.cross_attention_layers:
            rgb_features, event_features = layer(rgb_features, event_features)
        
        # 自适应权重融合
        fused_features = self.adaptive_fusion(rgb_features, event_features)
        
        # 最终归一化
        fused_features = self.final_norm(fused_features)
        
        return fused_features



if __name__ == "__main__":
