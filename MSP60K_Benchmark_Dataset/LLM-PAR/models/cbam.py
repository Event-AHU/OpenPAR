import torch
import torch.nn as nn
import torch.nn.init as init

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

        # 初始化卷积和线性层参数
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

        # 初始化卷积层参数
        nn.init.kaiming_normal_(self.conv1.weight)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_channels, output_dim=1, reduction=16, kernel_size=7, hidden_dims=None):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction=reduction)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)
        self.relu = nn.ReLU()

        self.channel_attention = ChannelAttention(in_channels, reduction=reduction)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)
        self.fc = nn.Linear(in_channels, output_dim)

        # 初始化线性层参数
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B, L, D] -> [B, D, L]
        x_out = x * self.channel_attention(x)
        x_out = x_out.permute(0, 2, 1)  # [B, D, L] -> [B, L, D]
        x_out = x_out * self.spatial_attention(x_out.permute(0, 2, 1)).permute(0, 2, 1)  # [B, L, D]
        x_out = torch.mean(x_out, dim=1)  # 对 L 维度进行全局平均池化，得到 [B, D]
        x_out = self.fc(self.relu(x_out))
        return x_out
