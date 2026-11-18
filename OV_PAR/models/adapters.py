import torch
import torch.nn as nn

class Adapter(nn.Module):

    def __init__(self, input_dim, out_dim, dim=8, dropout_rate=0.1):
        super().__init__()
        self.adapter_down = nn.Linear(input_dim, dim)
        self.adapter_mid = nn.Linear(dim, dim)
        self.adapter_up = nn.Linear(dim, out_dim)
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.xavier_uniform_(self.adapter_mid.weight)
        nn.init.zeros_(self.adapter_mid.bias)
        nn.init.xavier_uniform_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.act(self.adapter_down(x))
        x = self.dropout(self.act(self.adapter_mid(x)))
        x = self.adapter_up(x)
        return x
