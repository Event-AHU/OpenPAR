import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalAttrGating(nn.Module):

    def __init__(self, attr_dim, num_layers, hidden_dim=128):
        super().__init__()
        self.num_layers = num_layers
        self.gating_net = nn.Sequential(nn.Linear(attr_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_layers))

    def forward(self, attr_feats, visual_feats):
        B, N_attr, D_feat = visual_feats[0].shape
        L = self.num_layers
        if isinstance(visual_feats, list):
            visual_tensor = torch.stack(visual_feats, dim=2)
        else:
            visual_tensor = visual_feats
        gate_logits = self.gating_net(attr_feats)
        gate_weights = F.softmax(gate_logits, dim=-1)
        gate_weights = gate_weights.unsqueeze(0).unsqueeze(-1)
        fused_feats = torch.sum(gate_weights * visual_tensor, dim=2)
        return fused_feats
