import torch.nn as nn
import torch
import mmcv
from torch.utils.data import DataLoader, Dataset
from mmcv.runner import load_checkpoint
from mmcls.models import build_backbone, build_classifier
from mmcls.utils import wrap_non_distributed_model
import mmcls_custom
from config import argument_parser
parser = argument_parser()
args = parser.parse_args()

class VRWKV(nn.Module):
    def __init__(self, config_path, checkpoint_path, device='cuda', gpu_ids=[0]):
        """
        初始化 VRWKV 模型。

        参数:
        config_path (str): 配置文件路径。
        checkpoint_path (str): 检查点文件路径。
        device (str): 设备类型，默认为 'cuda'。
        gpu_ids (list): GPU ID 列表，默认为 [0]。
        """
        super(VRWKV, self).__init__()

        # 加载配置文件
        cfg = mmcv.Config.fromfile(config_path)
        cfg.device = device
        cfg.gpu_ids = gpu_ids

        # 构建模型并加载检查点
        self.vrwkv_model = build_classifier(cfg.model)
        self.fully_connected= nn.Linear(768, 768).cuda()
        # self.fc_egb = nn.Linear(768, 768).cuda() 
        # self.fc_event = nn.Linear(768, 768).cuda() 
        load_checkpoint(self.vrwkv_model, checkpoint_path, map_location='cuda')
        self.vrwkv_model = wrap_non_distributed_model(self.vrwkv_model, device=cfg.device, device_ids=cfg.gpu_ids)
        self.device = device

    def forward(self, x):
        """
        自定义的 forward 方法，用于返回特征并进行维度变换。

        参数:
        x (torch.Tensor): 输入张量。

        返回:
        torch.Tensor: 处理后的特征张量。
        """
        x = x.to(self.device)
    
        batch_size, num_frames, channels, height, width = x.size()
        x=x.view(-1, channels, height, width)#(DF, C, H, W)

        features = self.vrwkv_model.module.backbone(x)  # 访问实际的模型对象
        
        
        if isinstance(features, tuple):
            features = features[0]  # 提取主要的特征张量
        # print(features.shape)
        
        DF, C, H, W = features.shape
        features = features.view(DF, C, H * W).permute(0, 2, 1).view(batch_size,num_frames,  H * W, C)
        #features = torch.mean(features,dim=1)

        features = self.fully_connected(features)
        #r, k, v, g, w=self.vrwkv_model.module.backbone.layers[0].att.forward(features,(16,8))

        return features

# def main():
#     model = VRWKV('models/vrwkv_configs/vrwkv/vrwkv_base_16xb64_in1k.py', 'checkpoints/vrwkv_b_in1k_224.pth')
#     print(model)

# if __name__ == '__main__':
#     main()
