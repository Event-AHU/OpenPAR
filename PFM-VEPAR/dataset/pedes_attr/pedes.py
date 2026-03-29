import glob
import os
import pickle

import numpy as np
import torch.utils.data as data
from PIL import Image
import random
from tools.function import get_pkl_rootpath
import torch
from easydict import EasyDict
class PedesAttr(data.Dataset):

    def __init__(self, cfg, split, transform=None, target_transform=None, idx=None):
        # 标志位先初始化，避免未定义
        self.use_duke = False

        assert cfg.DATASET.NAME in ['PETA','EventPAR','Mars','DUKE', 'PA100k', 'RAP', 'RAP2', 'MSPAR'], \
            f'dataset name {cfg.DATASET.NAME} is not exist'
        # # ============ Mars 分支 ============
        # if cfg.DATASET.NAME == 'Mars':
        #     self.cfg = cfg
        #     self.split = split
        #     self.transform = transform
        #     self.target_transform = target_transform
        #     # 允许从 YAML 指定 PKL，否则按 ROOT 推断
        #     pkl_path = getattr(cfg.DATASET, 'PKL', None) or os.path.join(cfg.DATASET.ROOT, 'mars.pkl')
        #     with open(pkl_path, 'rb') as f:
        #         dataset_info = pickle.load(f)

        #     # 兼容 dict 或对象（attr 访问）
        #     getv = (lambda k: dataset_info[k]) if isinstance(dataset_info, dict) else (lambda k: getattr(dataset_info, k))

        #     self.attributes = list(getv('attr_name'))
        #     self.attr_num = len(self.attributes)
        #     self.eval_attr_num = self.attr_num

        #     # 根路径优先用 pkl 的 root，若不存在则用 YAML 的 ROOT
        #     self.root_path = getv('root') if ('root' in dataset_info if isinstance(dataset_info, dict) else hasattr(dataset_info, 'root')) else cfg.DATASET.ROOT

        #     if split == 'trainval':
        #         self.img_id = list(getv('trainval_name'))  # 形如 ['0001C1T0002', ...]
        #         self.label = np.array(getv('trainval_gt_list'))
        #     elif split in ['test', 'val']:
        #         # YAML 用 VAL_SPLIT='test'，这里兼容
        #         self.img_id = list(getv('test_name'))
        #         self.label = np.array(getv('test_gt_list'))
        #     else:
        #         raise ValueError(f'Unknown split for Mars: {split}')

        #     # 防止 label 与 id 数量不一致
        #     assert len(self.img_id) == len(self.label), f'len(img_id)={len(self.img_id)} != len(label)={len(self.label)}'

        #     self.label_word = np.array(self.attributes)
        #     self.img_num = len(self.img_id)
        #     return
        # # 如果是DUKE数据集，使用新的加载方式
        # if cfg.DATASET.NAME == 'DUKE':
        #     from dataset.pedes_attr.duke_dataset import DukeDataset
        #     # 设置DUKE数据集的路径
        #     data_root = '/root/autodl-tmp/datasets/DUKE'
        #     csv_path = '/root/autodl-tmp/datasets/DUKE/duke_pre/new_encoded.csv'
        #     trainval_txt = '/root/autodl-tmp/datasets/DUKE/duke_pre/trainval_name.txt'
        #     test_txt = '/root/autodl-tmp/datasets/DUKE/duke_pre/test_name.txt'
        
        #     self.duke_dataset = DukeDataset(
        #         cfg=cfg,
        #         split=split,
        #         transform=transform,
        #         target_transform=target_transform,
        #         data_root=data_root,
        #         csv_path=csv_path,
        #         trainval_txt=trainval_txt,
        #         test_txt=test_txt
        #     )
        
        #     # 复制必要属性
        #     self.transform = transform
        #     self.target_transform = target_transform
        #     self.attributes = self.duke_dataset.attributes
        #     self.attr_num = self.duke_dataset.attr_num
        #     self.eval_attr_num = self.duke_dataset.eval_attr_num
        #     self.img_id = self.duke_dataset.img_names
        #     self.label = self.duke_dataset.label
        #     self.label_all = self.label
        #     self.use_duke = True
        #     self.words = self.attributes  # 兼容性
        #     return  # 重要：直接返回，避免初始化 EventPAR 字段

        # ===== 以下是原 EventPAR 逻辑 =====
        dataset_info = pickle.load(open("/root/autodl-tmp/datasets/EventPAR/annotation/dataset_reorder.pkl",'rb+'))
        img_id = dataset_info.image_name
        attr_label = dataset_info.label

        assert split in dataset_info.partition.keys(), f'split {split} is not exist'

        self.transform = transform
        self.target_transform = target_transform
        self.attributes = dataset_info.attr_name
        self.root_path = '/root/autodl-tmp/datasets/EventPAR/EventPAR'
        print(self.root_path)
        self.attr_num = len(self.attributes)
        self.eval_attr_num = len(self.attributes)
        self.img_idx = dataset_info.partition[split]

        if isinstance(self.img_idx, list):
            self.img_idx = self.img_idx[0]
        self.img_num = self.img_idx.shape[0]
        self.img_id = [img_id[i] for i in self.img_idx]

        self.label = attr_label[self.img_idx]
        self.label_all = self.label
        
        self.label_vector = dataset_info.attr_vectors
        self.label_word = np.array(dataset_info.attr_name)
        self.words = self.label_word.tolist()

    def __getitem__(self, index):
        
        # if self.cfg.DATASET.NAME == 'Mars':
        #     imgname = self.img_id[index]         # 例如 '0001C1T0002'
        #     gt_label = self.label[index].astype(np.float32)
        #     # rgb_dir = os.path.join(self.root_path, imgname)
        #     # event_dir = os.path.join(self.root_path, f'{imgname}_event')

        #     # # 列出并排序帧文件
        #     # rgb_frames = sorted([f for f in os.listdir(rgb_dir) if os.path.isfile(os.path.join(rgb_dir, f))])
        #     # # event_frames = sorted([f for f in os.listdir(event_dir) if os.path.isfile(os.path.join(event_dir, f))])

        #     # # RGB：选择第二帧（如果不足两帧则回退到第一帧）
        #     # rgb_idx = 1 if len(rgb_frames) > 1 else 0
        #     # rgb_choice = rgb_frames[rgb_idx]

        #     # # # Event：选择前 5 帧（如果不足 5 帧则全部）
        #     # # k = min(5, len(event_frames))
        #     # # event_choices = event_frames[:k]

        #     # # 读取与变换
        #     # # rgb_img = _safe_open_image(os.path.join(rgb_dir, rgb_choice))
        #     # rgb_img = Image.open(os.path.join(rgb_dir, rgb_choice)).convert('RGB')
        #     # if self.transform is not None:
        #     #     rgb_img = self.transform(rgb_img)
        #     # rgb_tensor = torch.stack([rgb_img])  # [1, C, H, W]

        #     # # --- Event 图像处理 (直接取前5帧) ---
        #     # event_dir = os.path.join(self.root_path, f'{imgname}_event')
        #     # event_frames = sorted(os.listdir(event_dir))
        #     # event_choices = event_frames[:3] # 直接切片获取前5个文件名

        #     # event_tensors = []
        #     # for fname in event_choices:
        #     #     # ev_img = _safe_open_image(os.path.join(event_dir, fname))
        #     #     ev_img = Image.open(os.path.join(event_dir, fname)).convert('RGB')
        #     #     if self.transform is not None:
        #     #         ev_img = self.transform(ev_img)
        #     #     event_tensors.append(ev_img)

        #     # event_tensor = torch.stack(event_tensors)  # [K(<=5), C, H, W]
        #     # 固定参数
        #     K = 5
        #     ALLOWED_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')

        #     rgb_dir = os.path.join(self.root_path, imgname)
        #     event_dir = os.path.join(self.root_path, f'{imgname}_event')

        #     # 列出并过滤为图像文件
        #     rgb_frames = sorted([
        #         f for f in os.listdir(rgb_dir)
        #         if os.path.isfile(os.path.join(rgb_dir, f)) and f.lower().endswith(ALLOWED_EXTS)
        #     ])
        #     event_frames = sorted([
        #         f for f in os.listdir(event_dir)
        #         if os.path.isfile(os.path.join(event_dir, f)) and f.lower().endswith(ALLOWED_EXTS)
        #     ])

        #     # RGB：第2帧，不足则第1帧
        #     rgb_idx = 1 if len(rgb_frames) > 1 else 0
        #     rgb_choice = rgb_frames[rgb_idx]
        #     rgb_img = Image.open(os.path.join(rgb_dir, rgb_choice)).convert('RGB')
        #     if self.transform is not None:
        #         rgb_img = self.transform(rgb_img)
        #     rgb_tensor = torch.stack([rgb_img])  # [1, C, H, W]

        #     # 事件：严格取前5帧；任一读取失败或不足，用零帧补齐
        #     event_imgs = []
        #     chosen = event_frames[:K]
 
        #     # 用 RGB 的 shape/dtype 构造零帧，确保一致的 [C,H,W]
        #     if isinstance(rgb_img, torch.Tensor):
        #         zero_frame = torch.zeros_like(rgb_img)
        #     else:
        #         # 保底尺寸（若 transform 为空的情况），用 YAML 中的尺寸
        #         C, H, W = 3, self.cfg.DATASET.HEIGHT, self.cfg.DATASET.WIDTH
        #         zero_frame = torch.zeros(C, H, W, dtype=torch.float32)
            
        #     for fname in chosen:
        #         try:
        #             ev_img = Image.open(os.path.join(event_dir, fname)).convert('RGB')
        #             if self.transform is not None:
        #                 ev_img = self.transform(ev_img)
        #             # 保险：强制形状一致（非常用，通常 transform 已经保证了）
        #             if ev_img.shape != zero_frame.shape:
        #                 # 可根据你需要统一为 zero_frame 形状，这里直接跳过并用零帧
        #                 ev_img = zero_frame
        #             event_imgs.append(ev_img)
        #         except Exception:
        #             event_imgs.append(zero_frame)

        #     # 补齐到5帧
        #     while len(event_imgs) < K:
        #         event_imgs.append(zero_frame)

        #     event_tensor = torch.stack(event_imgs)  # [5, C, H, W]

        #     if self.target_transform:
        #         gt_label = gt_label[self.target_transform]

        #     return rgb_tensor, event_tensor, gt_label, imgname
        # # DUKE 分流：直接使用子数据集
        # if getattr(self, 'use_duke', False):
        #     return self.duke_dataset[index]

        # EventPAR 逻辑
        imgname, gt_label, imgidx = self.img_id[index], self.label[index], self.img_idx[index]

        event_imgpath = os.path.join(self.root_path, imgname, 'event_frames')
        rgb_imgpath = os.path.join(self.root_path, imgname, 'rgb_degraded')
        event_imgs = os.listdir(event_imgpath)
        rgb_imgs = sorted(os.listdir(rgb_imgpath))

        rgb_imgs_trans=[]
        event_imgs_trans=[]

        rgb_img_pil = Image.open(os.path.join(rgb_imgpath, rgb_imgs[2]))
        if self.transform is not None:
            rgb_img_pil = self.transform(rgb_img_pil)
        rgb_imgs_trans.append(rgb_img_pil)
            
        for event_img in event_imgs:
            event_img_pil = Image.open(os.path.join(event_imgpath, event_img))
            if self.transform is not None:
                event_img_pil = self.transform(event_img_pil)
            event_imgs_trans.append(event_img_pil)

        rgb_imgs_trans = torch.stack(rgb_imgs_trans)
        event_imgs_trans = torch.stack(event_imgs_trans)

        gt_label = gt_label.astype(np.float32)
        if self.target_transform:
            gt_label = gt_label[self.target_transform]

        return rgb_imgs_trans, event_imgs_trans, gt_label, imgname

    def __len__(self):
        # # DUKE 分流
        # if getattr(self, 'use_duke', False):
        #     return len(self.duke_dataset)
        # EventPAR
        return len(self.img_id)