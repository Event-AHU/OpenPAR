import os
import pickle
from random import sample
import numpy as np
import torch.utils.data as data
from PIL import Image
import cv2
from torch.utils.data.dataloader import default_collate
from tools.function import get_pkl_rootpath
import torchvision.transforms as T
import torch
from torchvision.transforms import functional as F
import random
class MultiModalAttrDataset(data.Dataset):

    def __init__(self, split, args, transform=None, target_transform=None):

        assert args.dataset in ['PA100k', 'RAPV1','RAPV2','PETA','WIDER','RAPzs','PETAzs','UPAR','YCJC',"EventPAR"], \
            f'dataset name {args.dataset} is not exist,The legal name is PA100k,RAPV1,RAPV2,PETA,WIDER'
        
        if args.dataset=='EventPAR':
            dataset_info = pickle.load(open("/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c01/DATA/dataset/annotation/dataset_reorder.pkl", 'rb+'))

        img_id = dataset_info.image_name
        attr_label = dataset_info.label

        assert split in dataset_info.partition.keys(), f'split {split} is not exist'

        self.dataset = args.dataset
        self.transform = transform
        self.target_transform = target_transform

        

        
        if args.dataset=='EventPAR':
            self.attributes=dataset_info.attr_name
            self.root_path = "/path/to/your/folder"
        self.attr_num = len(self.attributes)
        
        
        self.img_idx = dataset_info.partition[split]

        if isinstance(self.img_idx, list):
            self.img_idx = self.img_idx[0]
        self.img_num = self.img_idx.shape[0]
        self.img_id = [img_id[i] for i in self.img_idx]

        self.label = attr_label[self.img_idx]
        self.label_all = self.label

        self.label_vector = dataset_info.attr_vectors

        self.multi=args.multi

    def __getitem__(self, index):
        imgname, gt_label, imgidx = self.img_id[index], self.label[index], self.img_idx[index]
        event_imgpath=os.path.join(self.root_path,imgname,'event_frames')
        rgb_imgpath=os.path.join(self.root_path, imgname,'rgb_raw')   #rgb_raw is the original data, and rgb_degraded is the noise-added data.
        event_imgs = os.listdir(event_imgpath)
        rgb_imgs = sorted(os.listdir(rgb_imgpath))


        rgb_imgs_trans=[]
        event_imgs_trans=[]
        
        if self.multi:

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
            all_imgs_trans = [rgb_imgs_trans,event_imgs_trans]
           
        else:
            for rgb in rgb_imgs:
                rgb_img_pil = Image.open(os.path.join(rgb_imgpath, rgb))
                if self.transform is not None:
                    rgb_img_pil = self.transform(rgb_img_pil)
                rgb_imgs_trans.append(rgb_img_pil)
            rgb_imgs_trans = torch.stack(rgb_imgs_trans)

      
           
        gt_label = gt_label.astype(np.float32)
        
        if self.target_transform is not None:
            gt_label = self.transform(gt_label)

        label_v = self.label_vector.astype(np.float32)
        if self.multi:
            return imgname,all_imgs_trans, gt_label, label_v
        else:
            return imgname,rgb_imgs_trans,gt_label, label_v


    def __len__(self):
        return len(self.img_id)


def get_transform(args):
    height = args.height
    width = args.width
    
    normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def pad_to_size(img):
        # 获取图像的原始大小
        img_width, img_height = img.size
        
        # 计算填充的大小
        pad_left = (width - img_width) // 2
        pad_right = width - img_width - pad_left
        pad_top = (height - img_height) // 2
        pad_bottom = height - img_height - pad_top
        image_pad=F.pad(
            img, (pad_left, pad_top, pad_right, pad_bottom), fill=0)
        
        # 使用黑色填充
        return image_pad
    train_transform = T.Compose([
        pad_to_size, 
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])

    valid_transform = T.Compose([
        pad_to_size, 
        T.ToTensor(),
        normalize
    ])

    return train_transform, valid_transform




