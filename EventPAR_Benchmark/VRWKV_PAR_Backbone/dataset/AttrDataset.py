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
     
        if args.dataset=='PETA':
            dataset_info = pickle.load(open('/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c01/DATA/dataset/PETA/dataset.pkl', 'rb+'))
            # dataset_info = pickle.load(open('/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c0/DATA/songhaoyu/PAR/VRWKV_PAR/dataset/test_PETA/dataset_temp0.pkl', 'rb+'))
        elif args.dataset=='RAPV1':
            dataset_info = pickle.load(open('/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c0/DATA/songhaoyu/PAR/VRWKV_PAR/dataset/RAPV1_dataset.pkl', 'rb+'))
        elif args.dataset=='PA100k':
            dataset_info = pickle.load(open("/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c01/DATA/dataset/PA100k/dataset.pkl", 'rb+'))
        elif args.dataset=='EventPAR':
            dataset_info = pickle.load(open("/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c02/DATA/dataset/annotation/dataset_reorder.pkl", 'rb+'))

        img_id = dataset_info.image_name
        attr_label = dataset_info.label

        assert split in dataset_info.partition.keys(), f'split {split} is not exist'

        self.dataset = args.dataset
        self.transform = transform
        self.target_transform = target_transform

        

        if args.dataset=='PA100k' or args.dataset=='RAPV1':
            self.attributes=dataset_info.attr_words
            #self.root_path = dataset_info.root
            self.root_path = ""
        elif args.dataset=='PETA':
            self.attributes=dataset_info.attr_name
            self.root_path = ""
        elif args.dataset=='EventPAR':
            self.attributes=dataset_info.attr_name
            self.root_path = "/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c02/DATA/dataset/EventPAR/"
        self.attr_num = len(self.attributes)
        
        
        self.img_idx = dataset_info.partition[split]

        if isinstance(self.img_idx, list):
            self.img_idx = self.img_idx[0]
        self.img_num = self.img_idx.shape[0]
        self.img_id = [img_id[i] for i in self.img_idx]

        self.label = attr_label[self.img_idx]
        self.label_all = self.label

        self.label_vector = dataset_info.attr_vectors
        self.frames=5

        self.multi=args.multi

    def __getitem__(self, index):
        imgname, gt_label, imgidx = self.img_id[index], self.label[index], self.img_idx[index]
        event_imgpath=os.path.join(self.root_path,imgname,'event')
        rgb_imgpath=os.path.join(self.root_path, imgname,'rgbv3')
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
            #imgs_trans = torch.cat([rgb_imgs_trans, event_imgs_trans],dim=0)
        else:
            for rgb in rgb_imgs[2:3]:
            #for rgb in rgb_imgs:
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
            return all_imgs_trans, gt_label, label_v
        else:
            return rgb_imgs_trans,gt_label, label_v


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




# def get_transform(args):
#     height = args.height
#     width = args.width

    
#     normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#     train_transform = T.Compose([
#         T.Resize((height, width)),
#         T.Pad(10),
#         T.RandomCrop((height, width)),
#         T.RandomHorizontalFlip(),
#         T.ToTensor(),
#         normalize,
#     ])

#     valid_transform = T.Compose([
#         T.Resize((height, width)),
#         T.ToTensor(),
#         normalize
#     ])

#     return train_transform, valid_transform

# def custom_collate_fn(batch):
#     # print(batch)
#     imgs, gt_labels, imgnames, imgtemps = zip(*batch)
#     imgs = default_collate(imgs)
#     gt_labels = default_collate(gt_labels)
#     imgnames = list(imgnames)
#     imgtemps = list(imgtemps)
#     return imgs, gt_labels, imgnames, imgtemps