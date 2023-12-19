import os
import pickle

import numpy as np
import torch.utils.data as data
from PIL import Image

from tools.function import get_pkl_rootpath
import torchvision.transforms as T
class MultiModalAttrDataset(data.Dataset):

    def __init__(self, split, args, transform=None, target_transform=None):
        if args.pkl_path is None:
            assert args.dataset in ['PA100k', 'RAPV1','RAPV2','PETA','WIDER','RAPzs','PETAzs'], \
                f'dataset name {args.dataset} is not exist,The legal name is PA100k,RAPV1,RAPV2,PETA,WIDER'
                
            dataset_dir='/amax/DATA/jinjiandong/dataset/' #Set this to the directory where the dataset is located
            if args.dataset=='RAPV1' :
                dataset_info = pickle.load(open(dataset_dir+args.dataset+"/expand_pad.pkl", 'rb+'))
            elif args.dataset!='RAPzs' and args.dataset!='PETAzs':
                dataset_info = pickle.load(open(dataset_dir+args.dataset+"/pad.pkl", 'rb+'))
            elif args.dataset=='RAPzs':
                dataset_info = pickle.load(open(dataset_dir+'RAPV2'+"/dataset_zs_pad.pkl", 'rb+'))
            elif args.dataset=='PETAzs':
                dataset_info = pickle.load(open(dataset_dir+'PETA'+"/dataset_zs_pad.pkl", 'rb+'))
        else :
            dataset_info = pickle.load(open(args.pkl_path, 'rb+'))
        img_id = dataset_info.image_name
        attr_label = dataset_info.label

        assert split in dataset_info.partition.keys(), f'split {split} is not exist'

        self.dataset = args.dataset
        self.transform = transform
        self.target_transform = target_transform

        if args.datasets_path is None :
            self.root_path = dataset_info.root
        else :
            self.root_path = args.datasets_path

        self.attr_id = dataset_info.attr_name
        self.attr_num = len(self.attr_id)

        self.attributes=dataset_info.attributes
        self.neg_attr_words=dataset_info.neg_attr_words
        self.expand_pos_attr_words=dataset_info.expand_pos_attr_words
        self.expand_neg_attr_words=dataset_info.expand_neg_attr_words        
        self.img_idx = dataset_info.partition[split]

        if isinstance(self.img_idx, list):
            self.img_idx = self.img_idx[0]
        self.img_num = self.img_idx.shape[0]
        self.img_id = [img_id[i] for i in self.img_idx]

        self.label = attr_label[self.img_idx]
        self.label_all = self.label

    def __getitem__(self, index):
        imgname, gt_label, imgidx = self.img_id[index], self.label[index], self.img_idx[index]
        imgpath = os.path.join(self.root_path, imgname)
        img_pil = Image.open(imgpath)
        if self.transform is not None:
            img_pil = self.transform(img_pil)

        gt_label = gt_label.astype(np.float32)
        
        if self.target_transform is not None:
            gt_label = self.transform(gt_label)
        
        return img_pil, gt_label, imgname

    def __len__(self):
        return len(self.img_id)

def get_transform(args):
    height = args.height
    width = args.width
    normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    train_transform = T.Compose([
        T.Resize((height, width)),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])

    valid_transform = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        normalize
    ])

    return train_transform, valid_transform
