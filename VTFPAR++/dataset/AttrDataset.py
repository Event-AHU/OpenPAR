import os
import pickle
from random import sample
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torch.utils.data import DataLoader
from config import argument_parser
from tools.function import get_pkl_rootpath
import torchvision.transforms as T
from operator import itemgetter
class MultiModalAttrDataset(data.Dataset):

    def __init__(self, split, args, transform=None, target_transform=None):
        assert args.dataset in ['MARS', 'DUKE'], \
            f'dataset name {args.dataset} is not exist,The legal name is MARS, DUKE'
        self.args = args

        if args.dataset =='MARS' :
            data_path='/path'
        else :
            data_path='/path'

        dataset_info = pickle.load(open(data_path, 'rb+'))

        track_id = dataset_info.track_name
        attr_label = dataset_info.label

        assert split in dataset_info.partition.keys(), f'split {split} is not exist'
        self.frames=args.frames
        self.dataset = args.dataset
        self.transform = transform
        self.target_transform = target_transform
        self.root_path = dataset_info.root
        self.attr_id = dataset_info.attr_name
        self.attr_num = len(self.attr_id)
        self.attributes=dataset_info.attr_name
        self.track_idx=dataset_info.partition[split]

        if isinstance(self.track_idx, list):
            self.track_idx = self.track_idx[0]
        self.track_num = self.track_idx.shape[0]
        self.track_id = [track_id[i] for i in self.track_idx]
        self.label = np.array(itemgetter(*self.track_id)(dataset_info.result_dict)) 
        self.label_all = self.label
        self.label_word = dataset_info.words
        self.label_vector=dataset_info.attr_vectors
        self.words = self.label_word.tolist()
        self.track_imgs_path=dataset_info.track_imgs_path
        self.result_dict=dataset_info.result_dict

    def __getitem__(self, index):
        trackname= self.track_id[index]
        gt_label = self.result_dict[trackname]
        gt_label=np.array(gt_label)
        track_img_path =self.track_imgs_path[trackname]
        imgs=[]
        if self.args.avg_frame_extract:
            my_sample = select_images(track_img_path, self.frames)
        else:
            if len(track_img_path)<=(self.frames-1):
                my_sample=np.random.choice(track_img_path, self.frames)
            else:
                my_sample=sample(track_img_path,self.frames)
        assert len(my_sample)==self.frames,print(len(my_sample))
        for i in my_sample:
            pil=Image.open(i)
            imgs.append(pil)
        imgs_trans=[]
        if self.transform is not None:
            for i in imgs:
                imgs_trans.append(self.transform(i))

        gt_label = gt_label.astype(np.float32)
        label_v=self.label_vector.astype(np.float32)

        if self.target_transform is not None:
            gt_label = self.transform(gt_label)

        return torch.stack(imgs_trans), gt_label,trackname,label_v
    def __len__(self):
        return len(self.track_id)

    selected_images = [sorted_img_paths[i] for i in range(0, len(sorted_img_paths), step)][:N]

    return selected_images

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
