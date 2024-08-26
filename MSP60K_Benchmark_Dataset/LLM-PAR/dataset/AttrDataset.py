import os
import pickle

import numpy as np
import torch.utils.data as data
from PIL import Image
from timm.data.random_erasing import RandomErasing
from local import get_pkl_rootpath
import torchvision.transforms as T
import random

class MultiModalAttrDataset(data.Dataset):

    def __init__(self, split, args, transform=None, target_transform=None):
        assert args.dataset in ['PA100k', 'RAPv1','RAPv2','PETA','WIDER','RAPzs','PETAzs','MSP','MSPCD'], \
            f'dataset name {args.dataset} is not exist,The legal name is PA100k,RAPV1,RAPV2,PETA,WIDER'
        pkl_path, root_path = get_pkl_rootpath(args.dataset)
        dataset_info = pickle.load(open(pkl_path, 'rb+'))
        
        img_id = dataset_info.image_name
        attr_label = dataset_info.label

        assert split in dataset_info.partition.keys(), f'split {split} is not exist'

        self.dataset = args.dataset
        self.transform = transform
        self.target_transform = target_transform

        self.root_path = root_path
        self.attr_id = dataset_info.attributes
        self.attr_num = len(self.attr_id)

        self.attributes= dataset_info.attributes
        self.img_idx = dataset_info.partition[split]
        self.sentences = dataset_info.sentences
        self.max_length = dataset_info.max_length
        self.max_length = (self.max_length // 5 + 1 ) * 5
        self.limit_words = dataset_info.limit_word
        if isinstance(self.img_idx, list):
            self.img_idx = self.img_idx[0]
        self.img_num = self.img_idx.shape[0]
        self.img_id = [img_id[i] for i in self.img_idx]

        self.label = attr_label[self.img_idx]
        self.label_all = self.label
        # self.sentences_buf = dataset_info['data_dict']\
        self.all_sentence=[]
        for imgname in self.img_id:
            self.all_sentence.append(self.sentences[imgname])
        self.random_sentences=randomized_list = [self.all_sentence[i] for i in random.sample(range(len(self.all_sentence)), len(self.all_sentence))]   
    def __getitem__(self, index):
        imgname, gt_label, imgidx = self.img_id[index], self.label[index], self.img_idx[index]

        sentences=self.sentences[imgname]
        imgpath = os.path.join(self.root_path, imgname)
        img_pil = Image.open(imgpath)
        img_pil = self.transform(img_pil)

        gt_label = gt_label.astype(np.float32)
        
        if self.target_transform is not None:
            gt_label = self.transform(gt_label)
        
        return img_pil, gt_label, imgname, sentences, self.random_sentences[index]

    def __len__(self):
        return len(self.img_id)

def get_transform(args):
    height = args.height
    width = args.width
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    random_erasing = T.RandomErasing(
        p=0.5,               # Probability of applying the transform
        scale=(0.02, 0.33),  # Proportion of the erased area against input image
        ratio=(0.3, 3.3),    # Aspect ratio of the erased area
        value='random',      # Fill erased area with random values
        inplace=False        # Apply the transform out-of-place
    )
    train_transform = T.Compose([
                T.Resize((height, width), interpolation=3),
                T.RandomHorizontalFlip(0.5),
                T.Pad(10),
                T.RandomCrop((height, width)),
                T.ToTensor(),
                normalize,
                random_erasing,
            ])
    valid_transform = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        normalize
    ])

    return train_transform, valid_transform

