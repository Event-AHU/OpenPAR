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
import random
from torchvision.transforms import functional as F
import random
from loss.CE_loss import CEL_Sigmoid

class PA100k(data.Dataset):

    def __init__(self, split, transform=None, target_transform=None, k=None, imgidx=None):


        #dataset_info = pickle.load(open('/wangx/DATA/Dataset/PA100k/dataset.pkl', 'rb+'))
        with open('/wangx/DATA/Dataset/PA100k/dataset.pkl', 'rb') as f:
            dataset_info = pickle.load(f)
        #  "/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c02/DATA/dataset/PA100k/dataset.pkl"
        img_id = dataset_info.image_name
        attr_label = dataset_info.label

        

        self.dataset = 'PA100k'
        self.transform = transform
        self.target_transform = target_transform

        self.root_path = "/wangx/DATA/Dataset/PA100k/data/"
        self.attributes=dataset_info.attr_words
        self.label_vector = dataset_info.attr_vectors

        self.attr_num = len(self.attributes)

        self.img_idx = dataset_info.partition[split]
        if isinstance(self.img_idx, list):
            self.img_idx = self.img_idx[0]

        if imgidx is not None:
            selected = np.random.permutation(self.img_idx)[:k]
            combined = np.concatenate([selected, np.array(imgidx)])
            self.img_idx = np.array(np.unique(combined), dtype=np.int64)

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
        
        label_v = self.label_vector.astype(np.float32)
        return img_pil, gt_label, label_v, imgidx

    def __len__(self):
        return len(self.img_id)

def get_PA100k_transform(args):
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

class MSP60k(data.Dataset):

    def __init__(self, split, transform=None, target_transform=None, k=None, imgidx=None):


        #dataset_info = pickle.load(open('/wangx/DATA/Dataset/PA100k/dataset.pkl', 'rb+'))
        with open("/wangx/DATA/Dataset/MSP60k/SUBMIT/dataset_random.pkl", 'rb') as f:
            dataset_info = pickle.load(f)
        #  "/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c02/DATA/dataset/PA100k/dataset.pkl"
        img_id = dataset_info.image_name
        attr_label = dataset_info.label

        

        self.dataset = 'MSP60k'
        self.transform = transform
        self.target_transform = target_transform

        self.root_path = "/wangx/DATA/Dataset/MSP60k/SUBMIT/images/"
        self.attributes=dataset_info.attr_words
        self.label_vector = dataset_info.attr_vectors

        self.attr_num = len(self.attributes)

        self.img_idx = dataset_info.partition[split]
        if isinstance(self.img_idx, list):
            self.img_idx = self.img_idx[0]

        if imgidx is not None:
            selected = np.random.permutation(self.img_idx)[:k]
            combined = np.concatenate([selected, np.array(imgidx)])
            self.img_idx = np.array(np.unique(combined), dtype=np.int64)

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
        
        label_v = self.label_vector.astype(np.float32)
        return img_pil, gt_label, label_v, imgidx

    def __len__(self):
        return len(self.img_id)

def get_MSP60k_transform(args):
    height = args.height
    width = args.width
    
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
                                                           
    random_erasing = T.RandomErasing(
        p=0.5,
        scale=(0.02,0.33),
        ratio=(0.3,3.3),
        value='random', 
        inplace=False 
    )
    train_transform = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(0.5),
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

class EventPAR(data.Dataset):

    def __init__(self, split, transform=None, target_transform=None, imgidx=None, k=None):



        dataset_info = pickle.load(open("/wdata/Dataset/EventPAR/annotation_EventPAR/dataset_reorder.pkl", 'rb+'))

        img_id = dataset_info.image_name
        attr_label = dataset_info.label

        assert split in dataset_info.partition.keys(), f'split {split} is not exist'

        self.dataset = 'EventPAR'
        self.transform = transform
        self.target_transform = target_transform


        self.attributes=dataset_info.attr_name
        self.root_path = "/wangx/DATA/Dataset/EventPAR/"
        self.attr_num = len(self.attributes)
        
        
        self.img_idx = dataset_info.partition[split]

        if isinstance(self.img_idx, list):
            self.img_idx = self.img_idx[0]

        if imgidx is not None:
            selected = np.random.permutation(self.img_idx)[:k]
            combined = np.concatenate([selected, np.array(imgidx)])
            self.img_idx = np.array(np.unique(combined), dtype=np.int64)

        self.img_num = self.img_idx.shape[0]
        self.img_id = [img_id[i] for i in self.img_idx]

        self.label = attr_label[self.img_idx]
        self.label_all = self.label

        self.label_vector = dataset_info.attr_vectors

        self.multi=True

    def __getitem__(self, index):
        imgname, gt_label, imgidx = self.img_id[index], self.label[index], self.img_idx[index]
        event_imgpath=os.path.join(self.root_path,imgname,'event_frames')
        rgb_imgpath=os.path.join(self.root_path, imgname,'rgb_raw')   #rgb is the original data, and rgbV3 is the noise-added data.
        event_imgs = os.listdir(event_imgpath)
        rgb_imgs = sorted(os.listdir(rgb_imgpath))


        rgb_imgs_trans=[]
        event_imgs_trans=[]
        # self.multi=True
        if self.multi:

            for rgb in rgb_imgs:
                rgb_img_pil = Image.open(os.path.join(rgb_imgpath, rgb))
                if self.transform is not None:
                    rgb_img_pil = self.transform(rgb_img_pil)
                rgb_imgs_trans.append(rgb_img_pil)
            rgb_imgs_trans = torch.stack(rgb_imgs_trans)
         
            for event_img in event_imgs:
                event_img_pil = Image.open(os.path.join(event_imgpath, event_img))
                
                if self.transform is not None:
                    event_img_pil = self.transform(event_img_pil)
                event_imgs_trans.append(event_img_pil)
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
            return all_imgs_trans, gt_label, label_v, imgidx
        else:
            return imgname,rgb_imgs_trans,gt_label, label_v


    def __len__(self):
        return len(self.img_id)
    
class DUKE(data.Dataset):

    def __init__(self, split, transform=None, target_transform=None, repet=None, k=None, imgidx=None):

        # 没有pad_duke.pkl文件
        #dataset_info = pickle.load(open("/wangx/DATA/Dataset/DUKE/pad_duke.pkl", 'rb+'))
        with open('/wangx/DATA/Dataset/DUKE/pad_duke.pkl', 'rb') as f:
            dataset_info = pickle.load(f)


        img_id = dataset_info.track_name
        #img_id = dataset_info.image_name
        attr_label = dataset_info.label

        assert split in dataset_info.partition.keys(), f'split {split} is not exist'

        self.dataset = 'DUKE'
        self.transform = transform
        self.target_transform = target_transform

        

        self.attributes=dataset_info.attr_name
        self.root_path = "/wangx/DATA/Dataset/pad_duke_dataset_event/"

        self.attr_num = len(self.attributes)
        
        
        self.img_idx = dataset_info.partition[split]

        if isinstance(self.img_idx, list):
            self.img_idx = self.img_idx[0]

        if imgidx is not None:
            selected = np.random.permutation(self.img_idx)[:k]
            combined = np.concatenate([selected, np.array(imgidx)])
            self.img_idx = np.array(np.unique(combined), dtype=np.int64)


        self.img_num = self.img_idx.shape[0]
        img_id = [img_id[i] for i in self.img_idx]
        if repet is not None and repet > 1:
            self.img_id = img_id * repet
        else:
            self.img_id = img_id

        self.label = attr_label[self.img_idx]
        self.label_all = self.label

        self.label_vector = dataset_info.attr_vectors

        self.result_dict=dataset_info.result_dict

    def __getitem__(self, index):

        imgname= self.img_id[index]
        gt_label = self.result_dict[imgname]
        imgidx = self.img_idx[index]
         
        rgb_imgpath = os.path.join(self.root_path, imgname)
        event_imgpath =rgb_imgpath.replace(imgname,imgname+'_event')
        rgb_img_list=sorted(os.listdir(rgb_imgpath))
        rgb_list=[]
        event_list=[]

        for filename in random.choices(rgb_img_list,k=5):
            rgb_flie = os.path.join(rgb_imgpath, filename)
            event_file= os.path.join(event_imgpath,filename)
            rgb_pil = Image.open(rgb_flie)
            event_pil = Image.open(event_file)
            if self.transform is not None:
                rgb_pil = self.transform(rgb_pil)
                event_pil = self.transform(event_pil)
            rgb_list.append(rgb_pil)
            event_list.append(event_pil)
        image_rgb = torch.stack(rgb_list, dim=0)
        image_event = torch.stack(event_list, dim=0)
        all_imgs_trans = [image_rgb,image_event]

      
           
        gt_label = gt_label.astype(np.float32)
        
        if self.target_transform is not None:
            gt_label = self.transform(gt_label)

        label_v = self.label_vector.astype(np.float32)
        
        return image_rgb, gt_label, label_v, imgidx
        



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

class MultiDataset(data.Dataset):
    def __init__(self):
        self.datasets = {}  # 存储所有数据集
        self.current_set = None  # 当前激活的数据集
        self.current_name = None  # 当前数据集名称

    def __len__(self):
        if self.current_set is None:
            return 0
        return len(self.current_set)

    def __getitem__(self, index):
        if self.current_set is None:
            raise RuntimeError("No dataset is active. Call init_set() first.")
        return self.current_set[index]

    def append_set(self, name, dataset):
        """添加一个新数据集"""
        self.datasets[name] = dataset
        # 首次添加时自动激活
        if self.current_set is None:
            self.init_set(name)

    def init_set(self, name):
        """切换当前使用的数据集"""
        if name not in self.datasets:
            raise KeyError(f"Dataset {name} not found. Available: {list(self.datasets.keys())}")
        self.current_set = self.datasets[name]
        self.current_name = name

    def get_current_name(self):
        """获取当前数据集名称"""
        return self.current_name
    

def get_multi_dataset(args):
    multi_train_set = MultiDataset()
    multi_valid_set = MultiDataset()
    criterion_dict = {}
    #breakpoint()
    #datasets = ['PA100k','EventPAR']
    #for dataset in args.dataset:
    for dataset in args.dataset:
        if dataset=='PA100k':
            train_tsfm, valid_tsfm = get_PA100k_transform(args)
            train_set = PA100k(split=args.train_split, transform=train_tsfm)
            valid_set = PA100k(split=args.valid_split, transform=valid_tsfm)
            multi_train_set.append_set(name=dataset, dataset=train_set)
            multi_valid_set.append_set(name=dataset, dataset=valid_set)
            labels = train_set.label
            sample_weight = labels.mean(0)
            criterion_dict['PA100k'] = CEL_Sigmoid(sample_weight, attr_idx=train_set.attr_num)
        
        elif dataset=='MSP60k':
            train_tsfm, valid_tsfm = get_MSP60k_transform(args)
            train_set = MSP60k(split=args.train_split, transform=train_tsfm)
            valid_set = MSP60k(split=args.valid_split, transform=valid_tsfm)
            multi_train_set.append_set(name=dataset, dataset=train_set)
            multi_valid_set.append_set(name=dataset, dataset=valid_set)
            labels = train_set.label
            sample_weight = labels.mean(0)
            criterion_dict['MSP60k'] = CEL_Sigmoid(sample_weight, attr_idx=train_set.attr_num)

        elif dataset=='EventPAR':
            train_tsfm, valid_tsfm = get_transform(args)
            train_set = EventPAR(split='train', transform=train_tsfm)
            valid_set = EventPAR(split=args.valid_split, transform=valid_tsfm)
            multi_train_set.append_set(name=dataset, dataset=train_set)
            multi_valid_set.append_set(name=dataset, dataset=valid_set)
            labels = train_set.label
            sample_weight = labels.mean(0)
            criterion_dict['EventPAR'] = CEL_Sigmoid(sample_weight, attr_idx=train_set.attr_num)
        
        elif dataset=='DUKE':
            train_tsfm, valid_tsfm = get_transform(args)
            train_set = DUKE(split=args.train_split, transform=train_tsfm)
            valid_set = DUKE(split=args.valid_split, transform=valid_tsfm)
            multi_train_set.append_set(name=dataset, dataset=train_set)
            multi_valid_set.append_set(name=dataset, dataset=valid_set)
            labels = train_set.label
            sample_weight = labels.mean(0)
            criterion_dict['DUKE'] = CEL_Sigmoid(sample_weight, attr_idx=train_set.attr_num)
    return multi_train_set, multi_valid_set, criterion_dict
    # return multi_train_set


class MixMultiDataset(data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.lengths = [len(dataset) for dataset in datasets]
        self.offsets = [0] + list(np.cumsum(self.lengths))[:-1]


    def __getitem__(self, index):
        for i, offset in enumerate(self.offsets):
            if index < offset + self.lengths[i]:
                sample = self.datasets[i][index - offset]
                # 添加一个额外的标签表示数据集的索引
                
                return sample + (i,)
        raise IndexError

    def __len__(self):
        return sum(self.lengths)
    
    def get_N(self):
        N_list = []
        attr_num = []
        for dataset in self.datasets:
            imgs, labels, _, _ = dataset[0]
            if isinstance(imgs, list):
                N_list.append(imgs[0].shape[0] + imgs[1].shape[0])
                attr_num.append(labels.shape[0])
            elif imgs.dim() == 4:
                N_list.append(imgs.shape[0])
                attr_num.append(labels.shape[0])
            else:
                N_list.append(1)
                attr_num.append(labels.shape[0])

        return N_list, attr_num
            
    
def custom_collate(batch, max_attr_num=100):

    imgs = [item[0] for item in batch]
    gt_labels = [item[1] for item in batch]
    label_vs = [item[2] for item in batch]
    imgidxs = [item[3] for item in batch]
    dataset_ids = [item[4] for item in batch]
    processed_imgs = []
    dataset_masks = []

    for img in imgs:
        if isinstance(img, torch.Tensor) and img.dim() == 3:
            processed_imgs.append(img)
            dataset_masks.append(0)  # 数据集编号0
            # print(dataset_ids,0)
            
        # 情况2: 预批处理图像 torch.Size([N,C,H,W])
        if isinstance(img, torch.Tensor) and img.dim() == 4:
            N = img.shape[0]
            processed_imgs.extend([img[i] for i in range(N)])
            dataset_masks.extend([1] * N)  # 数据集编号1
            # print(dataset_ids,1)
        elif isinstance(img, list):
            rgb, event = img[0], img[1]
            N1, N2 = rgb.shape[0], event.shape[0]
            processed_imgs.extend([rgb[i] for i in range(N1)])
            processed_imgs.extend([event[i] for i in range(N2)])
            dataset_masks.extend([2] * (N1 + N2))  # 数据集编号2
            # print(dataset_ids,2)
    imgs = torch.stack(processed_imgs, 0)  # (N',C,H,W)
    mask_id = torch.tensor(dataset_masks, dtype=torch.long)  # (N',)


    dataset_ids = torch.tensor(dataset_ids)

    # # 处理不同尺寸的 gt_labels
    gt_labels_padded = []
    label_vs_padded = []

    for label in gt_labels:
        padded_label = torch.nn.functional.pad(torch.tensor(label), (0, max_attr_num - label.shape[0]), value=0)
        gt_labels_padded.append(padded_label)
    gt_labels = torch.stack(gt_labels_padded, 0)

    for label_v in label_vs:
        padded_label = torch.nn.functional.pad(
            torch.tensor(label_v),
            (0, 0, 0, max_attr_num - label_v.shape[0]),  # 只在第0维度(行)填充
            value=0
        )
        label_vs_padded.append(padded_label)
    label_vs = torch.stack(label_vs_padded, 0)


    return imgs, gt_labels, label_vs, dataset_ids, mask_id, imgidxs

def get_mix_multi_dataset(args):
    train_set_list = []
    multi_valid_set = MultiDataset()
    criterion_dict = {}
    for dataset in args.dataset:
        if dataset=='PA100k':
            train_tsfm, valid_tsfm = get_PA100k_transform(args)
            train_set = PA100k(split=args.train_split, transform=train_tsfm)
            valid_set = PA100k(split=args.valid_split, transform=valid_tsfm)
            train_set_list.append(train_set)
            multi_valid_set.append_set(name=dataset, dataset=valid_set)
            labels = train_set.label
            sample_weight = labels.mean(0)
            criterion_dict['PA100k'] = CEL_Sigmoid(sample_weight, attr_idx=train_set.attr_num)

        elif dataset=='MSP60k':
            train_tsfm, valid_tsfm = get_MSP60k_transform(args)
            train_set = MSP60k(split=args.train_split, transform=train_tsfm)
            valid_set = MSP60k(split=args.valid_split, transform=valid_tsfm)
            train_set_list.append(train_set)
            multi_valid_set.append_set(name=dataset, dataset=valid_set)
            labels = train_set.label
            sample_weight = labels.mean(0)
            criterion_dict['MSP60k'] = CEL_Sigmoid(sample_weight, attr_idx=train_set.attr_num)       
        
        elif dataset=='EventPAR':
            train_tsfm, valid_tsfm = get_transform(args)
            train_set = EventPAR(split='train', transform=train_tsfm)
            valid_set = EventPAR(split=args.valid_split, transform=valid_tsfm)
            train_set_list.append(train_set)
            multi_valid_set.append_set(name=dataset, dataset=valid_set)
            labels = train_set.label
            sample_weight = labels.mean(0)
            criterion_dict['EventPAR'] = CEL_Sigmoid(sample_weight, attr_idx=train_set.attr_num)
        
        elif dataset in 'DUKE':
            train_tsfm, valid_tsfm = get_transform(args)
            train_set = DUKE(split=args.train_split, transform=train_tsfm, repet=args.repet_DUKE)
            valid_set = DUKE(split=args.valid_split, transform=valid_tsfm)
            train_set_list.append(train_set)
            multi_valid_set.append_set(name=dataset, dataset=valid_set)
            labels = train_set.label
            sample_weight = labels.mean(0)
            criterion_dict['DUKE'] = CEL_Sigmoid(sample_weight, attr_idx=train_set.attr_num)
    multi_train_set = MixMultiDataset(train_set_list)
    return multi_train_set, multi_valid_set, criterion_dict

def get_mix_balance_dataset(args, imgidx_PA100k, imgidx_DUKE):
    train_set_list = []
    multi_valid_set = MultiDataset()
    criterion_dict = {}
    for dataset in args.dataset:
        if dataset=='PA100k':
            train_tsfm, valid_tsfm = get_PA100k_transform(args)
            train_set = PA100k(split=args.train_split, transform=train_tsfm, k=len(imgidx_PA100k), imgidx=imgidx_PA100k)
            valid_set = PA100k(split=args.valid_split, transform=valid_tsfm)
            train_set_list.append(train_set)
            multi_valid_set.append_set(name=dataset, dataset=valid_set)
            labels = train_set.label
            sample_weight = labels.mean(0)
            criterion_dict['PA100k'] = CEL_Sigmoid(sample_weight, attr_idx=train_set.attr_num)

        elif dataset=='MSP60k':
            train_tsfm, valid_tsfm = get_MSP60k_transform(args)
            train_set = MSP60k(split=args.train_split, transform=train_tsfm, k=len(imgidx_MSP60k), imgidx=imgidx_PA100k)
            valid_set = MSP60k(split=args.valid_split, transform=valid_tsfm)
            train_set_list.append(train_set)
            multi_valid_set.append_set(name=dataset, dataset=valid_set)
            labels = train_set.label
            sample_weight = labels.mean(0)
            criterion_dict['MSP60k'] = CEL_Sigmoid(sample_weight, attr_idx=train_set.attr_num)       

        elif dataset=='EventPAR':
            train_tsfm, valid_tsfm = get_transform(args)
            train_set = EventPAR(split='train', transform=train_tsfm, k=10000, imgidx=[])
            valid_set = EventPAR(split=args.valid_split, transform=valid_tsfm)
            train_set_list.append(train_set)
            multi_valid_set.append_set(name=dataset, dataset=valid_set)
            labels = train_set.label
            sample_weight = labels.mean(0)
            criterion_dict['EventPAR'] = CEL_Sigmoid(sample_weight, attr_idx=train_set.attr_num)
        
        elif dataset in 'DUKE':
            train_tsfm, valid_tsfm = get_transform(args)
            train_set = DUKE(split=args.train_split, transform=train_tsfm, repet=args.repet_DUKE, k=len(imgidx_DUKE), imgidx=imgidx_DUKE)
            valid_set = DUKE(split=args.valid_split, transform=valid_tsfm)
            train_set_list.append(train_set)
            multi_valid_set.append_set(name=dataset, dataset=valid_set)
            labels = train_set.label
            sample_weight = labels.mean(0)
            criterion_dict['DUKE'] = CEL_Sigmoid(sample_weight, attr_idx=train_set.attr_num)
    multi_train_set = MixMultiDataset(train_set_list)
    return multi_train_set, multi_valid_set, criterion_dict
