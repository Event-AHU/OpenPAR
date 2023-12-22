import os
import pickle

import numpy as np
import torch.utils.data as data
from PIL import Image
import cv2

from tools.function import get_pkl_rootpath
import torchvision.transforms as T
class MultiModalAttrDataset(data.Dataset):

    def __init__(self, split, args, transform=None, target_transform=None):

        assert args.dataset in ['PA100k', 'RAPV1','RAPV2','PETA','WIDER','RAPzs','PETAzs','UPAR','YCJC',], \
            f'dataset name {args.dataset} is not exist,The legal name is PA100k,RAPV1,RAPV2,PETA,WIDER'
            
        dataset_dir='/data/jinjiandong/datasets/' #Set this to the directory where the dataset is located
        if args.dataset!='RAPzs' and args.dataset!='PETAzs':
            dataset_info = pickle.load(open(dataset_dir+args.dataset+"/pad.pkl", 'rb+'))
        elif args.dataset=='RAPzs':
            dataset_info = pickle.load(open(dataset_dir+'RAPV2'+"/dataset_zs_pad.pkl", 'rb+'))
        elif args.dataset=='PETAzs':
            dataset_info = pickle.load(open(dataset_dir+'PETA'+"/dataset_zs_pad.pkl", 'rb+'))   
        #dataset_info = pickle.load(open('/data/jinjiandong/datasets/yichangjiance/pad.pkl', 'rb+'))

        img_id = dataset_info.image_name
        attr_label = dataset_info.label

        assert split in dataset_info.partition.keys(), f'split {split} is not exist'

        self.dataset = args.dataset
        self.transform = transform
        self.target_transform = target_transform

        self.root_path = dataset_info.root

        self.attr_id = dataset_info.attributes
        self.attr_num = len(self.attr_id)
        self.attributes=dataset_info.attributes
        
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
        
        # img=cv2.imread(imgpath)#112*410,#1108
        # heigth,width = img.shape[0],img.shape[1]   #获取图片的长和宽#255,104
        # n = heigth/224
        # #等比例缩小图片
        # new_heigth = heigth/n 
        # new_width = width/n
        # a,b,c=0,0,0
        # if (new_heigth <=224) and (new_width <=224):
        #     img = cv2.resize(img, (int(new_width), int(new_heigth)))
        #     a = int((224 - new_heigth) / 2)
        #     b = int((224 - new_width) / 2)
        #     change_width=a*2+img.shape[0]
        #     if(change_width<224): 
        #         c=a+224-change_width
        #     elif (change_width==224): 
        #         c=a
        #     else : 
        #         c=a-224+change_width
        #     #cv2.imwrite("saved_image_file\\"+str(1)+".jpg", cv2.copyMakeBorder(img, c,a, b, b, cv2.BORDER_CONSTANT, value=[255, 255, 255]))
        #     #打印保存成功
        #     #print("save {}.jpg successful !".format(1))
        # img_pil=Image.fromarray(cv2.cvtColor(cv2.copyMakeBorder(img, c,a, b, b, cv2.BORDER_CONSTANT, value=[0, 0, 0]), cv2.COLOR_BGR2RGB))
        
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
