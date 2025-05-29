import os
import copy
os.environ['CUDA_LAUNCH_BLOCKING'] = '3'
import numpy as np
import random
import pickle
import json
from easydict import EasyDict

np.random.seed(0)
random.seed(0)
from tqdm import tqdm
attributes=[
    "Female",   #[0]
    "Child", "Adult", "Elderly",    #[1:4]
    "Fat", "Normal", "Thin",    #[4:7]
    "Bald", "Long hair", "Black hair", "Hat", "Glasses", "Mask", "Helmet", "Scarf", "Gloves",   #[7:16]
    "Front", "Back", "Side",    #[16:19]
    "Short sleeves", "Long sleeves", "Shirt", "Jacket", "Suit",  "Vest", "Cotton-padded coat", "Coat", "Graduation gown", "Chef uniform",   #[19:29]
    "Trousers", "Shorts", "Jeans", "Long skirt", "Short skirt", "Dress",    #[29:35]
    "Leather shoes", "Casual shoes", "Boots", "Sandals", "Other shoes", #[35:40]
    "Backpack", "Shoulder bag", "Handbag", "Plastic bag", "Paper bag", "Suitcase", "Others",    #[40:47]
    "Making a phone call", "Smoking", "Hands behind back", "Arms crossed",  #[47:51]
    "Walking", "Running", "Standing", "Riding a bicycle", "Riding an scooter", "Riding a skateboard"    #[51:56]
]

def make_bad_label(labels):
    groups = [0,(1,4),(4,7),(7,16),(16,19),(19,29),(29,35),(35,40),(40,47),(47,51),(51,56)]
    # 随机交换每组中的唯一的 1
    bad_labels = []
    # breakpoint()
    for label in labels:
        bad_label = copy.deepcopy(label)
        for indexs in groups:
            if type(indexs) == tuple:
                start, end = indexs
                # breakpoint()
                # 找到当前组中唯一的 1 的索引
                ones_indices = np.where(label[start:end] == 1)[0] + start
                # 如果该组内有 `1`
                if len(ones_indices) > 0:
                    # 获取当前组的所有索引
                    group_indices = np.arange(start, end)
                    
                    # 从所有组内的位置随机选择新的索引，但必须不同于原来 ones_indices 的位置
                    available_indices = np.setdiff1d(group_indices, ones_indices)  # 除去已有1的位置
                    # 如果可用位置足够
                    if len(available_indices) >= len(ones_indices):
                        new_indices = np.random.choice(available_indices, size=len(ones_indices), replace=False)
                    else:
                        # 如果可用位置不足，则将所有 available_indices 用上
                        new_indices = available_indices
                    
                    # 保证原来有 `1` 的位置现在随机移动到新位置
                    # 将打乱后的索引前 len(ones_indices) 个设置为 1，其余设置为 0
                    bad_label[start:end] = 0  # 先将整个组设为 0
                    bad_label[new_indices] = 1  # 随机选择新的位置为 1

            else:
                # breakpoint()
                bad_label[indexs] = 1 - bad_label[indexs]
        bad_labels.append(bad_label)
    # breakpoint()
    return np.array(bad_labels)

def generate_pkl(save_dir, exist_pkl_path=None):
    exist_pkl = pickle.load(open(exist_pkl_path, 'rb+'))
    dataset = EasyDict()
    dataset = exist_pkl
    dataset.root = '/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c01/DATA/kongweizhe/PromptPAR/dataset/MSPAR/MSP_degrade/images'
    dataset.badlabel = make_bad_label(dataset.label)
    with open(os.path.join(save_dir, f'dataset.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)

save_dir = '/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c01/DATA/kongweizhe/PromptPAR/dataset/MSPAR'
exist_pkl_path = '/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c01/DATA/kongweizhe/PromptPAR/dataset/MSPAR/dataset_ms_split_1.pkl'
generate_pkl(save_dir, exist_pkl_path)