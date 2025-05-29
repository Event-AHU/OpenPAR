import copy
import os
import numpy as np
import random
import pickle

from easydict import EasyDict
from scipy.io import loadmat
from sentence_transformers import SentenceTransformer
np.random.seed(0)
random.seed(0)

# note: ref by annotation.md
"""
attr_words = [
   'accessory hat','accessory muffler','accessory nothing','accessory sunglasses','accessory long hair',
   'upper body casual', 'upper body formal', 'upper body jacket', 'upper body logo', 'upper body plaid', 
   'upper body short sleeve', 'upper body thin stripes', 'upper body t-shirt','upper body other','upper body v-neck',
   'lower body Casual', 'lower body Formal', 'lower body Jeans', 'lower body Shorts', 'lower body Short Skirt','lower body Trousers',
   'foot wear Leather shoes', 'foot wear Sandals shoes', 'foot wear shoes', 'foot wear sneaker shoes',
   'carrying Backpack', 'carrying Other', 'carrying messenger bag', 'carrying nothing', 'carrying plastic bags',
   'personal less 30','personal less 45','personal less 60','personal larger 60',
   'personal male'
] # 54
"""
attr_words = [
   'head hat','head muffler','head nothing','head sunglasses','head long hair',
   'upper casual', 'upper formal', 'upper jacket', 'upper logo', 'upper plaid', 
   'upper short sleeve', 'upper thin stripes', 'upper t-shirt','upper other','upper v-neck',
   'lower Casual', 'lower Formal', 'lower Jeans', 'lower Shorts', 'lower Short Skirt','lower Trousers',
   'shoes Leather', 'shoes Sandals', 'shoes other', 'shoes sneaker',
   'attach Backpack', 'attach Other', 'attach messenger bag', 'attach nothing', 'attach plastic bags',
   'age less 30','age 30 45','age 45 60','age over 60',
   'male'
] # 54 

group_order = [10, 18, 19, 30, 15, 7, 9, 11, 14, 21, 26, 29, 32, 33, 34, 6, 8, 12, 25, 27, 31, 13, 23, 24, 28, 4, 5,
               17, 20, 22, 0, 1, 2, 3, 16]


def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

def get_label_embeds(labels):
    model = SentenceTransformer('/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c02/DATA/kongweizhe/PARMamba/VTB-main/checkpoints/all-mpnet-base-v2')
    embeddings = model.encode(labels)
    return embeddings

def make_bad_label(labels):
    groups = [(0,5),(5,15),(15,21),(21,25),(25,30),(30,34),34]
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


def make_trans_label(labels):
    # 转换字典
    conversion_dict = {
        # 头部
        0: 1, 
        1: 2, 
        2: 3, 
        3: 0, 
        4: 4, # 长发
        # 上身
        5: 6, 
        6: 7, 
        7: 8, 
        8: 9, 
        9: 10, 
        10: 11, 
        11: 12, 
        12: 13, 
        13: 14, 
        14: 5, 
        # 下身
        15: 16,
        16: 17,
        17: 18,
        18: 19,
        19: 20,
        20: 15,
        # 鞋子
        21: 22,
        22: 23,
        23: 24,
        24: 21,
        # 手部
        25: 26,
        26: 27,
        27: 28,
        28: 29,
        29: 25,
        # 年龄
        30: 31,
        31: 32,
        32: 33,
        33: 30,
        # 性别
        34: 34
    }

    bad_labels = []
    for label in labels:
        bad_label = copy.deepcopy(label)
        # 执行转换
        for index, target_index in conversion_dict.items():
            if index == target_index:
                bad_label[index] = 1 - bad_label[index]
            elif bad_label[index] == 1:
                bad_label[index] = 0
                bad_label[target_index] = 1
        bad_labels.append(bad_label)
            

    return np.array(bad_labels)

def generate_data_description(save_dir):
    """
    create a dataset description file, which consists of images, labels
    """
    peta_data = loadmat(os.path.join(save_dir, 'PETA.mat'))
    dataset = EasyDict()
    dataset.description = 'peta'
    dataset.reorder = 'group_order'
    dataset.root = os.path.join(save_dir, 'images')
    dataset.image_name = [f'{i + 1:05}.png' for i in range(19000)]
    raw_attr_name = [i[0][0] for i in peta_data['peta'][0][0][1]]
    # (19000, 105)
    raw_label = peta_data['peta'][0][0][0][:, 4:]
    dataset.label = raw_label[:, group_order]
    dataset.badlabel = make_bad_label(dataset.label)
    # (19000, 35)
    dataset.attr_words = np.array(attr_words)
    #dataset.label = raw_label
    dataset.attr_name = [raw_attr_name[i] for i in group_order]
    # dataset.attr_vectors = get_label_embeds(attr_words)
    dataset.attributes=attr_words    
    dataset.label_idx = EasyDict()
    dataset.label_idx.eval = group_order

    dataset.partition = EasyDict()
    dataset.partition.train = []
    dataset.partition.val = []
    dataset.partition.trainval = []
    dataset.partition.test = []

    dataset.weight_train = []
    dataset.weight_trainval = []
    for idx in range(5):
        train = peta_data['peta'][0][0][3][idx][0][0][0][0][:, 0] - 1
        val = peta_data['peta'][0][0][3][idx][0][0][0][1][:, 0] - 1
        test = peta_data['peta'][0][0][3][idx][0][0][0][2][:, 0] - 1
        trainval = np.concatenate((train, val), axis=0)

        dataset.partition.train.append(train)
        dataset.partition.val.append(val)
        dataset.partition.trainval.append(trainval)
        dataset.partition.test.append(test)

        weight_train = np.mean(dataset.label[train], axis=0)
        weight_trainval = np.mean(dataset.label[trainval], axis=0)

        dataset.weight_train.append(weight_train)
        dataset.weight_trainval.append(weight_trainval)

        """
        dataset.pkl 只包含评价属性的文件 35 label
        dataset_all.pkl 包含所有属性的文件 105 label
        """
    with open(os.path.join(save_dir, 'dataset.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    save_dir = '/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c02/DATA/kongweizhe/PromptPAR/dataset/PETA'
    generate_data_description(save_dir)
