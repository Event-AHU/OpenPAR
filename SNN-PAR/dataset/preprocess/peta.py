import os
import numpy as np
import random
import pickle
from transformers import AutoTokenizer
from mamba_ssm import MambaLMHeadModel
import torch
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
] 

group_order = [10, 18, 19, 30, 15, 7, 9, 11, 14, 21, 26, 29, 32, 33, 34, 6, 8, 12, 25, 27, 31, 13, 23, 24, 28, 4, 5,
               17, 20, 22, 0, 1, 2, 3, 16]


def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

def get_label_embeds(labels):
    # model = SentenceTransformer('all-mpnet-base-v2')
    # embeddings = model.encode(labels)

    tokenizer = AutoTokenizer.from_pretrained("./checkpoints/bert-base-cased")  
    mamba = MambaLMHeadModel.from_pretrained("./checkpoints/mamba-130M/", dtype=torch.float16, device="cuda:7")
    tokens = tokenizer(labels,return_tensors="pt",padding='max_length')
    input_ids = tokens.input_ids.to(device="cuda:7")
    embeddings = mamba.backbone(input_ids, inference_params=None)
    embeddings = embeddings.mean(1)
    embeddings = embeddings.detach().cpu().numpy()

    return embeddings

def generate_data_description(save_dir):
    """
    create a dataset description file, which consists of images, labels
    """
    peta_data = loadmat(os.path.join(save_dir, 'PETA.mat'))
    # print(peta_data.shape)
    dataset = EasyDict()
    dataset.description = 'peta'
    dataset.reorder = 'group_order'
    dataset.root = os.path.join(save_dir, 'images')
    dataset.image_name = [f'{i + 1:05}.png' for i in range(19000)]
    raw_attr_name = [i[0][0] for i in peta_data['peta'][0][0][1]]
    # (19000, 105)
    raw_label = peta_data['peta'][0][0][0][:, 4:]
    dataset.label = raw_label[:, group_order]
    # (19000, 35)
    dataset.attr_words = np.array(attr_words)
    #dataset.label = raw_label
    dataset.attr_name = [raw_attr_name[i] for i in group_order]
    dataset.attr_vectors = get_label_embeds(attr_words)
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
        # print(peta_data['peta'][0][0][3][idx])
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
    save_dir = './dataset/PETA/'
    generate_data_description(save_dir)
