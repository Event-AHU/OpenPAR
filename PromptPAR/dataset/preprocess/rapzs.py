import os
import numpy as np
import random
import pickle
from scipy.io import loadmat
from easydict import EasyDict

from sentence_transformers import SentenceTransformer

np.random.seed(0)
random.seed(0)
attr_words = [
    'head bald head', 'head long hair', 'head black hair', 'head hat', 'head glasses',
    'upper shirt', 'upper sweater', 'upper vest', 'upper t-shirt', 'upper cotton', 'upper jacket', 'upper suit up',
    'upper tight', 'upper short sleeve','upper other',
    'lower long trousers', 'lower skirt', 'lower short skirt', 'lower dress', 'lower jeans', 'lower tight trousers',
    'shoes leather', 'shoes sport', 'shoes boots', 'shoes cloth', 'shoes casual','shoes other',
    'attach backpack', 'attach shoulder bag', 'attach hand bag', 'attach box', 'attach plastic bag', 'attach paper bag',
    'attach hand trunk', 'attach other',
    'age less 16', 'age 17 30', 'age 31 45',
    'female', 
    'body fat', 'body normal', 'body thin',
    'customer', 'employee',
    'action calling', 'action talking', 'action gathering', 'action holding', 'action pushing', 'action pulling',
    'action carry arm', 'action carry hand','action other',
] # 53 
group_order = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
               36, 37, 38, 39, 40, 41, 42, 43, 44, 1, 2, 3, 0, 5, 6, 7, 8, 9, 45, 46, 47, 48, 49, 50, 51, 52, 53]


def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

def get_label_embeds(labels):
    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings = model.encode(labels)
    return embeddings

def generate_data_description(save_dir, new_split_path):
    data = loadmat(os.path.join(save_dir, 'RAP_annotation/RAP_annotation.mat'))
    data = data['RAP_annotation']
    dataset = EasyDict()
    dataset.description = 'rap2'
    dataset.reorder = 'group_order'
    dataset.root = os.path.join(save_dir, 'RAP_datasets')
    dataset.image_name = [data['name'][0][0][i][0][0] for i in range(84928)]
    raw_attr_name = [data['attribute'][0][0][i][0][0] for i in range(152)]
    raw_label = data['data'][0][0]
    selected_attr_idx = (data['selected_attribute'][0][0][0] - 1)[group_order].tolist()  # 54
    
    dataset.label = raw_label[:, selected_attr_idx]  # (n, 119)
    dataset.attributes=attr_words
    dataset.attr_name = [raw_attr_name[i] for i in selected_attr_idx]
    print(dataset.attr_name)
    dataset.attr_words = np.array(attr_words)#attr_words #51
    dataset.attr_vectors = get_label_embeds(attr_words)

    dataset.label_idx = EasyDict()
    dataset.label_idx.eval = group_order  # 54

    dataset.partition = EasyDict()
    dataset.partition.train = []
    dataset.partition.val = []
    dataset.partition.test = []
    dataset.partition.trainval = []

    dataset.weight_train = []
    dataset.weight_trainval = []
    
    if new_split_path: #若新的分割路径存在  这里指RAP_zs.pkl

        with open(new_split_path, 'rb+') as f:#加载pkl
            new_split = pickle.load(f)
        
        train = np.array(new_split.partition.train)
        val = np.array(new_split.partition.val)
        test = np.array(new_split.partition.test)
        
        trainval = np.concatenate((train, val), axis=0)
        print(np.concatenate([trainval, test]).shape)
        
        dataset.partition.train = train
        dataset.partition.val = val
        dataset.partition.trainval = trainval
        dataset.partition.test = test

        weight_train = np.mean(dataset.label[train], axis=0).astype(np.float32)
        weight_trainval = np.mean(dataset.label[trainval], axis=0).astype(np.float32)
        print(dataset.label.shape)
        dataset.weight_train.append(weight_train)
        dataset.weight_trainval.append(weight_trainval)
        with open(os.path.join(save_dir, f'dataset_zs.pkl'), 'wb+') as f:
            pickle.dump(dataset, f)


if __name__ == "__main__":
    save_dir = '/data/jinjiandong/datasets/RAPV2'
    new_split_path = f'/data/jinjiandong/datasets/RAPV2/dataset_zs_run0.pkl'
    generate_data_description(save_dir, new_split_path)
