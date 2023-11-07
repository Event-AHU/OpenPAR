import os
import numpy as np
import random
import pickle

from easydict import EasyDict
from scipy.io import loadmat

np.random.seed(0)
random.seed(0)

attr_words = [
    'A pedestrian with a bald head', 'A pedestrian with long hair', 'A pedestrian with black hair',
    'A pedestrian wearing a hat', 'A pedestrian wearing glasses',
    'A pedestrian in a shirt', 'A pedestrian in a sweater', 'A pedestrian in a vest',
    'A pedestrian in a t-shirt', 'A pedestrian in cotton clothing', 'A pedestrian in a jacket',
    'A pedestrian dressed in a suit', 'A pedestrian in tight-fitting clothes', 'A pedestrian in a short-sleeved top',
    'A pedestrian with other upper wear', 'A pedestrian in long trousers', 'A pedestrian in a skirt',
    'A pedestrian in a short skirt', 'A pedestrian in a dress', 'A pedestrian in jeans',
    'A pedestrian in tight trousers', 'A pedestrian in leather shoes', 'A pedestrian in sports shoes',
    'A pedestrian wearing boots', 'A pedestrian in cloth shoes', 'A pedestrian in casual shoes',
    'A pedestrian in other footwear', 'A pedestrian carrying a backpack', 'A pedestrian with a shoulder bag',
    'A pedestrian with a handbag', 'A pedestrian with a box', 'A pedestrian with a plastic bag',
    'A pedestrian with a paper bag', 'A pedestrian carrying a hand trunk', 'A pedestrian with other attachments',
    'A pedestrian under the age of 16', 'A pedestrian between the ages of 17 and 30', 'A pedestrian between the ages of 31 and 45',
    'A pedestrian between the ages of 46 and 60', 'A female pedestrian', 
    'A pedestrian with a larger body build', 'A pedestrian with a normal body build', 'A pedestrian with a slender body build',
    'A customer', 'An employee',
    'A pedestrian making a call', 'A pedestrian engaged in conversation', 'A pedestrian gathered with others',
    'A pedestrian holding something', 'A pedestrian pushing something', 'A pedestrian pulling something',
    'A pedestrian carrying something in their arm', 'A pedestrian carrying something in their hand', 'A pedestrian engaged in other actions',
]# 54 
group_order = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
               36, 37, 38, 39, 40, 41, 42, 43, 44, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 45, 46, 47, 48, 49, 50, 51, 52, 53]#54
def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)


def generate_data_description(save_dir):
    """
    create a dataset description file, which consists of images, labels
    """

    data = loadmat(os.path.join(save_dir, 'RAP_annotation/RAP_annotation.mat'))
    data = data['RAP_annotation']
    dataset = EasyDict()
    dataset.description = 'rap2'
    dataset.reorder = 'group_order'
    dataset.root = os.path.join(save_dir, 'Pad_datasets')
    dataset.image_name = [data['name'][0][0][i][0][0] for i in range(84928)]
    raw_attr_name = [data['attribute'][0][0][i][0][0] for i in range(152)]
    raw_label = data['data'][0][0]
    selected_attr_idx = (data['selected_attribute'][0][0][0] - 1)[group_order].tolist()  # 54
    dataset.label = raw_label[:, selected_attr_idx]  # (n, 119)
    
    dataset.attr_name = [raw_attr_name[i] for i in selected_attr_idx]
    dataset.attributes=attr_words
    dataset.label_idx = EasyDict()
    dataset.label_idx.eval = group_order  # 54

    dataset.partition = EasyDict()
    dataset.partition.train = []
    dataset.partition.val = []
    dataset.partition.test = []
    dataset.partition.trainval = []

    dataset.weight_train = []
    dataset.weight_trainval = []

    for idx in range(5) :
        train = data['partition_attribute'][0][0][0][idx]['train_index'][0][0][0] - 1
        val = data['partition_attribute'][0][0][0][idx]['val_index'][0][0][0] - 1
        test = data['partition_attribute'][0][0][0][idx]['test_index'][0][0][0] - 1
        breakpoint()
        trainval = np.concatenate([train, val])
        dataset.partition.train.append(train)
        dataset.partition.val.append(val)
        dataset.partition.test.append(test)
        dataset.partition.trainval.append(trainval)

        # cls_weight
        weight_train = np.mean(dataset.label[train], axis=0)
        weight_trainval = np.mean(dataset.label[trainval], axis=0)
        dataset.weight_train.append(weight_train)
        dataset.weight_trainval.append(weight_trainval)
    with open(os.path.join(save_dir, 'pad.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    save_dir = '/data/jinjiandong/datasets/RAPV2'
    generate_data_description(save_dir)
