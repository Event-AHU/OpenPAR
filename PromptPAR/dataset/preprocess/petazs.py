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
    'A pedestrian wearing a hat', 'A pedestrian wearing a muffler', 'A pedestrian with no headwear', 'A pedestrian wearing sunglasses', 'A pedestrian with long hair',
    'A pedestrian in casual upper wear', 'A pedestrian in formal upper wear', 'A pedestrian in a jacket', 'A pedestrian in upper wear with a logo', 'A pedestrian in plaid upper wear',
    'A pedestrian in a short-sleeved top', 'A pedestrian in upper wear with thin stripes', 'A pedestrian in a t-shirt', 'A pedestrian in other upper wear', 'A pedestrian in upper wear with a V-neck',
    'A pedestrian in casual lower wear', 'A pedestrian in formal lower wear', 'A pedestrian in jeans', 'A pedestrian in shorts', 'A pedestrian in a short skirt', 'A pedestrian in trousers',
    'A pedestrian in leather shoes', 'A pedestrian in sandals', 'A pedestrian in other types of shoes', 'A pedestrian in sneakers',
    'A pedestrian with a backpack', 'A pedestrian with other types of attachments', 'A pedestrian with a messenger bag', 'A pedestrian with no attachments', 'A pedestrian with plastic bags',
    'A pedestrian under the age of 30', 'A pedestrian between the ages of 30 and 45', 'A pedestrian between the ages of 45 and 60', 'A pedestrian over the age of 60',
    'A male pedestrian'
]
 # 54 

group_order = [10, 18, 19, 30, 15, 7, 9, 11, 14, 21, 26, 29, 32, 33, 34, 6, 8, 12, 25, 27, 31, 13, 23, 24, 28, 4, 5,
               17, 20, 22, 0, 1, 2, 3, 16]


def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

def get_label_embeds(labels):
    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings = model.encode(labels)
    return embeddings

def generate_data_description(save_dir,new_split_path):
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

    if new_split_path:

        with open(new_split_path, 'rb+') as f:
            new_split = pickle.load(f)

        train = np.array(new_split.partition.train)
        val = np.array(new_split.partition.val)
        test = np.array(new_split.partition.test)
        trainval = np.concatenate((train, val), axis=0)

        dataset.partition.train = train
        dataset.partition.val = val
        dataset.partition.trainval = trainval
        dataset.partition.test = test

        weight_train = np.mean(dataset.label[train], axis=0).astype(np.float32)
        weight_trainval = np.mean(dataset.label[trainval], axis=0).astype(np.float32)

        dataset.weight_train.append(weight_train)
        dataset.weight_trainval.append(weight_trainval)
        with open(os.path.join(save_dir, 'dataset_zs.pkl'), 'wb+') as f:
            pickle.dump(dataset, f)


if __name__ == "__main__":
    save_dir = '/data/jinjiandong/datasets/PETA'
    new_split_path = '/data/jinjiandong/datasets/PETA/dataset_zs_run0.pkl'
    generate_data_description(save_dir, new_split_path)
