import os
import numpy as np
import random
import pickle

from easydict import EasyDict
from scipy.io import loadmat
np.random.seed(0)
random.seed(0)

# note: ref by annotation.md
attr_words = [
    'A pedestrian wearing a hat', 'A pedestrian wearing a muffler', 'A pedestrian with no headwear', 'A pedestrian wearing sunglasses', 'A pedestrian with long hair',
    'A pedestrian in casual upper wear', 'A pedestrian in formal upper wear', 'A pedestrian in a jacket', 'A pedestrian in upper wear with a logo', 'A pedestrian in plaid upper wear',
    'A pedestrian in a short-sleeved top', 'A pedestrian in upper wear with thin stripes', 'A pedestrian in a t-shirt', 'A pedestrian in other upper wear', 'A pedestrian in upper wear with a V-neck',
    'A pedestrian in casual lower wear', 'A pedestrian in formal lower wear', 'A pedestrian in jeans', 'A pedestrian in shorts', 'A pedestrian in a short skirt', 'A pedestrian in trousers',
    'A pedestrian in leather shoes', 'A pedestrian in sandals', 'A pedestrian in other types of shoes', 'A pedestrian in sneakers',
    'A pedestrian with a backpack', 'A pedestrian with other types of attachments', 'A pedestrian with a messenger bag', 'A pedestrian with no attachments', 'A pedestrian with plastic bags',
    'A pedestrian under the age of 30', 'A pedestrian between the ages of 30 and 45', 'A pedestrian between the ages of 45 and 60', 'A pedestrian over the age of 60',
    'A male pedestrian'
] # 54 

group_order = [10, 18, 19, 30, 15, 7, 9, 11, 14, 21, 26, 29, 32, 33, 34, 6, 8, 12, 25, 27, 31, 13, 23, 24, 28, 4, 5,
               17, 20, 22, 0, 1, 2, 3, 16]


def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

def generate_data_description(save_dir,new_split_path):
    """
    create a dataset description file, which consists of images, labels
    """
    peta_data = loadmat(os.path.join(save_dir, 'PETA.mat'))
    dataset = EasyDict()
    dataset.description = 'peta'
    dataset.reorder = 'group_order'
    dataset.root = os.path.join(save_dir, 'peta/images')
    dataset.image_name = [f'{i + 1:05}.png' for i in range(19000)]
    raw_attr_name = [i[0][0] for i in peta_data['peta'][0][0][1]]
    # (19000, 105)
    raw_label = peta_data['peta'][0][0][0][:, 4:]
    dataset.label = raw_label[:, group_order]
    # (19000, 35)
    #dataset.label = raw_label
    dataset.attr_name = [raw_attr_name[i] for i in group_order]
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
        with open(os.path.join(save_dir, 'peta_dataset_zs_pad.pkl'), 'wb+') as f:
            pickle.dump(dataset, f)


if __name__ == "__main__":
    save_dir = '/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb61/DATA/wushujuan/'
    new_split_path = '/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb61/DATA/wushujuan/peta_dataset_zs_run0.pkl'
    generate_data_description(save_dir, new_split_path)
