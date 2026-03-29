import os
import numpy as np
import random
import pickle
from scipy.io import loadmat
from easydict import EasyDict

np.random.seed(0)
random.seed(0)

group_order = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
               36, 37, 38, 39, 40, 41, 42, 43, 44, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 45, 46, 47, 48, 49, 50, 51, 52, 53]


def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)


def generate_data_description(save_dir, reorder, new_split_path, version):
    data = loadmat(os.path.join(save_dir, 'RAP_annotation/RAP_annotation.mat'))
    data = data['RAP_annotation']
    dataset = EasyDict()
    dataset.description = 'rap2'
    dataset.reorder = 'group_order'
    dataset.root = os.path.join(save_dir, 'RAP_dataset')
    dataset.image_name = [data['name'][0][0][i][0][0] for i in range(84928)]
    raw_attr_name = [data['attribute'][0][0][i][0][0] for i in range(152)]
    raw_label = data['data'][0][0]
    selected_attr_idx = (data['selected_attribute'][0][0][0] - 1)[group_order].tolist()  # 54

    color_attr_idx = list(range(31, 45)) + list(range(53, 67)) + list(range(74, 88))  # 42
    extra_attr_idx = np.setdiff1d(range(152), color_attr_idx + selected_attr_idx).tolist()[:24]
    extra_attr_idx = extra_attr_idx[:15] + extra_attr_idx[16:]

    dataset.label = raw_label[:, selected_attr_idx + color_attr_idx + extra_attr_idx]  # (n, 119)
    dataset.attr_name = [raw_attr_name[i] for i in selected_attr_idx + color_attr_idx + extra_attr_idx]

    dataset.label_idx = EasyDict()
    dataset.label_idx.eval = list(range(54))  # 54
    dataset.label_idx.color = list(range(54, 96))  # not aligned with color label index in label
    dataset.label_idx.extra = list(range(96, 119))  # not aligned with extra label index in label

    if reorder:
        dataset.label_idx.eval = list(range(54))

    dataset.partition = EasyDict()
    dataset.partition.train = []
    dataset.partition.val = []
    dataset.partition.test = []
    dataset.partition.trainval = []

    dataset.weight_train = []
    dataset.weight_trainval = []

    if new_split_path:

        # remove Age46-60
        dataset.label_idx.eval.remove(38)  # 54

        with open(new_split_path, 'rb+') as f:
            new_split = pickle.load(f)

        train = np.array(new_split.train_idx)
        val = np.array(new_split.val_idx)
        test = np.array(new_split.test_idx)
        trainval = np.concatenate((train, val), axis=0)

        print(np.concatenate([trainval, test]).shape)

        dataset.partition.train = train
        dataset.partition.val = val
        dataset.partition.trainval = trainval
        dataset.partition.test = test

        weight_train = np.mean(dataset.label[train], axis=0).astype(np.float32)
        weight_trainval = np.mean(dataset.label[trainval], axis=0).astype(np.float32)

        print(weight_trainval[38])

        dataset.weight_train.append(weight_train)
        dataset.weight_trainval.append(weight_trainval)
        with open(os.path.join(save_dir, f'dataset_zs_run{version}.pkl'), 'wb+') as f:
            pickle.dump(dataset, f)

    else:
        for idx in range(5):
            train = data['partition_attribute'][0][0][0][idx]['train_index'][0][0][0] - 1
            val = data['partition_attribute'][0][0][0][idx]['val_index'][0][0][0] - 1
            test = data['partition_attribute'][0][0][0][idx]['test_index'][0][0][0] - 1
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
        with open(os.path.join(save_dir, 'dataset_all.pkl'), 'wb+') as f:
            pickle.dump(dataset, f)


if __name__ == "__main__":
    save_dir = '/mnt/data1/jiajian/datasets/attribute/RAP2/'
    reorder = True

    for i in range(5):
        new_split_path = f'/mnt/data1/jiajian/code/Rethinking_of_PAR/datasets/jian_split/index_rap2_split_id50_img300_ratio0.03_{i}.pkl'
        generate_data_description(save_dir, reorder, new_split_path, i)
