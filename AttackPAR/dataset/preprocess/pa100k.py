import os
import copy
os.environ['CUDA_LAUNCH_BLOCKING'] = '3'
import numpy as np
import random
import pickle

from easydict import EasyDict
from scipy.io import loadmat

from sentence_transformers import SentenceTransformer

np.random.seed(0)
random.seed(0)

attr_words = [
    'female',   #[0]
    'age over 60', 'age 18 to 60', 'age less 18',   #[1:4]
    'front', 'side', 'back',    #[4:7]
    'hat', 'glasses',   #[7:9]
    'hand bag', 'shoulder bag', 'backpack', 'hold objects in front',    #[9:13]
    'short sleeve', 'long sleeve', 'upper stride', 'upper logo', 'upper plaid', 'upper splice', #[13:19]
    'lower stripe', 'lower pattern', 'long coat', 'trousers', 'shorts', 'skirt and dress', 'boots'  #[19:25]
]

def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

def make_bad_label(labels):
    groups = [0,(1,4),(4,7),(7,9),(9,13),(13,19),(19,25)]
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

def get_label_embeds(labels):
    model = SentenceTransformer('/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c01/DATA/kongweizhe/PARMamba/VTB-main/checkpoints/all-mpnet-base-v2')
    embeddings = model.encode(labels)
    return embeddings


def generate_data_description(save_dir):
    """
    create a dataset description file, which consists of images, labels
    """
    pa100k_data = loadmat(os.path.join(save_dir, 'annotation.mat'))

    dataset = EasyDict()
    dataset.description = 'pa100k'
    dataset.root = os.path.join(save_dir, 'data')

    train_image_name = [pa100k_data['train_images_name'][i][0][0] for i in range(80000)]
    val_image_name = [pa100k_data['val_images_name'][i][0][0] for i in range(10000)]
    test_image_name = [pa100k_data['test_images_name'][i][0][0] for i in range(10000)]
    dataset.image_name = train_image_name + val_image_name + test_image_name
    dataset.label = np.concatenate((pa100k_data['train_label'], pa100k_data['val_label'], pa100k_data['test_label']), axis=0)
    dataset.badlabel = make_bad_label(dataset.label)
    assert dataset.label.shape == (100000, 26)
    dataset.attr_name = [pa100k_data['attributes'][i][0][0] for i in range(26)]
    print(dataset.attr_name)
    dataset.attributes=attr_words
    dataset.attr_words = np.array(attr_words)
    dataset.attr_vectors = get_label_embeds(attr_words)

    dataset.partition = EasyDict()
    dataset.partition.train = np.arange(0, 80000)  # np.array(range(80000))
    dataset.partition.val = np.arange(80000, 90000)  # np.array(range(80000, 90000))
    dataset.partition.test = np.arange(90000, 100000)  # np.array(range(90000, 100000))
    dataset.partition.trainval = np.arange(0, 90000)  # np.array(range(90000))

    dataset.weight_train = np.mean(dataset.label[dataset.partition.train], axis=0).astype(np.float32)
    dataset.weight_trainval = np.mean(dataset.label[dataset.partition.trainval], axis=0).astype(np.float32)

    with open(os.path.join(save_dir, 'dataset.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    save_dir = '/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c01/DATA/kongweizhe/PromptPAR/dataset/PA100k/'
    generate_data_description(save_dir)
