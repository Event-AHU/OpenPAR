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



def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

def get_label_embeds(labels):
    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings = model.encode(labels)
    return embeddings


def generate_data_description(save_dir):
    """
    create a dataset description file, which consists of images, labels
    """
    peta_data = loadmat(os.path.join(save_dir, 'PETA.mat'))
    dataset = EasyDict()
    dataset.description = 'peta'
    dataset.root = os.path.join(save_dir, 'images')
    dataset.image_name = [f'{i + 1:05}.png' for i in range(19000)]

    raw_attr_name = [i[0][0] for i in peta_data['peta'][0][0][1]]
    # (19000, 105)
    raw_label = peta_data['peta'][0][0][0][:, 4:]

    # (19000, 35)
    dataset.label = raw_label[:, :35]
    dataset.attr_name = raw_attr_name[:35]

    dataset.attributes=attr_words 
    dataset.attr_words = np.array(attr_words)
    dataset.attr_vectors = get_label_embeds(attr_words)
    
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
    with open(os.path.join(save_dir, 'dataset.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    save_dir = './data/PETA/'
    generate_data_description(save_dir)
