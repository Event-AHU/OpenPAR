import os
import numpy as np
import random
import pickle

from easydict import EasyDict
from scipy.io import loadmat

from sentence_transformers import SentenceTransformer

np.random.seed(0)
random.seed(0)

attr_words = [
    'female', 'age less 16', 'age 17 30', 'age 31 45',
    'body fat', 'body normal', 'body thin', 'customer', 'clerk',
    'head bald head', 'head long hair', 'head black hair', 'head hat', 'head glasses', 'head muffler',
    'upper shirt', 'upper sweater', 'upper vest', 'upper t-shirt', 'upper cotton', 'upper jacket', 'upper suit up',
    'upper tight', 'upper short sleeve',
    'lower long trousers', 'lower skirt', 'lower short skirt', 'lower dress', 'lower jeans', 'lower tight trousers',
    'shoes leather', 'shoes sport', 'shoes boots', 'shoes cloth', 'shoes casual',
    'attach backpack', 'attach shoulder bag', 'attach hand bag', 'attach box', 'attach plastic bag', 'attach paper bag',
    'attach hand trunk', 'attach other',
    'action calling', 'action talking', 'action gathering', 'action holding', 'action pushing', 'action pulling',
    'action carry arm', 'action carry hand'
]

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

    data = loadmat(os.path.join(save_dir, 'RAP_annotation/RAP_annotation.mat'))

    dataset = EasyDict()
    dataset.description = 'rap'
    dataset.root = os.path.join(save_dir, 'RAP_dataset')
    dataset.image_name = [data['RAP_annotation'][0][0][5][i][0][0] for i in range(41585)]
    raw_attr_name = [data['RAP_annotation'][0][0][3][i][0][0] for i in range(92)]
    # (41585, 92)
    raw_label = data['RAP_annotation'][0][0][1]
    dataset.label = raw_label[:, np.array(range(51))]
    dataset.attr_name = [raw_attr_name[i] for i in range(51)]

    dataset.attr_words = np.array(attr_words)
    dataset.attr_vectors = get_label_embeds(attr_words)

    dataset.partition = EasyDict()
    dataset.partition.trainval = []
    dataset.partition.test = []

    dataset.weight_trainval = []

    for idx in range(5):
        trainval = data['RAP_annotation'][0][0][0][idx][0][0][0][0][0, :] - 1
        test = data['RAP_annotation'][0][0][0][idx][0][0][0][1][0, :] - 1

        dataset.partition.trainval.append(trainval)
        dataset.partition.test.append(test)

        weight_trainval = np.mean(dataset.label[trainval], axis=0).astype(np.float32)
        dataset.weight_trainval.append(weight_trainval)

    with open(os.path.join(save_dir, 'dataset.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    save_dir = '/dataset/RAP/'
    generate_data_description(save_dir)
