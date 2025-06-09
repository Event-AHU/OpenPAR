import os
import numpy as np
import random
import pickle
import json
from easydict import EasyDict
# from scipy.io import loadmat

from sentence_transformers import SentenceTransformer

np.random.seed(0)
random.seed(0)

attr_words = [
    'Male', 'Female',
    'Child', 'Adult', 'Elderly',
    'Fat', 'Normal', 'Thin',
    'LongHair', 'BlackHair', 'Glasses', 'Hat', 'Mask', 'Scarf', 'Headpones',
    'Front', 'Back', 'Side',
    'ShortSleeve', 'LongSleeve',
    'Shirt', 'Coat', 'Cotton-padded coat',
    'Walking', 'Running', 'Standing', 'sitting',
    'Trousers', 'Shorts', 'LongSkirt', 'ShortSkirt',
    'Jeans', 'Casual pants', 'Dress',
    'Casual shoes', 'Other shoes',
    'Backpack', 'SingleShoulderBag', 'HandBag', 'Umbrella', 'Cup', 'CellPhone',
    'Smoking', 'Reading',
    'Happy', 'Sad', 'Anger', 'Hate', 'Fear', 'Surprise'
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

   
    with open(R"C:\Users\24349\Desktop\yolo\reordered_all.json", 'r',
              encoding='utf-8') as f:
        data = json.load(f)

    # 打乱键值对的顺序
    items = list(data.items())  # 转换为 (键, 值) 元组列表
    random.shuffle(items)  # 随机打乱顺序
    random.shuffle(items)

    # 重新转换为字典
    data = dict(items)

    dataset = EasyDict()
    dataset.description = 'EventPAR'
    dataset.root = "/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c01/DATA/datasets/EventPAR"

    train_image_name = [list(data.keys())[i] for i in range(90000)]
    test_image_name = [list(data.keys())[i] for i in range(90000, 100000)]
    dataset.image_name = train_image_name + test_image_name

    train_label = list(data.values())[:90000]
    test_label = list(data.values())[90000:]
    dataset.label = train_label + test_label
    dataset.label = np.array(dataset.label)

    dataset.attr_name = np.array(attr_words)

    dataset.attr_vectors = get_label_embeds(attr_words)

    dataset.partition = EasyDict()
    dataset.partition.train = np.arange(0, 90000) 
    dataset.partition.test = np.arange(90000, 100000)  

    dataset.weight_train = np.mean(dataset.label[dataset.partition.train], axis=0).astype(np.float32)

    with open(os.path.join(save_dir, 'dataset_reorder.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    save_dir = r'/path/to/your/folder'
    generate_data_description(save_dir)