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
    'female', 'age less 16', 'age 17 30', 'age 31 45',#[0] [1:4]
    'body fat', 'body normal', 'body thin', 'customer', 'clerk',#[4:7][7:9]
    'head bald head', 'head long hair', 'head black hair', 'head hat', 'head glasses', 'head muffler',#[9:15]
    'upper shirt', 'upper sweater', 'upper vest', 'upper t-shirt', 'upper cotton', 'upper jacket', 'upper suit up',
    'upper tight', 'upper short sleeve',#[15:24]
    'lower long trousers', 'lower skirt', 'lower short skirt', 'lower dress', 'lower jeans', 'lower tight trousers',#[24:30]
    'shoes leather', 'shoes sport', 'shoes boots', 'shoes cloth', 'shoes casual',#[30:35]
    'attach backpack', 'attach shoulder bag', 'attach hand bag', 'attach box', 'attach plastic bag', 'attach paper bag',
    'attach hand trunk', 'attach other',#[35:43]
    'action calling', 'action talking', 'action gathering', 'action holding', 'action pushing', 'action pulling',
    'action carry arm', 'action carry hand'#[43:-1]
]#Group1-10
pos_attr_words = [
    'woman', 'age less 16', 'age 17 30', 'age 31 45',#[0] [1:4]
    'fat', 'normal', 'thin', 'customer', 'clerk',#[4:7][7:9]
    'head bald head', 'head long hair', 'head black hair', 'head hat', 'head glasses', 'head muffler',#[9:15]
    'upper shirt', 'upper sweater', 'upper vest', 'upper t-shirt', 'upper cotton', 'upper jacket', 'upper suit up',
    'upper tight', 'upper short sleeve',#[15:24]
    'lower long trousers', 'lower skirt', 'lower short skirt', 'lower dress', 'lower jeans', 'lower tight trousers',#[24:30]
    'shoes leather', 'shoes sport', 'shoes boots', 'shoes cloth', 'shoes casual',#[30:35]
    'backpack', 'shoulder bag', 'hand bag', 'box', 'plastic bag', 'paper bag',
    'hand trunk', 'carring other',#[35:43]
    'calling', 'talking', 'gathering', 'holding', 'pushing', 'pulling',
    'carry arm', 'carry hand'#[43:-1]
]#Group1-10
neg_attr_words = [
    'man', 'age over 16', 'age less 17 or over 30', 'age less 31 or over 45',#[0] [1:4]
    'not fat', 'not normal', 'not thin', 'not customer', 'not clerk',#[4:7][7:9]
    'head without bald head', 'head without long hair', 'head without black hair', 'head without hat', 'head without glasses', 'head without muffler',#[9:15]
    'upper without shirt', 'upper without sweater', 'upper without vest', 'upper without t-shirt', 'upper without cotton', 'upper without jacket', 'upper without suit up',
    'upper without tight', 'upper without short sleeve',#[15:24]
    'lower without long trousers', 'lower without skirt', 'lower without short skirt', 'lower without dress', 'lower without jeans', 'lower without tight trousers',#[24:30]
    'shoes not leather', 'shoes not sport', 'shoes not boots', 'shoes not cloth', 'shoes not casual',#[30:35]
    'without backpack', 'without shoulder bag', 'without hand bag', 'without box', 'without plastic bag', 'without paper bag',
    'without hand trunk', 'carrying any of a backpack, shoulder bag, handbag, box, plastic bag, paper bag, hand trunk',#[35:43]
    'not calling', 'not talking', 'not gathering', 'not holding', 'not pushing', 'not pulling',
    'not carry arm', 'not carry hand'#[43:-1]
]#Group1-10
"""
postive_attr_words = [
    'A pedestrian whose gender is female', 'A pedestrian whose age is less 16', 'A pedestrian whose age is between 17 and 30', 'A pedestrian whose age is between 31 and 45',
    'A pedestrian whose body is fat', 'A pedestrian whose body is normal', 'A pedestrian whose body is thin', 'A pedestrian whose identity is customer', 'A pedestrian whose identity is clerk',
    'A pedestrian whose head is bald head', 'A pedestrian whose head is long hair', 'A pedestrian whose head is black hair', 'A pedestrian with a hat', 'A pedestrian with a glasses', 'A pedestrian with a muffler',
    'A pedestrian whose upper body is wearing shirt', 'A pedestrian whose upper body is wearing sweater', 'A pedestrian whose upper body is wearing vest', 'A pedestrian whose upper body is wearing t-shirt', 'A pedestrian whose upper body is wearing cotton', 
    'A pedestrian whose upper body is wearing jacket', 'A pedestrian whose upper body is wearing suit up',
    'A pedestrian whose upper body is wearing tight', 'A pedestrian whose upper body is wearing short sleeve',
    'A pedestrian whose lower body is wearing trousers', 'A pedestrian whose lower body is wearing skirt', 'A pedestrian whose lower body is wearing short skirt', 
    'A pedestrian whose lower body is wearing dress', 'A pedestrian whose lower body is wearing jeans', 'A pedestrian whose lower body is wearing tight trousers',
    'A pedestrian whose shoes is leather shoes', 'A pedestrian whose shoes is sport shoes', 'A pedestrian whose shoes is boots shoes', 'A pedestrian whose shoes is cloth shoes', 'A pedestrian whose shoes is casual shoes',
    'A pedestrian carrying a backpack', 'A pedestrian carrying a shoulder bag', 'A pedestrian carrying a handbag', 'A pedestrian carrying a box', 'A pedestrian carrying a plastic bag', 'A pedestrian carrying a paper bag',
    'A pedestrian carrying a hand trunk', 'A pedestrian not carrying a backpack, shoulder bag, handbag, box, plastic bag, paper bag, hand trunk or any of them',
    'A pedestrian who is calling', 'A pedestrian who is talking', 'A pedestrian who is gathering', 'A pedestrian who is holding', 'A pedestrian who is pushing', 'A pedestrian who is pulling',
    'A pedestrian who is carry arm', 'A pedestrian who is carry hand'
]
negtive_attr_words = [
    'A pedestrian whose gender is male', 'A pedestrian whose age is over 16', 'A pedestrian whose age is not between 17 and 30', 'A pedestrian whose age is not between 31 and 45',
    'A pedestrian whose body is not fat', 'A pedestrian whose body is not normal', 'A pedestrian whose body is not thin', 'A pedestrian whose identity is not customer', 'A pedestrian whose identity is not clerk',
    'A pedestrian whose head is not bald head', 'A pedestrian whose head is not long hair', 'A pedestrian whose head is not black hair', 'A pedestrian without a hat', 'A pedestrian without a glasses', 'A pedestrian without a muffler',
    'A pedestrian whose upper body is not wearing shirt', 'A pedestrian whose upper body is not wearing sweater', 'A pedestrian whose upper body is not wearing vest', 'A pedestrian whose upper body is not wearing t-shirt', 'A pedestrian whose upper body is not wearing cotton', 
    'A pedestrian whose upper body is not wearing jacket', 'A pedestrian whose upper body is not wearing suit up',
    'A pedestrian whose upper body is not wearing tight', 'A pedestrian whose upper body is not wearing short sleeve',
    'A pedestrian whose lower body is not wearing trousers', 'A pedestrian whose lower body is not wearing skirt', 'A pedestrian whose lower body is not wearing short skirt', 
    'A pedestrian whose lower body is not wearing dress', 'A pedestrian whose lower body is not wearing jeans', 'A pedestrian whose lower body is not wearing tight trousers',
    'A pedestrian whose shoes is not leather shoes', 'A pedestrian whose shoes is not sport shoes', 'A pedestrian whose shoes is not boots shoes', 'A pedestrian whose shoes is not cloth shoes', 'A pedestrian whose shoes is not casual shoes',
    'A pedestrian not carrying a backpack', 'A pedestrian not carrying a shoulder bag', 'A pedestrian not carrying a handbag', 'A pedestrian not carrying a box', 'A pedestrian not carrying a plastic bag', 'A pedestrian not carrying a paper bag',
    'A pedestrian not carrying a hand trunk', 'A pedestrian carrying any of a backpack, shoulder bag, handbag, box, plastic bag, paper bag, hand trunk',
    'A pedestrian who is not calling', 'A pedestrian who is not talking', 'A pedestrian who is not gathering', 'A pedestrian who is not holding', 'A pedestrian who is not pushing', 'A pedestrian who is not pulling',
    'A pedestrian who is not carry arm', 'A pedestrian who is not carry hand'
]
"""
def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

def get_label_embeds(labels):
    model = SentenceTransformer('all-mpnet-base-v2')# clip-ViT-L-14 all-mpnet-base-v2
    embeddings = model.encode(labels)
    return embeddings


def generate_data_description(save_dir):
    """
    create a dataset description file, which consists of images, labels
    """

    data = loadmat(os.path.join(save_dir, 'RAP_annotation/RAP_annotation.mat'))

    dataset = EasyDict()
    dataset.description = 'rap'
    dataset.root = os.path.join(save_dir, 'RAP_datasets')
    dataset.image_name = [data['RAP_annotation'][0][0][5][i][0][0] for i in range(41585)]
    raw_attr_name = [data['RAP_annotation'][0][0][3][i][0][0] for i in range(92)]
    # (41585, 92)
    raw_label = data['RAP_annotation'][0][0][1]
    dataset.label = raw_label[:, np.array(range(1,7))]
    dataset.attr_name = [raw_attr_name[i] for i in range(1,7)]#all 和 select区别
    dataset.attributes=attr_words[1:7]
    dataset.attr_words = np.array(attr_words[1:7])
    dataset.attr_vectors = get_label_embeds(attr_words[1:7])

    if attr_words and neg_attr_words :
        dataset.postive_attr_vectors=get_label_embeds(pos_attr_words)
        dataset.negtive_attr_vectors=get_label_embeds(neg_attr_words)
        print("Saved positive and negative attributes vectors")

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

    with open(os.path.join(save_dir, 'dataset_clip.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    save_dir = '/data/jinjiandong/datasets/RAPV1'
    generate_data_description(save_dir)