import os
import numpy as np
import random
import pickle

from easydict import EasyDict
from scipy.io import loadmat
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
attr_words_template1 = [
    'A female pedestrian', 'A pedestrian under the age of 16', 'A pedestrian between the ages of 17 and 30', 'A pedestrian between the ages of 31 and 45',
    'A pedestrian with a larger body build', 'A pedestrian with a normal body build', 'A pedestrian with a slender body build',
    'A customer', 'A clerk',
    'A pedestrian with a bald head', 'A pedestrian with long hair', 'A pedestrian with black hair', 'A pedestrian wearing a hat', 'A pedestrian wearing glasses', 'A pedestrian wearing a muffler',
    'A pedestrian in a shirt', 'A pedestrian in a sweater', 'A pedestrian in a vest', 'A pedestrian in a t-shirt', 'A pedestrian in cotton clothing', 'A pedestrian in a jacket', 'A pedestrian in a suit', 'A pedestrian in tight-fitting clothes', 'A pedestrian in a short-sleeved top',
    'A pedestrian in long trousers', 'A pedestrian in a skirt', 'A pedestrian in a short skirt', 'A pedestrian in a dress', 'A pedestrian in jeans', 'A pedestrian in tight trousers',
    'A pedestrian in leather shoes', 'A pedestrian in sports shoes', 'A pedestrian in boots', 'A pedestrian in cloth shoes', 'A pedestrian in casual shoes',
    'A pedestrian carrying a backpack', 'A pedestrian with a shoulder bag', 'A pedestrian with a handbag', 'A pedestrian with a box', 'A pedestrian with a plastic bag', 'A pedestrian with a paper bag', 'A pedestrian carrying a hand trunk', 'A pedestrian with other attachments',
    'A pedestrian making a call', 'A pedestrian engaged in conversation', 'A pedestrian gathered with others', 'A pedestrian holding something', 'A pedestrian pushing something', 'A pedestrian pulling something', 'A pedestrian carrying something in their arm', 'A pedestrian carrying something in their hand'
]
import random

attr_words_template2 = []

for attr in attr_words:
    if 'female' in attr:
        templates = 'A female pedestrian'
    elif 'age' in attr:
        age_range = attr.split(' ')
        if 'less' not in age_range:
            templates = f'A pedestrian between the ages of {age_range[1]} and {age_range[2]}'
        else:
            templates = f'A pedestrian aged {age_range[2]} or younger'
    elif 'body' in attr:
        templates =  f'A pedestrian with a {attr} build'
    else:
        templates = f'A {attr} pedestrian'
        
    attr_words_template2.append(templates)

# 打印示例1的扩充结果
print(attr_words_template2)

attr_words_template3 = []

for attr in attr_words:
    if 'female' in attr:
        template = f'A female pedestrian'
    elif 'age' in attr:
        age_range = attr.split(' ')
        if 'less' in age_range:
            template = f'A pedestrian under the age of {age_range[2]}'
        else:
            template = f'A pedestrian between the ages of {age_range[1]} and {age_range[2]}'

    elif 'body' in attr:
        if 'fat' in attr:
            template = f'A pedestrian with a larger body build'
        elif 'normal' in attr:
            template = f'A pedestrian with a normal body build'
        else:
            template = f'A pedestrian with a slender body build'
    else:
        template = f'A pedestrian with {attr}'

    attr_words_template3.append(template)

# 打印示例2的扩充结果
print(attr_words_template3)

neg_attr_words = [
    'male', 'age over 16', 'age less 17 or age over 30', 'age less 31 or age over 45',
    'body is not fat', 'body is not normal', 'body is not thin', 'is clerk', 'is customer',
    'head is not bald head', 'head is not long hair', 'head is not black hair', 'head without hat', 'head without glasses', 'head without muffler',
    'upper is not shirt', 'upper is not sweater', 'upper is not vest', 'upper is not t-shirt', 'upper is not cotton', 'upper is not jacket', 'upper is not suit up',
    'upper is not tight', 'upper is not short sleeve',
    'lower is not long trousers', 'lower is not skirt', 'lower is not short skirt', 'lower is not dress', 'lower is not jeans', 'lower is not tight trousers',
    'shoes is not leather', 'shoes is not sport', 'shoes is not boots', 'shoes is not cloth', 'shoes is not casual',
    'attach is not backpack', 'attach is not shoulder bag', 'attach is not hand bag', 'attach is not box', 'attach is not plastic bag', 'attach is not paper bag',
    'attach is not hand trunk', 'attach is not other',
    'action is not calling', 'action is not talking', 'action is not gathering', 'action is not holding', 'action is not pushing', 'action is not pulling',
    'action carry is not arm', 'action is not carry hand'
]
expand_pos_attr_words = [
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
expand_neg_attr_words = [
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

    dataset = EasyDict()
    dataset.description = 'rap'
    dataset.root = os.path.join(save_dir, 'Pad_datasets')
    dataset.image_name = [data['RAP_annotation'][0][0][5][i][0][0] for i in range(41585)]
    raw_attr_name = [data['RAP_annotation'][0][0][3][i][0][0] for i in range(92)]
    # (41585, 92)
    raw_label = data['RAP_annotation'][0][0][1]
    dataset.label = raw_label[:, np.array(range(51))]
    dataset.attr_name = [raw_attr_name[i] for i in range(51)]#all 和 select区别
    
    dataset.attributes=attr_words
    dataset.neg_attr_words=neg_attr_words
    dataset.expand_pos_attr_words=expand_pos_attr_words
    dataset.expand_neg_attr_words=expand_neg_attr_words
    
    dataset.attr_words_template1=attr_words_template1
    dataset.attr_words_template2= attr_words_template2
    dataset.attr_words_template3= attr_words_template3
    
    
    dataset.partition = EasyDict()
    dataset.partition.trainval = []
    dataset.partition.test = []

    dataset.weight_trainval = []

    for idx in range(5):
        trainval = data['RAP_annotation'][0][0][0][idx][0][0][0][0][0, :] - 1
        test = data['RAP_annotation'][0][0][0][idx][0][0][0][1][0, :] - 1
        print(f'trainval{len(trainval)},test{len(test)}')
        dataset.partition.trainval.append(trainval)
        dataset.partition.test.append(test)

        weight_trainval = np.mean(dataset.label[trainval], axis=0).astype(np.float32)
        dataset.weight_trainval.append(weight_trainval)

    with open(os.path.join(save_dir, 'expand_pad.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)

if __name__ == "__main__":
    save_dir = '/data/jinjiandong/datasets/RAPV1'
    generate_data_description(save_dir)