import os
import numpy as np
import random
import pickle
from easydict import EasyDict
from scipy.io import loadmat
from process_rap1 import generate_sentence
import re
np.random.seed(0)
random.seed(0)

new_attr_words = [
    'head bald head', 'head long hair', 'head black hair', 'head hat', 'head glasses', 'head muffler', #[9:15]
    'upper shirt', 'upper sweater', 'upper vest', 'upper t-shirt', 'upper cotton', 'upper jacket', 'upper suit up',
    'upper tight', 'upper short sleeve', #[15:24]
    'lower long trousers', 'lower skirt', 'lower short skirt', 'lower dress', 'lower jeans', 'lower tight trousers', #[24:30]
    'shoes leather', 'shoes sport', 'shoes boots', 'shoes cloth', 'shoes casual', #[30:35]
    'attach backpack', 'attach shoulder bag', 'attach hand bag', 'attach box', 'attach plastic bag', 'attach paper bag',
    'attach hand trunk', 'attach other', #[35:43]
    'female', 'age less 16', 'age 17 30', 'age 31 45', #[0] [1:4]
    'body fat', 'body normal', 'body thin', 'customer', 'clerk', #[4:7][7:9]
    'action calling', 'action talking', 'action gathering', 'action holding', 'action pushing', 'action pulling',
    'action carry arm', 'action carry hand' #[43:-1]
] # Group1-10
new_attr_indices = [
    9, 10, 11, 12, 13, 14,  # [0-5] head attributes
    15, 16, 17, 18, 19, 20, 21, 22, 23,  # [6-14] upper attributes
    24, 25, 26, 27, 28, 29,  # [15-20] lower attributes
    30, 31, 32, 33, 34,  # [21-25] shoes attributes
    35, 36, 37, 38, 39, 40, 41, 42,  # [26-33] attach attributes
    0, 1, 2, 3,  # [34-37] gender and age attributes
    4, 5, 6, 7, 8,  # [38-42] body and occupation attributes
    43, 44, 45, 46, 47, 48, 49, 50  # [43-50] action attributes
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
    dataset.attr_name = [raw_attr_name[i] for i in range(51)]  # all 和 select 区别
    dataset.label = dataset.label[:, new_attr_indices]
    # Rearrange labels to match new_attr_words order
    
    dataset.attributes = new_attr_words

    dataset.partition = EasyDict()
    dataset.partition.trainval = []
    dataset.partition.test = []
    
    simple_complete_sentences = generate_sentence(dataset.label)
    sentences = {dataset.image_name[i]:simple_complete_sentences[i]  for i in range(len(simple_complete_sentences))}
    dataset.sentences = sentences
    max_len = 0 
    maxsentence=''
    for elem in simple_complete_sentences:
        cur_len=len(elem.split(' ')) + elem.count(',')
        if max_len< cur_len:
            max_len=cur_len 
            maxsentence = elem
    print(f'completet_sentences max length {max_len}')
    print(f'max sentence {maxsentence}')
    # 使用集合来存储唯一的单词和符号
    unique_words_and_symbols = set()

    # 统计句子中的单词和符号
    for sentence in simple_complete_sentences:
        words_and_symbols = re.findall(r'\b\w+\b|[^\w\s]', sentence)
        unique_words_and_symbols.update(words_and_symbols)

    # 将集合转换为列表
    unique_words_and_symbols_list = list(unique_words_and_symbols)
    dataset.max_length = max_len
    dataset.limit_word = unique_words_and_symbols_list
    # 打印唯一的单词和符号列表
    print(unique_words_and_symbols_list)
    
    dataset.weight_trainval = []

    for idx in range(5):
        trainval = data['RAP_annotation'][0][0][0][idx][0][0][0][0][0, :] - 1
        test = data['RAP_annotation'][0][0][0][idx][0][0][0][1][0, :] - 1
        dataset.partition.trainval.append(trainval)
        dataset.partition.test.append(test)
        weight_trainval = np.mean(dataset.label[trainval], axis=0).astype(np.float32)
        dataset.weight_trainval.append(weight_trainval)

    with open(os.path.join('./', 'rap1_template.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)

if __name__ == "__main__":
    save_dir = '/data/jinjiandong/datasets/RAPV1'
    generate_data_description(save_dir)
