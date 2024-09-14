import os
import numpy as np
import random
import pickle

from easydict import EasyDict
from scipy.io import loadmat

from sentence_transformers import SentenceTransformer
np.random.seed(0)
random.seed(0)
from process_rap1 import generate_sentence
import re
attr_words = [
    'head bald head', 'head long hair', 'head black hair', 'head hat', 'head glasses',
    'upper shirt', 'upper sweater', 'upper vest', 'upper t-shirt', 'upper cotton', 
    'upper jacket', 'upper suit up','upper tight', 'upper short sleeve','upper other',
    'lower long trousers', 'lower skirt', 'lower short skirt', 'lower dress', 'lower jeans', 'lower tight trousers',
    'shoes leather', 'shoes sport', 'shoes boots', 'shoes cloth', 'shoes casual','shoes other',
    'attach backpack', 'attach shoulder bag', 'attach hand bag', 'attach box', 'attach plastic bag', 'attach paper bag',
    'attach hand trunk', 'attach other',
    'age less 16', 'age 17 30', 'age 31 45','age 46 60',
    'female', 
    'body fat', 'body normal', 'body thin',
    'customer', 'employee',
    'action calling', 'action talking', 'action gathering', 'action holding', 'action pushing', 'action pulling',
    'action carry arm', 'action carry hand','action other',
] # 54 
print(attr_words.index('female'))
group_order = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
               36, 37, 38, 39, 40, 41, 42, 43, 44, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 45, 46, 47, 48, 49, 50, 51, 52, 53]#54
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
    data = data['RAP_annotation']
    
    dataset = EasyDict()
    dataset.description = 'rap2'
    breakpoint()
    dataset.reorder = 'group_order'
    dataset.root = os.path.join(save_dir, 'RAP_datasets')
    dataset.image_name = [data['name'][0][0][i][0][0] for i in range(84928)]
    raw_attr_name = [data['attribute'][0][0][i][0][0] for i in range(152)]
    raw_label = data['data'][0][0]
    selected_attr_idx = (data['selected_attribute'][0][0][0] - 1)[group_order].tolist()  # 54
    dataset.label = raw_label[:, selected_attr_idx]  # (n, 119)
    dataset.attributes=attr_words
    dataset.attr_name = [raw_attr_name[i] for i in selected_attr_idx]
    dataset.attr_words = np.array(attr_words)#attr_words #51

    dataset.label_idx = EasyDict()
    dataset.label_idx.eval = group_order  # 54

    dataset.partition = EasyDict()
    dataset.partition.train = []
    dataset.partition.val = []
    dataset.partition.test = []
    dataset.partition.trainval = []
    
    simple_complete_sentences = generate_sentence(dataset.label)
    sentences = {dataset.image_name[i]:simple_complete_sentences[i]  for i in range(len(simple_complete_sentences))}
    dataset.sentences = sentences
    max_len = 0 
    for elem in simple_complete_sentences:
        cur_len=len(elem.split(' ')) + elem.count(',')
        if max_len< cur_len:
            max_len=cur_len 
    print(f'completet_sentences max length {max_len}')
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
    
    
    dataset.weight_train = []
    dataset.weight_trainval = []

    for idx in range(5) :
        train = data['partition_attribute'][0][0][0][idx]['train_index'][0][0][0] - 1
        val = data['partition_attribute'][0][0][0][idx]['val_index'][0][0][0] - 1
        test = data['partition_attribute'][0][0][0][idx]['test_index'][0][0][0] - 1
        trainval = np.concatenate([train, val])
        dataset.partition.train.append(train)
        print(train)
        dataset.partition.val.append(val)
        dataset.partition.test.append(test)
        dataset.partition.trainval.append(trainval)
        # cls_weight
        weight_train = np.mean(dataset.label[train], axis=0)
        weight_trainval = np.mean(dataset.label[trainval], axis=0)
        dataset.weight_train.append(weight_train)
        dataset.weight_trainval.append(weight_trainval)
    with open(os.path.join('./', 'rap2_template.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    save_dir = '/data/jinjiandong/datasets/RAPV2'
    generate_data_description(save_dir)
