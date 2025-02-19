import os
import numpy as np
import random
import pickle

from easydict import EasyDict
from scipy.io import loadmat
from sentence_transformers import SentenceTransformer
np.random.seed(0)
random.seed(0)
from process_peta import generate_sentence
import re
# note: ref by annotation.md
attr_words = [
   'head hat','head muffler','head nothing','head sunglasses','head long hair',
   'upper casual', 'upper formal', 'upper jacket', 'upper logo', 'upper plaid', 
   'upper short sleeve', 'upper thin stripes', 'upper t-shirt','upper other','upper v-neck',
   'lower Casual', 'lower Formal', 'lower Jeans', 'lower Shorts', 'lower Short Skirt','lower Trousers',
   'shoes Leather', 'shoes Sandals', 'shoes other', 'shoes sneaker',
   'attach Backpack', 'attach Other', 'attach messenger bag', 'attach nothing', 'attach plastic bags',
   'age less 30','age 30 45','age 45 60','age over 60',
   'male'
] # 54 
def count_unique_rows(array):
    # 将二维数组转换为字符串形式
    str_array = [''.join(map(str, row)) for row in array]
    
    # 使用集合存储不同的行
    unique_rows = set(str_array)
    
    # 计算不同的行数
    num_unique_rows = len(unique_rows)
    
    # 对行进行分类
    categories = {}
    for idx, row in enumerate(str_array):
        if row in categories:
            categories[row].append(idx)
        else:
            categories[row] = [idx]
    
    return num_unique_rows, categories

group_order = [10, 18, 19, 30, 15, 7, 9, 11, 14, 21, 26, 29, 32, 33, 34, 6, 8, 12, 25, 27, 31, 13, 23, 24, 28, 4, 5,
               17, 20, 22, 0, 1, 2, 3, 16]


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
    peta_data = loadmat('/data/jinjiandong/datasets/PETA/PETA.mat')
    dataset = EasyDict()
    dataset.description = 'peta'
    dataset.reorder = 'group_order'
    dataset.root = os.path.join(save_dir, 'images')
    dataset.image_name = [f'{i + 1:05}.png' for i in range(19000)]
    raw_attr_name = [i[0][0] for i in peta_data['peta'][0][0][1]]
    # (19000, 105)
    raw_label = peta_data['peta'][0][0][0][:, 4:]
    dataset.label = raw_label[:, group_order]
    # (19000, 35)
    dataset.attr_words = np.array(attr_words)
    #dataset.label = raw_label
    dataset.attr_name = [raw_attr_name[i] for i in group_order]
    # dataset.attr_vectors = get_label_embeds(attr_words)
    dataset.attributes = attr_words    
    dataset.label_idx = EasyDict()
    dataset.label_idx.eval = group_order

    dataset.partition = EasyDict()
    dataset.partition.train = []
    dataset.partition.val = []
    dataset.partition.trainval = []
    dataset.partition.test = []

    dataset.weight_train = []
    dataset.weight_trainval = []

    # #首先生成一个随机数矩阵 大小为[19000,53]范围是0-3
    # random_matrix = np.random.randint(0, 4, size=(dataset.label.shape[0], dataset.label.shape[1]))
    simple_complete_sentences = generate_sentence(dataset.label)
    
    for i in range(10):
        for lidx,att in enumerate(dataset.label[i]):
            if att:
                print(attr_words[lidx], end=', ')
        print('')
        print(simple_complete_sentences[i])
    sentences = {dataset.image_name[i]:simple_complete_sentences[i]  for i in range(len(simple_complete_sentences))}
    dataset.sentences = sentences
    max_len = 0 
    for elem in simple_complete_sentences:
        cur_len=len(elem.split(' '))
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
    dataset.limit_word=unique_words_and_symbols_list
    # 打印唯一的单词和符号列表
    print(unique_words_and_symbols_list)
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

        """
        dataset.pkl 只包含评价属性的文件 35 label
        dataset_all.pkl 包含所有属性的文件 105 label
        """
    unique_sentences = set()
    train_sentence=[]
    for i in list(dataset.partition.test[0]):
        train_sentence.append(simple_complete_sentences[i])
    print(len(set(train_sentence)))
    with open(os.path.join('./', 'peta_template.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)



if __name__ == "__main__":
    save_dir = '/data/jinjiandong/datasets/PETA_Sentence/'
    generate_data_description(save_dir)
