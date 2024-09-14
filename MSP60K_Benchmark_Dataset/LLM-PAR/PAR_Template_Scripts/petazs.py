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

def generate_data_description(save_dir, new_split_path):
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
    dataset.attr_vectors = get_label_embeds(attr_words)
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
    sentences = {dataset.image_name[i]:simple_complete_sentences[i]  for i in range(len(simple_complete_sentences))}
    dataset.sentences = sentences
    max_len = 0 
    text_length=[]
    for elem in simple_complete_sentences:
        cur_len=len(elem.split(' '))+elem.count(',')
        text_length.append(cur_len)
        if max_len< cur_len:
            max_len=cur_len 
    import matplotlib.pyplot as plt
    # 绘制柱状图
    plt.figure(figsize=(10, 6))  # 设置图形大小

    # 绘制柱状图，bins 表示条状的数量
    plt.hist(text_length, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Text Length')  # x 轴标签
    plt.ylabel('Frequency')    # y 轴标签
    plt.title('Distribution of Text Length')  # 标题
    plt.grid(True)  # 显示网格
    
    plt.savefig('text_length_distribution.png')
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

    if new_split_path:
        with open(new_split_path, 'rb+') as f:
            new_split = pickle.load(f)

        train = np.array(new_split.partition.train)
        val = np.array(new_split.partition.val)
        test = np.array(new_split.partition.test)
        trainval = np.concatenate((train, val), axis=0)

        dataset.partition.train = train
        dataset.partition.val = val
        dataset.partition.trainval = trainval
        dataset.partition.test = test

        weight_train = np.mean(dataset.label[train], axis=0).astype(np.float32)
        weight_trainval = np.mean(dataset.label[trainval], axis=0).astype(np.float32)

        dataset.weight_train.append(weight_train)
        dataset.weight_trainval.append(weight_trainval)
        """
        dataset.pkl 只包含评价属性的文件 35 label
        dataset_all.pkl 包含所有属性的文件 105 label
        """
    with open(os.path.join('./', 'petazs_template.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)



if __name__ == "__main__":
    save_dir = '/data/jinjiandong/datasets/PAR_Sentences/PETA_Sentence/'
    new_split_path = '/data/jinjiandong/datasets/PETA/dataset_zs_run0.pkl'
    generate_data_description(save_dir, new_split_path)
