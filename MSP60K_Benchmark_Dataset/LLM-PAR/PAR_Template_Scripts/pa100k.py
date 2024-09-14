import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '3'
import numpy as np
import random
import pickle

from easydict import EasyDict
from scipy.io import loadmat

from sentence_transformers import SentenceTransformer
from process_pa100k import generate_sentence
np.random.seed(0)
random.seed(0)
import re
attr_words = [
    'female',
    'age over 60', 'age 18 to 60', 'age less 18',
    'front', 'side', 'back',
    'hat', 'glasses', 
    'hand bag', 'shoulder bag', 'backpack', 'hold objects in front', 
    'short sleeve', 'long sleeve', 'upper stride', 'upper logo', 'upper plaid', 'upper splice',
    'lower stripe', 'lower pattern', 'long coat', 'trousers', 'shorts', 'skirt and dress', 'boots'
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
    pa100k_data = loadmat(os.path.join(save_dir, 'annotation.mat'))

    dataset = EasyDict()
    dataset.description = 'pa100k'
    dataset.root = os.path.join(save_dir, 'data')
    breakpoint()
    train_image_name = [pa100k_data['train_images_name'][i][0][0] for i in range(80000)]
    val_image_name = [pa100k_data['val_images_name'][i][0][0] for i in range(10000)]
    test_image_name = [pa100k_data['test_images_name'][i][0][0] for i in range(10000)]
    dataset.image_name = train_image_name + val_image_name + test_image_name
    dataset.label = np.concatenate((pa100k_data['train_label'], pa100k_data['val_label'], pa100k_data['test_label']), axis=0)
    assert dataset.label.shape == (100000, 26)
    dataset.attr_name = [pa100k_data['attributes'][i][0][0] for i in range(26)]
    print(dataset.attr_name)
    dataset.attributes=attr_words
    dataset.attr_words = np.array(attr_words)
    dataset.attr_vectors = get_label_embeds(attr_words)
    
    simple_complete_sentences = generate_sentence(dataset.label)
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
    
    dataset.partition = EasyDict()
    dataset.partition.train = np.arange(0, 80000)  # np.array(range(80000))
    dataset.partition.val = np.arange(80000, 90000)  # np.array(range(80000, 90000))
    dataset.partition.test = np.arange(90000, 100000)  # np.array(range(90000, 100000))
    dataset.partition.trainval = np.arange(0, 90000)  # np.array(range(90000))
    
    dataset.weight_train = np.mean(dataset.label[dataset.partition.train], axis=0).astype(np.float32)
    dataset.weight_trainval = np.mean(dataset.label[dataset.partition.trainval], axis=0).astype(np.float32)
    dataset.weight_test = np.mean(dataset.label[dataset.partition.test], axis=0).astype(np.float32)
    for i in range(26):
        print(dataset.weight_trainval[i], dataset.weight_test[i])
    with open(os.path.join('./', 'pa100k_template.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    save_dir = '/data/jinjiandong/datasets/PA100k/'
    generate_data_description(save_dir)
