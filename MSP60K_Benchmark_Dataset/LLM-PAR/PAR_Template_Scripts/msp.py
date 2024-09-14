import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '3'
import numpy as np
import random
import pickle
import json
from easydict import EasyDict
from process_msp import generate_sentence
np.random.seed(0)
random.seed(0)
from tqdm import tqdm
import re
attributes=[
    #0
    "Female",
    # 1,2,3
    "Child", "Adult", "Elderly",
    # 4,5,6
    "Fat", "Normal", "Thin",
    # 7,8,9,10,11,12,13,14,15
    "Bald", "Long hair", "Black hair", "Hat", "Glasses", "Mask", "Helmet", "Scarf", "Gloves",
    # 16,17,18
    "Front", "Back", "Side",
    #19,20,21,22,23,24,25,26,27,28
    "Short sleeves", "Long sleeves", "Shirt", "Jacket", "Suit",  "Vest", "Cotton-padded coat", "Coat", "Graduation gown", "Chef uniform",
    #29,30,31,32,33,34
    "Trousers", "Shorts", "Jeans", "Long skirt", "Short skirt", "Dress", 
    #35,36,37,38,39
    "Leather shoes", "Casual shoes", "Boots", "Sandals", "Other shoes",
    #40,41,42,43,44,45,46
    "Backpack", "Shoulder bag", "Handbag", "Plastic bag", "Paper bag", "Suitcase", "Others", 
    #47,48,49,50,51,52,53,54,55,56
    "Making a phone call", "Smoking", "Hands behind back", "Arms crossed",
    "Walking", "Running", "Standing", "Riding a bicycle", "Riding an scooter", "Riding a skateboard"
]
new_attributs=[
    # 7,8,9,10,11,12,13,14,15
    "Bald", "Long hair", "Black hair", "Hat", "Glasses", "Mask", "Helmet", "Scarf", "Gloves",
    #19,20,21,22,23,24,25,26,27,28
    "Short sleeves", "Long sleeves", "Shirt", "Jacket", "Suit",  "Vest", "Cotton-padded coat", "Coat", "Graduation gown", "Chef uniform",
    #29,30,31,32,33,34
    "Trousers", "Shorts", "Jeans", "Long skirt", "Short skirt", "Dress", 
    #35,36,37,38,39
    "Leather shoes", "Casual shoes", "Boots", "Sandals", "Other shoes",
    #40,41,42,43,44,45,46
    "Backpack", "Shoulder bag", "Handbag", "Plastic bag", "Paper bag", "Suitcase", "Others", 
    #0
    "Female",
    # 1,2,3
    "Child", "Adult", "Elderly",
    # 4,5,6
    "Fat", "Normal", "Thin",
    # 16,17,18
    "Front", "Back", "Side",
    #47,48,49,50,51,52,53,54,55,56
    "Making a phone call", "Smoking", "Hands behind back", "Arms crossed",
    "Walking", "Running", "Standing", "Riding a bicycle", "Riding an scooter", "Riding a skateboard"
]
print(new_attributs.index("Female"))
new_attr_indices = [
    7,8,9,10,11,12,13,14,15,  
    9,20,21,22,23,24,25,26,27,28,  
    29,30,31,32,33,34,  
    35,36,37,38,39,  
    40,41,42,43,44,45,46,  
    0, 1, 2, 3, 
    4, 5, 6, 16,17,18,  
    47,48,49,50,51,52,53,54,55,56  
]
def generate_pkl_random(save_dir, exist_pkl_path=None, only_change_root=False):
    """
    create a dataset description file, which consists of images, labels
    """
    with open(os.path.join('/data/jinjiandong/datasets/MSP_degrade', f'random_split.json'), 'r') as f:
        split_data = json.load(f)
    if only_change_root and exist_pkl_path is not None:
        exist_pkl = pickle.load(open(exist_pkl_path, 'rb+'))
        dataset = EasyDict()
        dataset = exist_pkl
        dataset.root = os.path.join('/data/jinjiandong/datasets/MSP_degrade', 'images')
    else:
        dataset = EasyDict()
        dataset.description = 'msp'
        dataset.root = os.path.join('/data/jinjiandong/datasets/MSP_degrade', 'images')
        train_image_name = [image_name for image_name in split_data['train'].keys()]
        val_image_name = [image_name for image_name in split_data['val'].keys()]
        test_image_name = [image_name for image_name in split_data['test'].keys()]
        
        train_labels = [labels for labels in split_data['train'].values()]
        val_labels = [labels for labels in split_data['val'].values()]
        test_labels = [labels for labels in split_data['test'].values()]
        train_labels = np.array(train_labels)
        val_labels = np.array(val_labels)
        test_labels = np.array(test_labels)
        
        dataset.image_name = train_image_name + val_image_name + test_image_name
        dataset.attributes = new_attributs

        dataset.label = np.concatenate((train_labels, val_labels, test_labels), axis=0)
        simple_complete_sentences = generate_sentence(dataset.label)
        for i in range(10):
            for lidx,att in enumerate(dataset.label[i]):
                if att:
                    print(attributes[lidx], end=', ')
            print('')
            print(simple_complete_sentences[i])
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
        dataset.label = dataset.label[:, new_attr_indices]
        if exist_pkl_path is not None:
            exist_pkl = pickle.load(open(exist_pkl_path, 'rb+'))
            dataset.attr_vectors = exist_pkl.attr_vectors[new_attr_indices]
        else:
            raise Exception('Must enter pkl!')
        
        dataset.attr_words = np.array(new_attributs)
        assert dataset.attr_vectors.shape == (len(dataset.attributes), 768)
        assert dataset.label.shape == (len(dataset.image_name), len(dataset.attributes))
        
        dataset.partition = EasyDict()
        
        dataset.partition.train = np.arange(0, len(train_image_name))
        dataset.partition.val = np.arange(len(train_image_name), len(train_image_name) + len(val_image_name))
        dataset.partition.test = np.arange(len(train_image_name) + len(val_image_name), len(dataset.image_name))
        dataset.partition.trainval = np.arange(0, len(train_image_name) + len(val_image_name))
        
        dataset.weight_train = np.mean(dataset.label[dataset.partition.train], axis=0).astype(np.float32)
        dataset.weight_trainval = np.mean(dataset.label[dataset.partition.trainval], axis=0).astype(np.float32)
        dataset.weight_test = np.mean(dataset.label[dataset.partition.test], axis=0).astype(np.float32)
    with open(os.path.join(save_dir, f'msp_random_template.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)

def generate_pkl_MS(save_dir, exist_pkl_path=None, only_change_root=False):
    """
    create a dataset description file, which consists of images, labels
    """
    with open(os.path.join('/data/jinjiandong/datasets/MSP_degrade', f'Multi_Sences2.json'), 'r') as f:
        ms_split = json.load(f)
    with open(os.path.join('/data/jinjiandong/datasets/MSP_degrade', f'merged_labels.json'), 'r') as f:
        all_label = json.load(f)   
    for split in ms_split['split'].keys():
        print(split)
        if only_change_root and exist_pkl_path is not None:
            exist_pkl = pickle.load(open(exist_pkl_path, 'rb+'))
            dataset = EasyDict()
            dataset = exist_pkl
            dataset.root = os.path.join('/data/jinjiandong/datasets/MSP_degrade', 'images')
        else:
            dataset = EasyDict()
            dataset.description = f'msp_{split}'
            dataset.root = os.path.join('/data/jinjiandong/datasets/MSP_degrade', 'images')
            if exist_pkl_path is not None:
                exist_pkl = pickle.load(open(exist_pkl_path, 'rb+'))
                dataset.attr_vectors = exist_pkl.attr_vectors[new_attr_indices]
            else:
                raise Exception('Must enter pkl!')
            dataset.attr_words = np.array(new_attributs)
            dataset.attributes = new_attributs
            #get image names
            trainval_image_name=[]
            for sence in ms_split['split'][split]['trainval']:
                trainval_image_name = trainval_image_name + ms_split['image_names'][sence]
            test_image_name=[]
            for sence in ms_split['split'][split]['test']:
                test_image_name = test_image_name + ms_split['image_names'][sence]
            print(f'trainval image number:{len(trainval_image_name)}, test image number:{len(test_image_name)}')
            dataset.image_name = trainval_image_name + test_image_name
            #get labels
            trainval_labels=[]
            for imgname in tqdm(trainval_image_name):
                trainval_labels.append(all_label[imgname])
            test_labels=[]
            for imgname in tqdm(test_image_name):
                test_labels.append(all_label[imgname])
            trainval_labels = np.array(trainval_labels)
            test_labels = np.array(test_labels)
            dataset.label = np.concatenate((trainval_labels, test_labels), axis=0)
            
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
            
            dataset.label = dataset.label[:,new_attr_indices]
            assert dataset.attr_vectors.shape == (len(dataset.attributes), 768)
            assert dataset.label.shape == (len(dataset.image_name), len(dataset.attributes))
            
            dataset.partition = EasyDict()
            dataset.partition.trainval = np.arange(0, len(trainval_image_name))
            dataset.partition.test = np.arange(len(trainval_image_name), len(dataset.image_name))
            dataset.weight_trainval = np.mean(dataset.label[dataset.partition.trainval], axis=0).astype(np.float32)
            dataset.weight_test = np.mean(dataset.label[dataset.partition.test], axis=0).astype(np.float32)
        with open(os.path.join(save_dir, f'msp_{split}_template.pkl'), 'wb+') as f:
            pickle.dump(dataset, f)
      
# Example usage:
save_dir = '/data/jinjiandong/datasets/par_temp'
pkl_name = ['dataset_random.pkl', 'dataset_ms_split_1.pkl', 'dataset_ms_split_2.pkl']
only_change_root = all(os.path.exists(os.path.join(save_dir, pkl)) for pkl in pkl_name)
print(only_change_root)
exist_pkl_path=os.path.join('/data/jinjiandong/datasets/MSP_degrade', 'base.pkl')
generate_pkl_random(save_dir, exist_pkl_path=exist_pkl_path, only_change_root=False)
generate_pkl_MS(save_dir, exist_pkl_path=exist_pkl_path, only_change_root=False)


