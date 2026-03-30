import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '3'
import numpy as np
import random
import pickle
import json
from easydict import EasyDict

np.random.seed(0)
random.seed(0)
from tqdm import tqdm
attributes=[
    "Female",
    "Child", "Adult", "Elderly",
    "Fat", "Normal", "Thin",
    "Bald", "Long hair", "Black hair", "Hat", "Glasses", "Mask", "Helmet", "Scarf", "Gloves",
    "Front", "Back", "Side",
    "Short sleeves", "Long sleeves", "Shirt", "Jacket", "Suit",  "Vest", "Cotton-padded coat", "Coat", "Graduation gown", "Chef uniform",
    "Trousers", "Shorts", "Jeans", "Long skirt", "Short skirt", "Dress", 
    "Leather shoes", "Casual shoes", "Boots", "Sandals", "Other shoes",
    "Backpack", "Shoulder bag", "Handbag", "Plastic bag", "Paper bag", "Suitcase", "Others", 
    "Making a phone call", "Smoking", "Hands behind back", "Arms crossed",
    "Walking", "Running", "Standing", "Riding a bicycle", "Riding an scooter", "Riding a skateboard"
]

def generate_pkl_random(save_dir, exist_pkl_path=None, only_change_root=False):
    """
    create a dataset description file, which consists of images, labels
    """
    with open(os.path.join(save_dir, f'random_split.json'), 'r') as f:
        split_data = json.load(f)
    if only_change_root and exist_pkl_path is not None:
        exist_pkl = pickle.load(open(exist_pkl_path, 'rb+'))
        dataset = EasyDict()
        dataset = exist_pkl
        dataset.root = os.path.join(save_dir, 'images')
    else:
        dataset = EasyDict()
        dataset.description = 'msp'
        dataset.root = os.path.join(save_dir, 'images')
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
        dataset.attributes = attributes

        dataset.label = np.concatenate((train_labels, val_labels, test_labels), axis=0)
        if exist_pkl_path is not None:
            exist_pkl = pickle.load(open(exist_pkl_path, 'rb+'))
            dataset.attr_vectors = exist_pkl.attr_vectors
        else:
            raise Exception('Must enter pkl!')
        
        dataset.attr_words = np.array(attributes)
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
    with open(os.path.join(save_dir, f'dataset_random.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)

def generate_pkl_MS(save_dir, exist_pkl_path=None, only_change_root=False):
    """
    create a dataset description file, which consists of images, labels
    """

    #pkl文件和json文件存储在一个地址
    with open(os.path.join(save_dir, f'Multi_Sences.json'), 'r') as f:
        ms_split = json.load(f)
    with open(os.path.join(save_dir, f'merged_labels.json'), 'r') as f:
        all_label = json.load(f)  
         
    for split in ms_split['split'].keys():
        print(split)
        if only_change_root and exist_pkl_path is not None:
            exist_pkl = pickle.load(open(exist_pkl_path, 'rb+'))
            dataset = EasyDict()
            dataset = exist_pkl
            dataset.root = os.path.join(save_dir, 'images')
        else:
            dataset = EasyDict()
            dataset.description = f'msp_{split}'
            dataset.root = os.path.join(save_dir, 'images')
            if exist_pkl_path is not None:
                exist_pkl = pickle.load(open(exist_pkl_path, 'rb+'))
                dataset.attr_vectors = exist_pkl.attr_vectors
            else:
                raise Exception('Must enter pkl!')
            dataset.attr_words = np.array(attributes)
            dataset.attributes = attributes
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
            assert dataset.attr_vectors.shape == (len(dataset.attributes), 768)
            assert dataset.label.shape == (len(dataset.image_name), len(dataset.attributes))
            
            dataset.partition = EasyDict()
            dataset.partition.trainval = np.arange(0, len(trainval_image_name))
            dataset.partition.test = np.arange(len(trainval_image_name), len(dataset.image_name))
            dataset.weight_trainval = np.mean(dataset.label[dataset.partition.trainval], axis=0).astype(np.float32)
            dataset.weight_test = np.mean(dataset.label[dataset.partition.test], axis=0).astype(np.float32)
        with open(os.path.join(save_dir, f'dataset_ms_{split}.pkl'), 'wb+') as f:
            pickle.dump(dataset, f)
      
# Example usage:
save_dir = '/data/jinjiandong/dataset/MSP_degrade'
pkl_name = ['dataset_random.pkl', 'dataset_ms_split_1.pkl', 'dataset_ms_split_2.pkl', 'dataset_ms_split_3.pkl']
only_change_root = all(os.path.exists(os.path.join(save_dir, pkl)) for pkl in pkl_name)
print(only_change_root)
exist_pkl_path=os.path.join(save_dir, 'base.pkl')
generate_pkl_MS(save_dir, exist_pkl_path=exist_pkl_path, only_change_root=False)
generate_pkl_random(save_dir, exist_pkl_path=exist_pkl_path, only_change_root=False)

