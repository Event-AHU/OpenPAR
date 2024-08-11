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
    'male',
    'long hair', 'sunglasses', 'hat',
    'upper t-shirt', 'upper long sleeve', 'upper formal suits', 
    'lower shorts','lower jeans', 'lower long pants','lower skirt',
    'face mask','logo','plaid'
]

def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

def get_label_embeds(labels):
    model = SentenceTransformer('./checkpoints/all-mpnet-base-v2')
    embeddings = model.encode(labels)
    return embeddings


def generate_data_description(save_dir):
    """
    create a dataset description file, which consists of images, labels
    """
    #pa100k_data = loadmat(os.path.join(save_dir, 'annotation.mat'))

    dataset = EasyDict()
    dataset.description = 'wider'
    dataset.root = os.path.join(save_dir, 'split_image')
    
    trainval_name=[]
    test_name=[]
    trainval_gt_list=[]
    test_gt_list=[]
    attr_name_list=[]
    dataset.attributes=attr_words    
    trainval_name_file=open("your/path/of/Wider/Annotations/trainval_name.txt",'r',encoding='utf8').readlines()
    for name in trainval_name_file :
        curLine=name.strip('\n')
        trainval_name.append(curLine)    
        
    test_name_file=open("your/path/of/Wider/Annotations/test_name.txt",'r',encoding='utf8').readlines()
    for name in test_name_file :
        curLine=name.strip('\n')
        test_name.append(curLine)        
    
    trainval_gt_file=open("your/path/of/Wider/Annotations/trainval_gt_label.txt",'r',encoding='utf8').readlines()
    for gt in trainval_gt_file :
        curLine=gt[1:-2].strip().split(",")
        count=0
        for elem in curLine:
            if int(elem)<=0 :
                curLine[count]=0
            else :
                curLine[count]=1
            count+=1
        trainval_gt_list.append(curLine)
        
    test_gt_file=open("your/path/of/Wider/Annotations/test_gt_label.txt",'r',encoding='utf8').readlines()
    for gt in test_gt_file :
        curLine=gt[1:-2].strip().split(",")
        count=0
        for elem in curLine:
            if int(elem)<=0 :
                curLine[count]=0
            else :
                curLine[count]=1
            count+=1
        test_gt_list.append(curLine)
        
    dataset.image_name = trainval_name + test_name
    attr_file=open("your/path/of/Wider/Annotations/attr_name.txt",'r',encoding='utf8')
    for attr in attr_file.readlines() :
        curLine=attr.strip('\n')
        attr_name_list.append(curLine)

    dataset.label = np.concatenate((np.array(trainval_gt_list),np.array(test_gt_list)), axis=0)
    assert dataset.label.shape == (28330+29161, 14)
    dataset.attr_name = attr_name_list
    
    dataset.attr_words = np.array(attr_words)
    dataset.attr_vectors = get_label_embeds(attr_words)

    dataset.partition = EasyDict()

    dataset.partition.test = np.arange(28330, 28330+29161)  # np.array(range(90000, 100000))
    dataset.partition.trainval = np.arange(0, 28330)  # np.array(range(90000))
    
    dataset.weight_trainval = np.mean(dataset.label[dataset.partition.trainval], axis=0).astype(np.float32)

    with open(os.path.join(save_dir, 'dataset.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    save_dir = 'your/path/of/Wider'
    generate_data_description(save_dir)
