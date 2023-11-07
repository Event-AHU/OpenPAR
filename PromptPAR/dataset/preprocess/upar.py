import os
import numpy as np
import random
import pickle
from scipy.io import loadmat
from easydict import EasyDict

np.random.seed(0)
random.seed(0)
neg_attr_words=['Age in not Young', 'Age in not Adult', 'Age in not Old', 
 'Male', 
 'Hair Length in not Short', 'Hair Length in not Long', 'Hair Length in not Bald', 
 'UpperBody Length in not Short', 'UpperBody Color in not Black', 'UpperBody Color in not Blue', 'UpperBody Color in not Brown', 'UpperBody Color in not Green', 'UpperBody Color in not Grey', 'UpperBody Color in not Orange', 
 'UpperBody Color in not Pink', 'UpperBody Color in not Purple', 'UpperBody Color in not Red', 'UpperBody Color in not White', 'UpperBody Color in not Yellow', 'UpperBody Color in not Other', 
 'LowerBody Length in not Short', 'LowerBody Color in not Black', 'LowerBody Color in not Blue', 'LowerBody Color in not Brown', 'LowerBody Color in not Green', 'LowerBody Color in not Grey', 
 'LowerBody Color in not Orange', 'LowerBody Color in not Pink', 'LowerBody Color in not Purple', 'LowerBody Color in not Red', 'LowerBody Color in not White', 'LowerBody Color in not Yellow', 
 'LowerBody Color in not Other', 'LowerBody Type in not Trousers or Shorts', 'LowerBody Type in not Skirt or Dress', 'Accessory in not Backpack', 'Accessory in not Bag', 'Accessory in not Glasses Normal', 
 'Accessory in not Glasses Sun', 'Accessory in not Hat']

 
attr_words=[
 'Age Young', 'Age Adult', 'Age Old', #[0:3]
 'Gender Female', #[3]
 'Hair Length Short', 'Hair Length Long', 'Hair Length Bald', #[4:6]
 'UpperBody Length Short', #[7]
 'UpperBody Color Black', 'UpperBody Color Blue', 'UpperBody Color Brown', 'UpperBody Color Green', 'UpperBody Color Grey', 'UpperBody Color Orange', 
 'UpperBody Color Pink', 'UpperBody Color Purple', 'UpperBody Color Red', 'UpperBody Color White', 'UpperBody Color Yellow', 'UpperBody Color Other', 
 
 'LowerBody Length Short',
 'LowerBody Color Black', 'LowerBody Color Blue', 'LowerBody Color Brown', 'LowerBody Color Green', 'LowerBody Color Grey', 
 'LowerBody Color Orange', 'LowerBody Color Pink', 'LowerBody Color Purple', 'LowerBody Color Red', 'LowerBody Color White', 'LowerBody Color Yellow', 
 'LowerBody Color Other', 
 
 'LowerBody Type Trousers or Shorts', 'LowerBody Type Skirt or Dress', 

 'Accessory Backpack', 'Accessory Bag', 'Accessory Glasses Normal', 
 'Accessory Glasses Sun', 'Accessory Hat']
def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

def generate_data_description(save_dir, pkl_path):
    
    if pkl_path: #若新的分割路径存在  这里指RAP_zs.pkl
        dataset = EasyDict()
        dataset.partition = EasyDict()
        dataset.partition.train = []
        dataset.partition.val = []
        dataset.partition.test = []
        dataset.partition.trainval = []

        dataset.weight_train = []
        dataset.weight_trainval = []
        dataset.description='upar'
        dataset.root ='/data/jinjiandong/datasets'
        with open(pkl_path, 'rb+') as f:#加载pkl
            new_split = pickle.load(f)
        image_name=new_split.image_name
        buf=[]
        for count,name in enumerate(image_name) :
            if 'RAP2' in name :
                name_spilt=name.split('/')
                new_name=os.path.join('RAPV2','Pad_datasets',name_spilt[-1])
                image_name[count]=new_name
            elif 'PETA' in name :
                name_spilt=name.split('/')
                new_name=os.path.join('PETA','Pad_datasets',name_spilt[-1])
                image_name[count]=new_name
            elif 'PA100k' in name :
                name_spilt=name.split('/')
                new_name=os.path.join('PA100k','Pad_datasets',name_spilt[-1])
                image_name[count]=new_name
            elif 'Market1501' in name :
                name_spilt=name.split('/')
                new_name=os.path.join('Market1501','Pad_datasets',name_spilt[-1])
                image_name[count]=new_name
        dataset.label=new_split.label
        dataset.image_name=image_name
        
        dataset.attributes=attr_words   
        dataset.neg_attr_words=neg_attr_words
        dataset.expand_pos_attr_words=None
        dataset.expand_neg_attr_words=None
        
        train = np.array(new_split.partition.train[0])
        val = np.array(new_split.partition.val[0])
        test=[]
        for elem in new_split.partition.test[0] :
            test+=elem
        test = np.array(test)   
        trainval=np.array(new_split.partition.trainval[0])
        print(trainval.shape)

        dataset.partition.train = train
        dataset.partition.val = val
        dataset.partition.trainval = trainval
        dataset.partition.test = test

        weight_train = new_split.weight_train[0].astype(np.float32)
        weight_trainval = new_split.weight_trainval[0].astype(np.float32)

        dataset.weight_train.append(weight_train)
        dataset.weight_trainval.append(weight_trainval)
        with open(os.path.join(save_dir, f'pad.pkl'), 'wb+') as f:
            pickle.dump(dataset, f)


if __name__ == "__main__":
    save_dir = '/data/jinjiandong/datasets/UPAR'
    new_split_path = f'/data/jinjiandong/datasets/UPAR/dataset_all.pkl'
    generate_data_description(save_dir, new_split_path)