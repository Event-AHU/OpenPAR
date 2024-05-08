import os
import numpy as np
import random
import pickle
import csv
from easydict import EasyDict
from scipy.io import loadmat
import glob
from sentence_transformers import SentenceTransformer

np.random.seed(0)
random.seed(0)
attr_words = [
    'top short', #top length 0 
    'bottom skirt', #bottom length 1
    'shoulder bag','backpack',#shoulder bag #backpack 2 3
    'hat', 'hand bag', 'long hair', 'female',# hat/hand bag/hair/gender 4 5 6 7
    'bottom short', #bottom type 8
    'frontal', 'lateral-frontal', 'lateral', 'lateral-back', 'back', 'pose varies',#pose[9:15]
    'walking', 'running','riding', 'staying', 'motion varies',#motion[15:20]
    'top black', 'top purple', 'top green', 'top blue','top gray', 'top white', 'top yellow', 'top red', 'top complex',#top color [20 :29]
    'bottom white','bottom purple', 'bottom black', 'bottom green', 'bottom gray', 'bottom pink', 'bottom yellow','bottom blue', 'bottom brown', 'bottom complex',#bottom color[29:39]
    'young', 'teenager', 'adult', 'old'#age[39:43]
]

def generate_imgs(path,track_name):
    imgs_path={}
    for _,i in enumerate(track_name):
        tracklet_path=path+'/'+str(i)+'/'
        imagenames=os.listdir(tracklet_path)
        for idx in range(len(imagenames)):
            imagenames[idx]= os.path.join(tracklet_path,imagenames[idx])
        imgs_path[i]=imagenames
    return imgs_path

def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

def generate_label(filename):
    with open(filename, "r") as csvfile:
        csv_reader = csv.reader(csvfile)
        header = next(csv_reader) 
        result_dict = {}
        for row in csv_reader:
            row_buf=[int(i) for i in row[1:]]
            result_dict[str(row[0])] = np.array(row_buf)
    return result_dict

def generate_data_description(save_dir):

    dataset = EasyDict()
    dataset.description = 'mars'
    dataset.root=os.path.join(save_dir,'pad_mars_dataset')
    dataset.attr_name=attr_words
    dataset.words=np.array(attr_words)
    result_dict=generate_label("/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/zhuqian/new_encoded2.csv")
    trainval_name=[]
    test_name=[]
    trainval_gt_list=[]
    test_gt_list=[]
    track_name=[]
    track_gt_list=[]
    track_name_file=open("/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/zhuqian/track_name.txt",'r',encoding='utf8').readlines()
    for name in track_name_file :
        curLine=name.strip('\n')
        track_name.append(curLine)
    trainval_name_file=open("/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/zhuqian/base_train.txt",'r',encoding='utf8').readlines()
    for name in trainval_name_file :
        curLine=name.strip('\n')
        trainval_name.append(curLine)
    test_name_file=open("/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/zhuqian/base_test.txt",'r',encoding='utf8').readlines()
    for name in test_name_file :
        curLine=name.strip('\n')
        test_name.append(curLine)

    for gt in track_name:
        curLine=name.strip('\n')
        track_gt_list.append(result_dict[curLine])

    for gt in trainval_name:
        curLine=name.strip('\n')
        trainval_gt_list.append(result_dict[curLine])

    for gt in test_name_file:
        curLine=name.strip('\n')
        test_gt_list.append(result_dict[curLine])
    
    dataset.test_name=test_name#4908
    dataset.trainval_name=trainval_name#11452
    dataset.track_name=dataset.trainval_name+dataset.test_name

    for elem in dataset.track_name:
        if len(elem)>0 :
            pass
        else :
            print("error")
    dataset.trainval_gt_list=trainval_gt_list
    dataset.test_gt_list=test_gt_list
    dataset.track_gt_list=track_gt_list
    dataset.result_dict = result_dict
    dataset.label = np.concatenate((np.array(trainval_gt_list),np.array(test_gt_list)), axis=0)
    assert dataset.label.shape == (8298+8062, 43)

    dataset.partition = EasyDict()
    dataset.attr_name = attr_words
    dataset.partition.test = np.arange(8298, 8298+8062)  
    dataset.partition.trainval = np.arange(0, 8298)  
    
    path1="/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/zhuqian/pad_mars_dataset"
    dataset.track_imgs_path=generate_imgs(path1, track_name)

    with open(os.path.join(save_dir, 'mars_pad_right.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)

if __name__ == "__main__":
    save_dir = '/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/zhuqian'
    generate_data_description(save_dir)
