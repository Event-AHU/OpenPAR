from itertools import count
import json
import os
from re import L
from xml.etree.ElementPath import prepare_predicate
import cv2
import numpy
import torch
"""
trainval_name=[]
trainval_name_file=open("./Annotations/trainval_name.txt",'r',encoding='utf8').readlines()

for name in trainval_name_file :
    trainval_name.append(name)
print(len(trainval_name))
"""
attr_name=[]
attr_file=open("/data/jinjiandong/datasets/WIDER/Annotations/attr_name.txt",'r',encoding='utf8')
for attr in attr_file.readlines() :
    curLine=attr.strip('\n')
    attr_name.append(curLine)
print(attr_name)

"""
gt_list=[]
gt_file=open("/data/jinjiandong/datasets/WIDER/Annotations/trainval_gt_label.txt",'r',encoding='utf8').readlines()
for gt in gt_file :
    curLine=gt[1:-2].strip().split(",")
    count=0
    for elem in curLine:
        if int(elem)<=0 :
            curLine[count]=0
        else :
            curLine[count]=1
        count+=1
    gt_list.append(curLine)
    
print(numpy.array(gt_list))
"""