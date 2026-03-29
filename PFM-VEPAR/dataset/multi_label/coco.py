import sys

import torch.utils.data as data
import json
import os
import subprocess
from PIL import Image
import numpy as np
import torch
import pickle
import logging

# from util import *


urls = {'train_img': 'http://images.cocodataset.org/zips/train2014.zip',
        'val_img': 'http://images.cocodataset.org/zips/val2014.zip',
        'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'}

# def download_coco2014(root, phase):
#     # if not os.path.exists(root):
#     #     os.makedirs(root)
#     # tmpdir = os.path.join(root, 'tmp/')
#     # data = os.path.join(root, 'data/')
#     # if not os.path.exists(data):
#     #     os.makedirs(data)
#     # if not os.path.exists(tmpdir):
#     #     os.makedirs(tmpdir)
#     # if phase == 'train':
#     #     filename = 'train2014.zip'
#     # elif phase == 'val':
#     #     filename = 'val2014.zip'
#     # cached_file = os.path.join(tmpdir, filename)
#     # if not os.path.exists(cached_file):
#     #     print('Downloading: "{}" to {}\n'.format(urls[phase + '_img'], cached_file))
#     #     os.chdir(tmpdir)
#     #     subprocess.call('wget ' + urls[phase + '_img'], shell=True)
#     #     os.chdir(root)
#     # # extract file
#     # img_data = os.path.join(data, filename.split('.')[0])
#     # if not os.path.exists(img_data):
#     #     print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=data))
#     #     command = 'unzip {} -d {}'.format(cached_file,data)
#     #     os.system(command)
#     # print('[dataset] Done!')
#
#     # train/val images/annotations
#     # cached_file = os.path.join(tmpdir, 'annotations_trainval2014.zip')
#     # if not os.path.exists(cached_file):
#     #     print('Downloading: "{}" to {}\n'.format(urls['annotations'], cached_file))
#     #     os.chdir(tmpdir)
#     #     subprocess.Popen('wget ' + urls['annotations'], shell=True)
#     #     os.chdir(root)
#     # annotations_data = os.path.join(data, 'annotations')
#     # if not os.path.exists(annotations_data):
#     #     print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=data))
#     #     command = 'unzip {} -d {}'.format(cached_file, data)
#     #     os.system(command)
#     # print('[annotation] Done!')
#
#     anno = os.path.join(data, '{}_anno.json'.format(phase))
#     img_id = {}
#     annotations_id = {}
#     if not os.path.exists(anno):
#         annotations_file = json.load(open(os.path.join(annotations_data, 'instances_{}2014.json'.format(phase))))
#         annotations = annotations_file['annotations']
#         category = annotations_file['categories']
#         category_id = {}
#         for cat in category:
#             category_id[cat['id']] = cat['name']
#         cat2idx = categoty_to_idx(sorted(category_id.values()))
#         images = annotations_file['images']
#         for annotation in annotations:
#             if annotation['image_id'] not in annotations_id:
#                 annotations_id[annotation['image_id']] = set()
#             annotations_id[annotation['image_id']].add(cat2idx[category_id[annotation['category_id']]])
#         for img in images:
#             if img['id'] not in annotations_id:
#                 continue
#             if img['id'] not in img_id:
#                 img_id[img['id']] = {}
#             img_id[img['id']]['file_name'] = img['file_name']
#             img_id[img['id']]['labels'] = list(annotations_id[img['id']])
#         anno_list = []
#         for k, v in img_id.items():
#             anno_list.append(v)
#         json.dump(anno_list, open(anno, 'w'))
#         if not os.path.exists(os.path.join(data, 'category.json')):
#             json.dump(cat2idx, open(os.path.join(data, 'category.json'), 'w'))
#         del img_id
#         del anno_list
#         del images
#         del annotations_id
#         del annotations
#         del category
#         del category_id
#     print('[json] Done!')


def categoty_to_idx(category):
    cat2idx = {}
    for cat in category:
        cat2idx[cat] = len(cat2idx)
    return cat2idx


class COCO14(data.Dataset):

    def __init__(self, cfg, split, transform=None, target_transform=None):

        root_path = './data/COCO14'
        self.img_dir = os.path.join(root_path, f'{split}2014')
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        list_path = os.path.join(root_path, 'ml_anno', f'coco14_{self.split}_anno.pkl')
        anno = pickle.load(open(list_path, 'rb+'))
        self.img_id = anno['image_name']
        self.label = anno['labels']
        self.img_idx = range(len(self.img_id))

        self.cat2idx = json.load(open(os.path.join(root_path, 'ml_anno', 'category.json'), 'r'))

        self.attr_id = list(self.cat2idx.keys())
        self.attr_num = len(self.cat2idx)

        # just for aligning with pedestrian attribute dataset
        self.eval_attr_num = len(self.cat2idx)

    def __len__(self):
        return len(self.img_id)

    def __getitem__(self, index):

        imgname, gt_label, imgidx = self.img_id[index], self.label[index], self.img_idx[index]
        imgpath = os.path.join(self.img_dir, imgname)
        img = Image.open(imgpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        gt_label = gt_label.astype(np.float32)

        if self.target_transform:
            gt_label = gt_label[self.target_transform]

        return img, gt_label, imgname


