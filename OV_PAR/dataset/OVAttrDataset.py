import os
import pickle
import json
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as T

def get_dataset_metadata(dataset):
    if dataset == 'MSPOV':
        group_index = {'base': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'novel': [5, 6, 7, 8, 9], 'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}
        group_sizes = {'base': [1, 3, 3, 9, 3, 7, 2, 3, 5, 4], 'novel': [3, 4, 2, 2, 6], 'all': [1, 3, 3, 9, 3, 10, 6, 5, 7, 10]}
        group_descriptions = ['Gender', 'Age', 'Body Shape', 'Head Appearance', 'Facing Direction', 'UpperBody Cloth', 'LowerBody Cloth', 'Shoes', 'Attach Bags', 'Performing Action']
    elif dataset == 'RAPOV2':
        group_index = {'base': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'novel': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        group_sizes = {'base': [7, 3, 7, 4, 4, 5, 6, 8, 8, 8, 5], 'novel': [3, 2, 3, 2, 2, 3, 3, 4, 4, 4, 3], 'all': [10, 5, 10, 6, 6, 8, 9, 12, 12, 12, 8]}
        group_descriptions = ['Identity Traits', 'Head Appearance', 'UpperBody Cloth', 'LowerBody Cloth', 'Shoes Type', 'Attach Bags', 'Performing Action', 'Upper Color', 'Lower Color', 'Shoes Color', 'Occlusion']
    return (group_index, group_sizes, group_descriptions)

def load_group_sentences_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    group_labels = {name: torch.stack([torch.tensor(json.loads(k), dtype=torch.float32) for k in data[name].keys()]) for name in data.keys()}
    group_sentences = {name: [k for k in data[name].values()] for name in data.keys()}
    print(f'✅ Loaded {len(data)} sentence lists from {json_path}')
    return (group_labels, group_sentences)

def get_transform(args):
    height = args.height
    width = args.width
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    normalize = T.Normalize(mean=mean, std=std)
    train_transform = T.Compose([PadToSquare(fill=0), T.Resize((height, width)), T.Pad(10), T.RandomCrop((height, width)), T.RandomHorizontalFlip(), T.ToTensor(), normalize])
    valid_transform = T.Compose([PadToSquare(fill=0), T.Resize((height, width)), T.ToTensor(), normalize])
    return (train_transform, valid_transform)

class PadToSquare:

    def __init__(self, fill=0):
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        if h == w:
            return img
        max_side = max(h, w)
        pad_left = (max_side - w) // 2
        pad_right = max_side - w - pad_left
        pad_top = (max_side - h) // 2
        pad_bottom = max_side - h - pad_top
        return T.functional.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=self.fill)

class OpenVocabAttrDataset(data.Dataset):

    def __init__(self, split, args, transform=None, target_transform=None):
        group_index, group_sizes, group_descriptions = get_dataset_metadata(args.dataset)
        with open(args.dataset_pkl, 'rb') as handle:
            dataset_info = pickle.load(handle)
        self.root = args.image_root
        self.group_index = getattr(dataset_info, 'group_index', group_index)
        self.group_sizes = getattr(dataset_info, 'group_sizes', group_sizes)
        self.group_descriptions = getattr(dataset_info, 'group_descriptions', group_descriptions)
        self.combos_labels, self.combos_sentences = load_group_sentences_from_json(args.semantic_bank_path)
        self.group_names = list(self.combos_sentences)
        self.base_attributes = list(dataset_info.base_attributes)
        self.novel_attributes = list(dataset_info.novel_attributes)
        self.base_attr_num = len(self.base_attributes)
        self.novel_attr_num = len(self.novel_attributes)
        self.base_indices = dataset_info.base_indices
        self.novel_indices = dataset_info.novel_indices
        self.split = split
        partition = dataset_info.partition[split]
        self.image_names = partition.image_name
        self.image_dirs = partition.image_dir
        self.img_idx = partition.img_idx
        self.labels = partition.label
        self.dataset = args.dataset
        self.transform = transform
        self.target_transform = target_transform
        print(f'{args.dataset} {split}: {len(self.img_idx)} images, {self.base_attr_num} base / {self.novel_attr_num} novel attributes')

    def __getitem__(self, index):
        imgname = self.image_names[index]
        gt_label = self.labels[index].astype(np.float32)
        image = Image.open(os.path.join(self.root, imgname))
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            gt_label = self.transform(gt_label)
        return (image, gt_label, imgname)

    def __len__(self):
        return len(self.img_idx)
