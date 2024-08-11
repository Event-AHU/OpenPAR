import os
import numpy as np
import random
import pickle

from easydict import EasyDict
from scipy.io import loadmat
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from mamba_ssm import MambaLMHeadModel
import torch
import json

np.random.seed(0)
random.seed(0)

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
    # model = SentenceTransformer('checkpoints/all-mpnet-base-v2')
    # embeddings = model.encode(labels)
    # print(type(embeddings))
    # print(embeddings.shape)

    tokenizer = AutoTokenizer.from_pretrained("./checkpoints/bert-base-cased")  
    mamba = MambaLMHeadModel.from_pretrained("./checkpoints/mamba-130M/", dtype=torch.float16, device="cuda")
    tokens = tokenizer(labels,return_tensors="pt",padding='max_length',max_length=10)
    input_ids = tokens.input_ids.to(device="cuda")
    embeddings = mamba.backbone(input_ids, inference_params=None)
    embeddings = embeddings.mean(1)
    embeddings = embeddings.detach().cpu().numpy()
    print(embeddings.shape)
    return embeddings


def generate_data_description(save_dir):
    """
    create a dataset description file, which consists of images, labels
    """
    pa100k_data = loadmat(os.path.join(save_dir, 'annotation.mat'))

    dataset = EasyDict()
    dataset.description = 'pa100k'
    dataset.root = os.path.join(save_dir, 'data')

    train_image_name = [pa100k_data['train_images_name'][i][0][0] for i in range(80000)]
    val_image_name = [pa100k_data['val_images_name'][i][0][0] for i in range(10000)]
    test_image_name = [pa100k_data['test_images_name'][i][0][0] for i in range(10000)]
    dataset.image_name = train_image_name + val_image_name + test_image_name

    dataset.label = np.concatenate((pa100k_data['train_label'], pa100k_data['val_label'], pa100k_data['test_label']), axis=0)
    assert dataset.label.shape == (100000, 26)
    dataset.attr_name = [pa100k_data['attributes'][i][0][0] for i in range(26)]

    dataset.attr_words = np.array(attr_words)
    # breakpoint()
    dataset.attr_vectors = get_label_embeds(attr_words)

    dataset.partition = EasyDict()
    dataset.partition.train = np.arange(0, 80000)  # np.array(range(80000))
    dataset.partition.val = np.arange(80000, 90000)  # np.array(range(80000, 90000))
    dataset.partition.test = np.arange(90000, 100000)  # np.array(range(90000, 100000))
    dataset.partition.trainval = np.arange(0, 90000)  # np.array(range(90000))

    dataset.weight_train = np.mean(dataset.label[dataset.partition.train], axis=0).astype(np.float32)
    dataset.weight_trainval = np.mean(dataset.label[dataset.partition.trainval], axis=0).astype(np.float32)

    with open(os.path.join(save_dir, 'dataset.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    save_dir = 'your/path/of/PA100k'
    generate_data_description(save_dir)
