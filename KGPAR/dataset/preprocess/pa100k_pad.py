import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '3'
import numpy as np
import random
import pickle

from easydict import EasyDict
from scipy.io import loadmat

np.random.seed(0)
random.seed(0)


attr_words = [
    'A female pedestrian',
    'A pedestrian over the age of 60', 'A pedestrian between the ages of 18 and 60', 'A pedestrian under the age of 18',
    'A pedestrian seen from the front', 'A pedestrian seen from the side', 'A pedestrian seen from the back',
    'A pedestrian wearing a hat', 'A pedestrian wearing glasses',
    'A pedestrian with a handbag', 'A pedestrian with a shoulder bag', 'A pedestrian with a backpack', 'A pedestrian holding objects in front',
    'A pedestrian in short-sleeved upper wear', 'A pedestrian in long-sleeved upper wear', 'A pedestrian in stride upper wear', 
    'A pedestrian in upper wear with a logo', 'A pedestrian in plaid upper wear', 'A pedestrian in splice upper wear',
    'A pedestrian in striped lower wear', 'A pedestrian in patterned lower wear', 'A pedestrian in a long coat', 
    'A pedestrian in trousers', 'A pedestrian in shorts', 'A pedestrian in skirts and dresses', 'A pedestrian wearing boots'
]

neg_attr_words = [
    'male',
    'age less 60', 'age is not between in 18 to 60', 'age over 18',
    'not front', 'not side', 'not back',
    'head without hat', 'head without glasses', 
    'attach is not hand bag', 'attach is not shoulder bag', 'attach is not backpack', 'without hold objects in front', 
    'upper is not short sleeve', 'upper is not long sleeve', 'upper is not stride', 'upper is not logo', 'upper is not plaid', 'upper is not splice',
    'lower is not stripe', 'lower is not pattern', 'upper is not long coat', 'lower is not trousers', 'lower is not shorts', 'lower is not skirt and dress', 'shoes is not boots'
]

def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
def generate_data_description(save_dir):
    """
    create a dataset description file, which consists of images, labels
    """
    pa100k_data = loadmat(os.path.join(save_dir, 'annotation.mat'))

    dataset = EasyDict()
    dataset.description = 'pa100k'
    dataset.root = os.path.join(save_dir, 'pa100k')

    train_image_name = [pa100k_data['train_images_name'][i][0][0] for i in range(80000)]
    val_image_name = [pa100k_data['val_images_name'][i][0][0] for i in range(10000)]
    test_image_name = [pa100k_data['test_images_name'][i][0][0] for i in range(10000)]
    dataset.image_name = train_image_name + val_image_name + test_image_name

    dataset.label = np.concatenate((pa100k_data['train_label'], pa100k_data['val_label'], pa100k_data['test_label']), axis=0)
    assert dataset.label.shape == (100000, 26)
    dataset.attr_name = [pa100k_data['attributes'][i][0][0] for i in range(26)]
    dataset.attributes=attr_words    
    dataset.attributes=attr_words   
    dataset.neg_attr_words=neg_attr_words
    dataset.expand_pos_attr_words=None
    dataset.expand_neg_attr_words=None

    dataset.partition = EasyDict()
    dataset.partition.train = np.arange(0, 80000)  # np.array(range(80000))
    dataset.partition.val = np.arange(80000, 90000)  # np.array(range(80000, 90000))
    dataset.partition.test = np.arange(90000, 100000)  # np.array(range(90000, 100000))
    dataset.partition.trainval = np.arange(0, 90000)  # np.array(range(90000))

    dataset.weight_train = np.mean(dataset.label[dataset.partition.train], axis=0).astype(np.float32)
    dataset.weight_trainval = np.mean(dataset.label[dataset.partition.trainval], axis=0).astype(np.float32)

    with open(os.path.join(save_dir, 'pad.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    save_dir = '/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb61/DATA/wushujuan/'
    generate_data_description(save_dir)
