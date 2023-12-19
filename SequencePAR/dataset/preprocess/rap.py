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
    'female', 'age less 16', 'age 17 30', 'age 31 45',
    'body fat', 'body normal', 'body thin', 'customer', 'clerk',
    'head bald head', 'head long hair', 'head black hair', 'head hat', 'head glasses', 'head muffler',
    'upper shirt', 'upper sweater', 'upper vest', 'upper t-shirt', 'upper cotton', 'upper jacket', 'upper suit up',
    'upper tight', 'upper short sleeve',
    'lower long trousers', 'lower skirt', 'lower short skirt', 'lower dress', 'lower jeans', 'lower tight trousers',
    'shoes leather', 'shoes sport', 'shoes boots', 'shoes cloth', 'shoes casual',
    'attach backpack', 'attach shoulder bag', 'attach hand bag', 'attach box', 'attach plastic bag', 'attach paper bag',
    'attach hand trunk', 'attach other',
    'action calling', 'action talking', 'action gathering', 'action holding', 'action pushing', 'action pulling',
    'action carry arm', 'action carry hand'
]
group_order = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
               36, 37, 38, 39, 40, 41, 42, 43, 44, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 45, 46, 47, 48, 49, 50, 51, 52, 53]
def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

def get_label_embeds(labels):
    model = SentenceTransformer('clip-ViT-L-14')
    embeddings = model.encode(labels)
    return embeddings


def generate_data_description(save_dir):
    """
    create a dataset description file, which consists of images, labels
    """

    data = loadmat(os.path.join(save_dir, 'RAP_annotation/RAP_annotation.mat'))#加载.mat文件
    data = data['RAP_annotation']
    dataset = EasyDict()#字典
    dataset.description = 'rap'
    dataset.reorder = 'group_order'
    dataset.root = os.path.join(save_dir, 'RAP_dataset')#存储路径
    dataset.image_name = [data['name'][0][0][i][0][0] for i in range(84928)]#读取图片名
    raw_attr_name = [data['attribute'][0][0][i][0][0] for i in range(152)]#读取标签
    raw_label = data['data'][0][0]
    selected_attr_idx = (data['selected_attribute'][0][0][0] - 1)[group_order].tolist()  # 54 #选择标签ID #这里应该是通过ID选择部分标签
    #这部分应该是对标签分类 分为 color extra eval
    """
    'label_idx': {
                'eval': 
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53], 
                'color': 
                [54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]
                'extra': 
                [96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118]
                 }
    """
    color_attr_idx = list(range(31, 45)) + list(range(53, 67)) + list(range(74, 88))  # 42
    extra_attr_idx = np.setdiff1d(range(152), color_attr_idx + selected_attr_idx).tolist()[:24]
    extra_attr_idx = extra_attr_idx[:15] + extra_attr_idx[16:]

    dataset.label = raw_label[:, selected_attr_idx + color_attr_idx + extra_attr_idx]  # (n, 119)
    dataset.attr_name = [raw_attr_name[i] for i in selected_attr_idx + color_attr_idx + extra_attr_idx]
    dataset.attr_words = np.array(attr_words)
    dataset.attr_vectors = get_label_embeds(attr_words)
    dataset.label_idx = EasyDict()
    dataset.label_idx.eval = list(range(54))  # 54
    dataset.label_idx.color = list(range(54, 96))  # not aligned with color label index in label 与标签中的颜色标签索引不对齐
    dataset.label_idx.extra = list(range(96, 119))  # not aligned with extra label index in label 与标签中的额外标签索引不对齐 

    dataset.partition = EasyDict() #数据集划分？
    dataset.partition.train = []
    dataset.partition.val = []
    dataset.partition.test = []
    dataset.partition.trainval = []

    dataset.weight_train = []
    dataset.weight_trainval = []
   
    for idx in range(5) :
        train = data['partition_attribute'][0][0][0][idx]['train_index'][0][0][0] - 1
        val = data['partition_attribute'][0][0][0][idx]['val_index'][0][0][0] - 1
        test = data['partition_attribute'][0][0][0][idx]['test_index'][0][0][0] - 1
        trainval = np.concatenate([train, val])
        dataset.partition.train.append(train)
        dataset.partition.val.append(val)
        dataset.partition.test.append(test)
        dataset.partition.trainval.append(trainval)
        # cls_weight 输出分类矩阵
        weight_train = np.mean(dataset.label[train], axis=0)#np.mean()是用来计算均值 axis = 0 计算矩阵每一列的平均值，显示结果为行向量
        weight_trainval = np.mean(dataset.label[trainval], axis=0)
        dataset.weight_train.append(weight_train)
        dataset.weight_trainval.append(weight_trainval)
    with open(os.path.join(save_dir, 'dataset.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    save_dir = '/home/wangxiao/workdir/jjd/VTB-main/dataset/RAP'
    generate_data_description(save_dir)
