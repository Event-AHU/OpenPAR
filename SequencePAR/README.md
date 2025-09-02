<div align="center">
 


 **Official PyTorch implementation of "SequencePAR: Understanding Pedestrian Attributes via A Sequence Generation Paradigm"**

 ------
 
</div>

> **[SequencePAR: Understanding Pedestrian Attributes via A Sequence Generation Paradigm]()**,
Jiandong Jin, Xiao Wang, Yin Lin, Chenglong Li, Lili Huang, Aihua Zheng, Jin Tang 


 



## News: 

SequencePAR is accepted by the journal **Pattern Recognition 2025**

## Usage
### Requirements
we use a single RTX A100 40G GPU for training and evaluation. 
```
Python 3.9.16
pytorch 1.12.1
torchvision 0.13.1
scipy 1.10.0
Pillow
easydict
torchtext
```
### Dataset Preparation
Download the PETA dataset from [here](http://mmlab.ie.cuhk.edu.hk/projects/PETA.html), PA100k dataset from [here](https://github.com/xh-liu/HydraPlus-Net#pa-100k-dataset) and RAP1 and RAP2 dataset form [here](https://www.rapdataset.com/), and We provide the processed WIDER dataset in [here]() 

Organize them in `your dataset root dir` folder as follows:
```
|-- your dataset root dir/
|   |-- <PETA>/
|       |-- images
|            |-- 00001.png
|            |-- 00002.png
|            |-- ...
|       |-- PETA.mat
|       |-- dataset_zs_run0.pkl
|
|   |-- <PA100k>/
|       |-- data
|            |-- 000001.jpg
|            |-- 000002.jpg
|            |-- ...
|       |-- annotation.mat
|
|   |-- <RAP1>/
|       |-- RAP_datasets
|       |-- RAP_annotation
|            |-- RAP_annotation.mat
|   |-- <RAP2>/
|       |-- RAP_datasets
|       |-- RAP_annotation
|            |-- RAP_annotation.mat
|       |-- dataset_zs_run0.pkl
```

## Data Preparation
 Run dataset/preprocess/peta_pad.py to get the dataset pkl file
 ```python
python dataset/preprocess/peta_pad.py
```
We fill the images in the original dataset as a square with a simple black border fill and store it in Pad_datasets, you can read the original dataset directly and use the fill code we provided in AttrDataset.py.
We provide processing code for the currently available publicly available pedestrian attribute identification dataset
## Training
```python
python train.py PETA
```

## Evaluation
```python
python eval.py PETA --check_point --dir your_dir
```

## Abstract 
Current pedestrian attribute recognition (PAR) algorithms are developed based on multi-label or multi-task learning frameworks, which aim to discriminate the attributes using specific classification heads. However, these discriminative models are easily influenced by imbalanced data or noisy samples. Inspired by the success of generative models, we rethink the pedestrian attribute recognition scheme and believe the generative models may perform better on modeling dependencies and complexity between human attributes. In this paper, we propose a novel sequence generation paradigm for pedestrian attribute recognition, termed SequencePAR. It extracts the pedestrian features using a pre-trained CLIP model and embeds the attribute set into query tokens under the guidance of text prompts. Then, a Transformer decoder is proposed to generate the human attributes by incorporating the visual features and attribute query tokens. The masked multi-head attention layer is introduced into the decoder module to prevent the model from remembering the next attribute while making attribute predictions during training. Extensive experiments on multiple widely used pedestrian attribute recognition datasets fully validated the effectiveness of our proposed SequencePAR. 





## Environment 


## Our Proposed Approach 
<img src="https://github.com/Event-AHU/OpenPAR/blob/main/SequencePAR/SequencePAR_frameworkV2.jpg" width="800">



## Dataset 



## Experimental Results 

