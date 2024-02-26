##PromptPAR

<div align="center">
 
<img src="https://github.com/Event-AHU/OpenPAR/blob/main/PromptPAR/figures/promptpar_logo.png" width="600">

 **Official PyTorch implementation of "Pedestrian Attribute Recognition via CLIP based Vision-Language Fusion with Prompt Tuning"**

 ------
 
</div>

> **[Pedestrian Attribute Recognition via CLIP based Vision-Language Fusion with Prompt Tuning]()**, Xiao Wang, Jiandong Jin, Chenglong Li*, Jin Tang, Cheng Zhang, Wei Wang 


## Usage
### Requirements
we use single RTX A6000 48G GPU for training and evaluation. 
```
Python 3.9.16
pytorch 1.12.1
torchvision 0.13.1
scipy 1.10.0
Pillow
easydict
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
|
|   |-- <WIDER>/
|       |-- split_image
|       |-- Annotations
|            |-- attr_name.txt
|            |-- error_name.txt
|            |-- test_gt_label.txt
|            |-- test_name.txt
|            |-- trainval_gt_label.txt
|            |-- trainval_name.txt
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
python train.py PETA --use_text_prompt --use_div --use_vismask --use_GL --use_mm_former
```
## Test
```python
python test_example.py PETA --checkpoint --dir your_dir/epochxx.pth --use_div --use_vismask --vis_prompt 50 --use_GL --use_textprompt --use_mm_former 
```

## Config
|Parameters |Implication|
|:---------------------|:---------:|
| ag_threshold    | Thresholding in global localized image text aggregation (0,1) |
| use_div    |  Whether or not to use regional splits  |
| use_vismask    |  Whether to use a visual mask  |
| use_GL    |  Whether or not to use global localized image text aggregation  |
| use_textprompt    |  Whether or not to use text prompt   |
| use_mm_former    |  Fusion of features using multimodal Transformer or linear layers  |
| div_num    |  Number of split regions  |
| overlap_row    |  Number of overlapping rows in the split regions   |
| text_prompt    |  Number of text prompts  |
| vis_prompt    |  Number of visual prompts |
| vis_depth    |  Depth of visual prompts [1,24]  |

## Checkpoint Download
Dataset  | Vit-Large | Extracted code
 ---- | -----  | -----
 
 RAP1  | [download]() |
 RAP2 | [download]() |
 PETA  | [download]() |
 PA100k  | [download]() |
## News: 


## Abstract 
Existing pedestrian attribute recognition (PAR) algorithms adopt pre-trained CNN (e.g., ResNet) as their backbone network for visual feature learning, which might obtain sub-optimal results due to the insufficient employment of the relations between pedestrian images and attribute labels. In this paper, we formulate PAR as a vision-language fusion problem and fully exploit the relations between pedestrian images and attribute labels. Specifically, the attribute phrases are first expanded into sentences, and then the pre-trained vision-language model CLIP is adopted as our backbone for feature embedding of visual images and attribute descriptions. The contrastive learning objective connects the vision and language modalities well in the CLIP-based feature space, and the Transformer layers used in CLIP can capture the long-range relations between pixels. Then, a multi-modal Transformer is adopted to fuse the dual features effectively and feed-forward network is used to predict attributes. To optimize our network efficiently, we propose the region-aware prompt tuning technique to adjust very few parameters(i.e., only the prompt vectors and classification heads) and fix both the pre-trained VL model and multi-modal Transformer. Our proposed PAR algorithm only adjusts 0.75\% learnable parameters compared with the fine-tuning strategy. It also achieves new state-of-the-art performance on both standard and zero-shot settings for PAR, including RAPv1, RAPv2, WIDER, PA100K, and PETA-ZS, RAP-ZS datasets. 

<img src="https://github.com/Event-AHU/OpenPAR/blob/main/PromptPAR/figures/frontImage.jpg" width="800">





## Our Proposed Approach 
<img src="https://github.com/Event-AHU/OpenPAR/blob/main/PromptPAR/figures/pipeline.jpg" width="800">




## Experimental Results 

<img src="https://github.com/Event-AHU/OpenPAR/blob/main/PromptPAR/figures/featuremap_vis.png" width="800">

<img src="https://github.com/Event-AHU/OpenPAR/blob/main/PromptPAR/figures/attResults_vis.jpg" width="800">

### Acknowledgments

Our code is extended from the following repositories. We sincerely appreciate for their contributions.

* [CLIP](https://github.com/openai/CLIP)
* [VTB](https://github.com/cxh0519/VTB)
* [Rethink of PAR](https://github.com/valencebond/Rethinking_of_PAR)


