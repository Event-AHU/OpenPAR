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
tqdm
opencv-python
ftfy
regex
```


### Configuration for Windows System 
```python
conda create -n OpenPAR
conda activate OpenPAR
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r .\requirements.txt
```

* Download and install **CUDA** from [https://developer.nvidia.com/cuda-toolkit-archive](https://developer.nvidia.com/cuda-toolkit-archive)

 



## Dataset Processing (Here, let's take the PETA for an example) 

* Download the PETA dataset directly from the Baidu Drive and place it under ``OpenPAR-main\PromptPAR\dataset"
```python
链接: https://pan.baidu.com/s/1Gv_VUSSb0QCDgD2evLrw7w?pwd=upfs 提取码: upfs
``` 

* Process the PETA dataset using the following script:
```python
  cd .\dataset\preprocess\ && python .\peta_pad.py
```
It will generate a new file ``pad.pkl". 

* The pre-trained weight file is needed:
  [[**jx_vit_base_p16_224-80ecf9dd.pth**](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth)] Download it and place it under ``OpenPAR-main/PromptPAR/" 


## Training Script 
```python
python train.py PETA --use_textprompt --use_div --use_vismask --use_GL --use_mm_former
```


If you meet the following issues

<img src="https://github.com/Event-AHU/OpenPAR/blob/main/PromptPAR/figures/screenshot_001.png" width="800">

please modify the **num_workers** in the **train.py** to a smaller one (default num_workers=8), such as: 
```python
    train_set = MultiModalAttrDataset(args=args, split=args.train_split, transform=train_tsfm) 
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batchsize, 
        shuffle=True,
        num_workers=2, 
        pin_memory=True,  
    )
    valid_set = MultiModalAttrDataset(args=args, split=args.valid_split, transform=valid_tsfm) 
    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
```


If you handle all the issues, you can train the PromptPAR successfully!
<img src="https://github.com/Event-AHU/OpenPAR/blob/main/PromptPAR/figures/screenshot_002.png" width="800">


## Test Script 
```python
python test_example.py PETA --checkpoint --dir your_dir --use_div --use_vismask --vis_prompt 50 --use_GL --use_textprompt --use_mm_former 
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

## Vit-Large Checkpoint Download
Dataset  | BaiduYun | Extracted code| GoogleDrive
|:-------------|:---------:|:---------:|:---------:|
| RAP  | [BaiduYun](https://pan.baidu.com/s/1IgXM3EYjuWPxKylVlQG7iA) | 1oen | [GoogleDrive](https://drive.google.com/drive/folders/1GkpaMjJjRDDRnLABK08uoNsOsKXN-nD5?usp=sharing) 
| PETA  | [BaiduYun](https://pan.baidu.com/s/196CDyMFX5rrMQEcC4kQ00w) | MMIC | [GoogleDrive](https://drive.google.com/drive/folders/1GkpaMjJjRDDRnLABK08uoNsOsKXN-nD5?usp=sharing)
| PA100k  | [BaiduYun](https://pan.baidu.com/s/196CDyMFX5rrMQEcC4kQ00w) | MMIC | [GoogleDrive](https://drive.google.com/drive/folders/1GkpaMjJjRDDRnLABK08uoNsOsKXN-nD5?usp=sharing)
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


