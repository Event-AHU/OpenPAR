<div align="center">
      
<img src="https://github.com/Event-AHU/OpenPAR/blob/main/OpenPAR_logo.png" width="600">

**An open-source framework for Pedestrian Attribute Recognition, based on PyTorch**

------

</div>

## :dart: Update Log：
#### :fire: [Mar-31-2024] We have updated the checkpoints for PromptPAR！

## :dart: Supported Datasets

* **Image-based PAR dataset** 
```
PETA, PA100K, RAPv1, RAPv2, WIDER, PETA-ZS, RAP-ZS
```

* **Video-based PAR dataset** 
```
MARS-Attribute dataset, DukeMTMC-VID-Attribute dataset
```


## :dart: Dataset Preparation

* **Download Dataset**

Download the PETA dataset from [here](http://mmlab.ie.cuhk.edu.hk/projects/PETA.html), PA100k dataset from [here](https://github.com/xh-liu/HydraPlus-Net#pa-100k-dataset) and RAP1 and RAP2 dataset form [here](https://www.rapdataset.com/), and We provide the processed WIDER dataset in [here](https://pan.baidu.com/s/1sqC7tr1bsyLl3FGCnYoYrw)[password:MMIC] 

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



* **Process the Dataset**

 Run dataset/preprocess/peta_pad.py to get the dataset pkl file
 ```python
python dataset/preprocess/peta_pad.py
```
We fill the images in the original dataset as a square with a simple black border fill and store it in Pad_datasets, you can read the original dataset directly and use the fill code we provided in AttrDataset.py.
We provide processing code for the currently available publicly available pedestrian attribute identification dataset



## :dart: [VTFPAR++](https://github.com/Event-AHU/OpenPAR/blob/main/VTFPAR%2B%2B/README.md) 
**Spatio-Temporal Side Tuning Pre-trained Foundation Models for Video-based Pedestrian Attribute Recognition**, arXiv:2404.17929 
Xiao Wang, Qian Zhu, Jiandong Jin, Jun Zhu, Futian Wang, Bo Jiang, Yaowei Wang, Yonghong Tian 
[[Paper](https://arxiv.org/abs/2404.17929)] 
[[Code](https://github.com/Event-AHU/OpenPAR/tree/main/VTFPAR%2B%2B)] 

Existing pedestrian attribute recognition (PAR) algorithms are mainly developed based on a static image, however, the performance is unreliable in challenging scenarios, such as heavy occlusion, motion blur, etc. In this work, we propose to understand human attributes using video frames that can fully use temporal information by fine-tuning a pre-trained multi-modal foundation model efficiently. Specifically, we formulate the video-based PAR as a vision-language fusion problem and adopt a pre-trained foundation model CLIP to extract the visual features. More importantly, we propose a novel spatiotemporal side-tuning strategy to achieve parameter-efficient optimization of the pre-trained vision foundation model. To better utilize the semantic information, we take the full attribute list that needs to be recognized as another input and transform the attribute words/phrases into the corresponding sentence via split, expand, and prompt operations. Then, the text encoder of CLIP is utilized for embedding processed attribute descriptions. The averaged visual tokens and text tokens are concatenated and fed into a fusion Transformer for multi-modal interactive learning. The enhanced tokens will be fed into a classification head for pedestrian attribute prediction. Extensive experiments on two large-scale video-based PAR datasets fully validated the effectiveness of our proposed framework.

![PromptPARfigure](https://github.com/Event-AHU/OpenPAR/blob/main/VTFPAR%2B%2B/figures/VTFPAR%2B%2Bframework.jpg)






## :dart: [PromptPAR](https://github.com/Event-AHU/OpenPAR/blob/main/PromptPAR/README.md)

**Pedestrian Attribute Recognition via CLIP based Prompt Vision-Language Fusion**, Xiao Wang, Jiandong Jin, Chenglong Li, Jin Tang, Cheng Zhang, Wei Wang 
[[Paper](https://arxiv.org/abs/2312.10692)]

Existing pedestrian attribute recognition (PAR) algorithms adopt pre-trained CNN (e.g., ResNet) as their backbone network for visual feature learning, which might obtain sub-optimal results due to the insufficient employment of the relations between pedestrian images and attribute labels. In this paper, we formulate PAR as a vision-language fusion problem and fully exploit the relations between pedestrian images and attribute labels. Specifically, the attribute phrases are first expanded into sentences, and then the pre-trained vision-language model CLIP is adopted as our backbone for feature embedding of visual images and attribute descriptions. The contrastive learning objective connects the vision and language modalities well in the CLIP-based feature space, and the Transformer layers used in CLIP can capture the long-range relations between pixels. Then, a multi-modal Transformer is adopted to fuse the dual features effectively and feed-forward network is used to predict attributes. To optimize our network efficiently, we propose the region-aware prompt tuning technique to adjust very few parameters (i.e., only the prompt vectors and classification heads) and fix both the pre-trained VL model and multi-modal Transformer. Our proposed PAR algorithm only adjusts 0.75% learnable parameters compared with the fine-tuning strategy. It also achieves new state-of-the-art performance on both standard and zero-shot settings for PAR, including RAPv1, RAPv2, WIDER, PA100K, and PETA-ZS, RAP-ZS datasets. 

![PromptPARfigure](https://github.com/Event-AHU/OpenPAR/blob/main/PromptPAR/figures/pipeline.jpg)



## :dart: [SequencePAR](https://github.com/Event-AHU/OpenPAR/blob/main/SequencePAR/README.md) 

**SequencePAR: Understanding Pedestrian Attributes via A Sequence Generation Paradigm**, Jiandong Jin, Xiao Wang, Chenglong Li, Lili Huang, Jin Tang, [[Paper](https://arxiv.org/abs/2312.01640)]

Current pedestrian attribute recognition (PAR) algorithms are developed based on multi-label or multi-task learning frameworks, which aim to discriminate the attributes using specific classification heads. However, these discriminative models are easily influenced by imbalanced data or noisy samples. Inspired by the success of generative models, we rethink the pedestrian attribute recognition scheme and believe the generative models may perform better on modeling dependencies and complexity between human attributes. In this paper, we propose a novel sequence generation paradigm for pedestrian attribute recognition, termed SequencePAR. It extracts the pedestrian features using a pre-trained CLIP model and embeds the attribute set into query tokens under the guidance of text prompts. Then, a Transformer decoder is proposed to generate the human attributes by incorporating the visual features and attribute query tokens. The masked multi-head attention layer is introduced into the decoder module to prevent the model from remembering the next attribute while making attribute predictions during training. Extensive experiments on multiple widely used pedestrian attribute recognition datasets fully validated the effectiveness of our proposed SequencePAR.

![SequencePARfigure](https://github.com/Event-AHU/OpenPAR/blob/main/SequencePAR/SequencePAR_frameworkV2.jpg)




## :chart_with_upwards_trend: License
This project is under the MIT license. See [[license](https://github.com/Event-AHU/OpenPAR/blob/main/LICENSE)] for details. 


## :newspaper: Citation 
If you find this work helps your research, please star this GitHub and cite the following papers: 
```bibtex
@article{wang2024VTFPARpp,
  title={Spatio-Temporal Side Tuning Pre-trained Foundation Models for Video-based Pedestrian Attribute Recognition},
  author={Wang, Xiao and Zhu, Qian and Jin, Jiandong and Zhu, Jun and Wang, Futian and Jiang, Bo and Wang, Yaowei and Tian, Yonghong},
  journal={arXiv preprint arXiv:2404.17929},
  year={2024}
}


@misc{jin2023sequencePAR,
      title={SequencePAR: Understanding Pedestrian Attributes via A Sequence Generation Paradigm}, 
      author={Jiandong Jin and Xiao Wang and Chenglong Li and Lili Huang and Jin Tang},
      year={2023},
      eprint={2312.01640},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}


@misc{wang2023promptPAR,
      title={Pedestrian Attribute Recognition via CLIP based Prompt Vision-Language Fusion}, 
      author={Xiao Wang and Jiandong Jin and Chenglong Li and Jin Tang and Cheng Zhang and Wei Wang},
      year={2023},
      eprint={2312.10692},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}


@article{wang2022PARSurvey,
  title={Pedestrian attribute recognition: A survey},
  author={Wang, Xiao and Zheng, Shaofei and Yang, Rui and Zheng, Aihua and Chen, Zhe and Tang, Jin and Luo, Bin},
  journal={Pattern Recognition},
  volume={121},
  pages={108220},
  year={2022},
  publisher={Elsevier}
}
```

If you have any questions about these works, please leave an issue. 
