<div align="center">
      
<img src="https://github.com/Event-AHU/OpenPAR/blob/main/OpenPAR_logo.png" width="600">

**An open-source framework for Pedestrian Attribute Recognition, based on PyTorch**

------

</div>



## :dart: Update Log：

:fire: [May-30-2025] Adversarial Semantic and Label Perturbation Attack for Pedestrian Attribute Recognition is released on 
      [[arXiv](https://arxiv.org/abs/2505.23313)] 

:fire: [Apr-15-2025] A new large-scale bimodal benchmark dataset, EventPAR, and an asymmetric RWKV fusion framework are released. 

:fire: [Dec-10-2024] LLM-PAR is accepted by AAAI-2025!

:fire: [Aug-27-2024] Slides for the talk [[Pedestrian Attribute Recognition in the Big Model Era](https://github.com/wangxiao5791509/Pedestrian-Attribute-Recognition-Paper-List/blob/master/PAR-CSIG-2024.08.27.pdf)] 
<img src="https://github.com/wangxiao5791509/Pedestrian-Attribute-Recognition-Paper-List/blob/master/PARtalk.png" width="400">

:fire: [Aug-23-2024] PromptPAR is accepted by IEEE TCSVT 2024.  

:fire: [Aug-19-2024] A new large-scale benchmark dataset MSP60K and baseline LLM-PAR is released. 

:fire: [Mar-31-2024] We have updated the checkpoints for PromptPAR！



## :dart: Papers:
* **Benchmark** 
      [MSP60K&LLM-PAR](https://arxiv.org/pdf/2408.09720),
      [EventPAR](http://arxiv.org/abs/2504.10018)
  
* **Image-based PAR** 
      [PromptPAR](https://arxiv.org/pdf/2312.10692), 
      [SequencePAR](https://arxiv.org/pdf/2312.01640), 
      [MambaPAR](https://arxiv.org/pdf/2407.10374), 
      [SNN-PAR](http://arxiv.org/abs/2410.07857) 

* **Video-based PAR** 
      [VTFPAR++](https://arxiv.org/pdf/2404.17929)
      


## :dart: Supported Datasets

* **Image-based PAR dataset** 
```
PETA, PA100K, RAPv1, RAPv2, WIDER, PETA-ZS, RAP-ZS, MSP60K
```

* **Video-based PAR dataset** 
```
MARS-Attribute dataset, DukeMTMC-VID-Attribute dataset, EventPAR dataset
```


## :dart: Dataset Preparation

* **Download Dataset**

Download the PETA dataset from [here](http://mmlab.ie.cuhk.edu.hk/projects/PETA.html), PA100k dataset from [here](https://github.com/xh-liu/HydraPlus-Net#pa-100k-dataset) and RAP1 and RAP2 dataset form [here](https://www.rapdataset.com/), and We provide the processed WIDER dataset in [Baiduyun](https://pan.baidu.com/s/1sqC7tr1bsyLl3FGCnYoYrw)[password:MMIC],[GoogleDrive](https://drive.google.com/drive/folders/1f0CH-H5V_Ej-rJu_pRQJJiktdbJLv58M?usp=drive_link) 

Organize them in `your dataset root dir` folder as follows:
```
|-- your dataset root dir/
|   |-- <EventPAR>/
|       |-- Video_1
|            |-- rgb_raw
|                |-- xxx.bmp
|                |-- xxx.bmp
|                |-- ...
|            |-- rgb_degraded
|                |-- xxx.bmp
|                |-- xxx.bmp
|                |-- ...
|            |-- event_frames
|                |-- xxx.bmp
|                |-- xxx.bmp
|                |-- ...
|            |-- event_streams
|                |-- xxx.npz
|                |-- xxx.npz
|                |-- ...
|       |-- ...
|       |-- datset.pkl
| 
|   |-- <MSP60K>/
|       |-- degrade_image
|            |-- xxxx.png
|            |-- xxxx.png
|            |-- ...
|       |-- datset_ms_split1.pkl
|       |-- dataset_random.pkl
|       |-- msp_withMS.py
|       |-- msp_split.py
|
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


## :dart: [Adversarial Attack and Defense for PAR]()  
**[arXiv:2505.23313] Adversarial Semantic and Label Perturbation Attack for Pedestrian Attribute Recognition**, arXiv 2025, 
Weizhe Kong, Xiao Wang*, Ruichong Gao, Chenglong Li*, Yu Zhang, Xing Yang, Yaowei Wang, Jin Tang
[[Paper](https://arxiv.org/abs/2505.23313)]

Pedestrian Attribute Recognition (PAR) is an indispensable task in human-centered research and has made great progress in recent years with the development of deep neural networks. However, the potential vulnerability and anti-interference ability have still not been fully explored. To bridge this gap, this paper proposes the first adversarial attack and defense framework for pedestrian attribute recognition. Specifically, we exploit both global- and patch-level attacks on the pedestrian images, based on the pre-trained CLIP-based PAR framework. It first divides the input pedestrian image into non-overlapping patches and embeds them into feature embeddings using a projection layer. Meanwhile, the attribute set is expanded into sentences using prompts and embedded into attribute features using a pre-trained CLIP text encoder. A multi-modal Transformer is adopted to fuse the obtained vision and text tokens, and a feed-forward network is utilized for attribute recognition. Based on the aforementioned PAR framework, we adopt the adversarial semantic and label-perturbation to generate the adversarial noise, termed ASL-PAR. We also design a semantic offset defense strategy to suppress the influence of adversarial attacks. Extensive experiments conducted on both digital domains (i.e., PETA, PA100K, MSP60K, RAPv2) and physical domains fully validated the effectiveness of our proposed adversarial attack and defense strategies for the pedestrian attribute recognition. The source code of this paper will be released on this https URL.
![AdvPAR](https://github.com/Event-AHU/OpenPAR/blob/main/AttackPAR/framework2.jpg)



## :dart: [PAR using an Event Camera]() 
**[arXiv:2504.10018] RGB-Event based Pedestrian Attribute Recognition: A Benchmark Dataset and An Asymmetric RWKV Fusion Framework**, arXiv 2025, 
Xiao Wang, Haiyang Wang, Shiao Wang, Qiang Chen, Jiandong Jin, Haoyu Song, Bo Jiang, Chenglong Li  
[[Paper](http://arxiv.org/abs/2504.10018)]

Existing pedestrian attribute recognition methods are generally developed based on RGB frame cameras. However, these approaches are constrained by the limitations of RGB cameras, such as sensitivity to lighting conditions and motion blur, which hinder their performance. Furthermore, current attribute recognition primarily focuses on analyzing pedestrians’ external appearance and clothing, lacking an exploration of emotional dimensions. In this paper, we revisit these issues and propose a novel multi-modal RGB-Event attribute recognition task by drawing inspiration from the advantages of event cameras in low-light, high-speed, and low-power consumption. Specifically, we introduce the first large-scale multimodal pedestrian attribute recognition dataset, termed EventPAR, comprising 100K paired RGB-Event samples that cover 50 attributes related to both appearance and six human emotions, diverse scenes, and various seasons. By retraining and evaluating mainstream PAR models on this dataset, we establish a comprehensive benchmark and provide a solid foundation for future research in terms of data and algorithmic baselines. In addition, we propose a novel RWKV-based multi-modal pedestrian attribute recognition framework, featuring an RWKV visual encoder and an asymmetric RWKV fusion module. Extensive experiments are conducted on our proposed dataset as well as two simulated datasets (MARS-Attribute and DukeMTMC-VID-Attribute), achieving state-of-the-art results. 
![EventPAR](https://github.com/Event-AHU/OpenPAR/blob/main/EventPAR_Benchmark/figures/dataset_sample.jpg)
![EventPAR](https://github.com/Event-AHU/OpenPAR/blob/main/EventPAR_Benchmark/figures/framework.jpg)


## :dart: [MSP60K Benchmark Dataset, LLM-PAR]() 
**[arXiv:2408.09720] Pedestrian Attribute Recognition: A New Benchmark Dataset and A Large Language Model Augmented Framework**, arXiv 2024, 
Jiandong Jin, Xiao Wang*, Qian Zhu, Haiyang Wang, Chenglong Li*  
[[Paper](https://arxiv.org/abs/2408.09720)] 

Pedestrian Attribute Recognition (PAR) is one of the indispensable tasks in human-centered research. However, existing datasets neglect different domains (e.g., environments, times, populations, and data sources), only conducting simple random splits, and the performance of these datasets has already approached saturation. In the past five years, no large-scale dataset has been opened to the public. To address this issue, this paper proposes a new large-scale, cross-domain pedestrian attribute recognition dataset to fill the data gap, termed MSP60K. It consists of 60,122 images and 57 attribute annotations across eight scenarios. Synthetic degradation is also conducted to further narrow the gap between the dataset and real-world challenging scenarios. To establish a more rigorous benchmark, we evaluate 17 representative PAR models under both random and cross-domain split protocols on our dataset. Additionally, we propose an innovative Large Language Model (LLM) augmented PAR framework, named LLM-PAR. This framework processes pedestrian images through a Vision Transformer (ViT) backbone to extract features and introduces a multi-embedding query Transformer to learn partial-aware features for attribute classification. Significantly, we enhance this framework with LLM for ensemble learning and visual feature augmentation. Comprehensive experiments across multiple PAR benchmark datasets have thoroughly validated the efficacy of our proposed framework. 

![MSP60K](https://github.com/Event-AHU/OpenPAR/blob/main/MSP60K_Benchmark_Dataset/figures/dataset_comparison.jpg)

![LLM-PAR](https://github.com/Event-AHU/OpenPAR/blob/main/MSP60K_Benchmark_Dataset/figures/LLMPAR_frameworkv2.jpg)




## :dart: [MambaPAR-Empirical-Study](https://github.com/Event-AHU/OpenPAR/blob/main/MambaPAR_Empirical_Study/Readme.MD) 
**[arXiv:2407.10374] An Empirical Study of Mamba-based Pedestrian Attribute Recognition**, arXiv 2024, 
Xiao Wang, Weizhe Kong, Jiandong Jin, Shiao Wang, Ruichong Gao, Qingchuan Ma, Chenglong Li, Jin Tang 
[[Paper](https://arxiv.org/abs/2407.10374)] 

Current strong pedestrian attribute recognition models are developed based on Transformer networks, which are computationally heavy. Recently proposed models with linear complexity (e.g., Mamba) have garnered significant attention and have achieved a good balance between accuracy and computational cost across a variety of visual tasks. Relevant review articles also suggest that while these models can perform well on some pedestrian attribute recognition datasets, they are generally weaker than the corresponding Transformer models. To further tap into the potential of the novel Mamba architecture for PAR tasks, this paper designs and adapts Mamba into two typical PAR frameworks, i.e., the text-image fusion approach and pure vision Mamba multi-label recognition framework. It is found that interacting with attribute tags as additional input does not always lead to an improvement, specifically, Vim can be enhanced, but VMamba cannot. This paper further designs various hybrid Mamba-Transformer variants and conducts thorough experimental validations. These experimental results indicate that simply enhancing Mamba with a Transformer does not always lead to performance improvements but yields better results under certain settings. We hope this empirical study can further inspire research in Mamba for PAR, and even extend into the domain of multi-label recognition, through the design of these network structures and comprehensive experimentation. 

![MambaPAR](https://github.com/Event-AHU/OpenPAR/blob/main/MambaPAR_Empirical_Study/figures/HybridMambaFormer.jpg)





## :dart: [VTFPAR++](https://github.com/Event-AHU/OpenPAR/blob/main/VTFPAR%2B%2B/README.md) 
**[arXiv:2404.17929] Spatio-Temporal Side Tuning Pre-trained Foundation Models for Video-based Pedestrian Attribute Recognition**,  
Xiao Wang, Qian Zhu, Jiandong Jin, Jun Zhu, Futian Wang, Bo Jiang, Yaowei Wang, Yonghong Tian 
[[Paper](https://arxiv.org/abs/2404.17929)] 
[[Code](https://github.com/Event-AHU/OpenPAR/tree/main/VTFPAR%2B%2B)] 

Existing pedestrian attribute recognition (PAR) algorithms are mainly developed based on a static image, however, the performance is unreliable in challenging scenarios, such as heavy occlusion, motion blur, etc. In this work, we propose to understand human attributes using video frames that can fully use temporal information by fine-tuning a pre-trained multi-modal foundation model efficiently. Specifically, we formulate the video-based PAR as a vision-language fusion problem and adopt a pre-trained foundation model CLIP to extract the visual features. More importantly, we propose a novel spatiotemporal side-tuning strategy to achieve parameter-efficient optimization of the pre-trained vision foundation model. To better utilize the semantic information, we take the full attribute list that needs to be recognized as another input and transform the attribute words/phrases into the corresponding sentence via split, expand, and prompt operations. Then, the text encoder of CLIP is utilized for embedding processed attribute descriptions. The averaged visual tokens and text tokens are concatenated and fed into a fusion Transformer for multi-modal interactive learning. The enhanced tokens will be fed into a classification head for pedestrian attribute prediction. Extensive experiments on two large-scale video-based PAR datasets fully validated the effectiveness of our proposed framework.

![PromptPARfigure](https://github.com/Event-AHU/OpenPAR/blob/main/VTFPAR%2B%2B/figures/VTFPAR%2B%2Bframework.jpg)






## :dart: [PromptPAR](https://github.com/Event-AHU/OpenPAR/blob/main/PromptPAR/README.md)

**[IEEE TCSVT 2024, arXiv:2312.10692] Pedestrian Attribute Recognition via CLIP based Prompt Vision-Language Fusion**, Xiao Wang, Jiandong Jin, Chenglong Li, Jin Tang, Cheng Zhang, Wei Wang 
[[Paper](https://arxiv.org/abs/2312.10692)] 

Existing pedestrian attribute recognition (PAR) algorithms adopt pre-trained CNN (e.g., ResNet) as their backbone network for visual feature learning, which might obtain sub-optimal results due to the insufficient employment of the relations between pedestrian images and attribute labels. In this paper, we formulate PAR as a vision-language fusion problem and fully exploit the relations between pedestrian images and attribute labels. Specifically, the attribute phrases are first expanded into sentences, and then the pre-trained vision-language model CLIP is adopted as our backbone for feature embedding of visual images and attribute descriptions. The contrastive learning objective connects the vision and language modalities well in the CLIP-based feature space, and the Transformer layers used in CLIP can capture the long-range relations between pixels. Then, a multi-modal Transformer is adopted to fuse the dual features effectively and feed-forward network is used to predict attributes. To optimize our network efficiently, we propose the region-aware prompt tuning technique to adjust very few parameters (i.e., only the prompt vectors and classification heads) and fix both the pre-trained VL model and multi-modal Transformer. Our proposed PAR algorithm only adjusts 0.75% learnable parameters compared with the fine-tuning strategy. It also achieves new state-of-the-art performance on both standard and zero-shot settings for PAR, including RAPv1, RAPv2, WIDER, PA100K, and PETA-ZS, RAP-ZS datasets. 

![PromptPARfigure](https://github.com/Event-AHU/OpenPAR/blob/main/PromptPAR/figures/pipeline.jpg)

## :dart: [SNN-PAR](https://github.com/Event-AHU/OpenPAR/blob/main/SNN-PAR/README.md)

**[arXiv:2410.07857]SNN-PAR: Energy Efficient Pedestrian Attribute Recognition via Spiking Neural Networks**, Haiyang Wang, Qian Zhu, Mowen She, Yabo Li, Haoyu Song,Minghe Xu, and Xiao Wang*
[[Paper](http://arxiv.org/abs/2410.07857)] 
[[Code](https://github.com/Event-AHU/OpenPAR/tree/main/SNN-PAR)] 

Artificial neural network based Pedestrian Attribute Recognition (PAR) has been widely studied in recent years, despite many progresses, however, the energy consumption is still high. To address this issue, in this paper, we propose a Spiking Neural Network (SNN) based framework for energy-efficient attribute recognition. Specifically, we first adopt a spiking tokenizer module to transform the given pedestrian image into spiking feature representations. Then, the output will be fed into the spiking Transformer backbone networks for energy-efficient feature extraction. We feed the enhanced spiking features into a set of feedforward networks for pedestrian attribute recognition. In addition to the widely used binary cross-entropy loss function, we also exploit knowledge distillation from the artificial neural network to the spiking Transformer network for more accurate attribute recognition. Extensive experiments on three widely used PAR benchmark datasets fully validated the effectiveness of our proposed SNN-PAR framework.
![SNN-PARfigure](https://github.com/Event-AHU/OpenPAR/blob/main/SNN-PAR/figures/SNNPAR_framework.png)


## :dart: [SequencePAR](https://github.com/Event-AHU/OpenPAR/blob/main/SequencePAR/README.md) 

**[arXiv:2312.01640] SequencePAR: Understanding Pedestrian Attributes via A Sequence Generation Paradigm**, Jiandong Jin, Xiao Wang, Chenglong Li, Lili Huang, Jin Tang, [[Paper](https://arxiv.org/abs/2312.01640)]

Current pedestrian attribute recognition (PAR) algorithms are developed based on multi-label or multi-task learning frameworks, which aim to discriminate the attributes using specific classification heads. However, these discriminative models are easily influenced by imbalanced data or noisy samples. Inspired by the success of generative models, we rethink the pedestrian attribute recognition scheme and believe the generative models may perform better on modeling dependencies and complexity between human attributes. In this paper, we propose a novel sequence generation paradigm for pedestrian attribute recognition, termed SequencePAR. It extracts the pedestrian features using a pre-trained CLIP model and embeds the attribute set into query tokens under the guidance of text prompts. Then, a Transformer decoder is proposed to generate the human attributes by incorporating the visual features and attribute query tokens. The masked multi-head attention layer is introduced into the decoder module to prevent the model from remembering the next attribute while making attribute predictions during training. Extensive experiments on multiple widely used pedestrian attribute recognition datasets fully validated the effectiveness of our proposed SequencePAR.

![SequencePARfigure](https://github.com/Event-AHU/OpenPAR/blob/main/SequencePAR/SequencePAR_frameworkV2.jpg)




## :chart_with_upwards_trend: License
This project is under the MIT license. See [[license](https://github.com/Event-AHU/OpenPAR/blob/main/LICENSE)] for details. 


## :newspaper: Citation 
If you find this work helps your research, please star this GitHub and cite the following papers: 
```bibtex

@misc{kong2025AdvPAR,
      title={Adversarial Semantic and Label Perturbation Attack for Pedestrian Attribute Recognition}, 
      author={Weizhe Kong and Xiao Wang and Ruichong Gao and Chenglong Li and Yu Zhang and Xing Yang and Yaowei Wang and Jin Tang},
      year={2025},
      eprint={2505.23313},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.23313}, 
}

@misc{wang2025EventPARbenchmark,
      title={RGB-Event based Pedestrian Attribute Recognition: A Benchmark Dataset and An Asymmetric RWKV Fusion Framework}, 
      author={Xiao Wang and Haiyang Wang and Shiao Wang and Qiang Chen and Jiandong Jin and Haoyu Song and Bo Jiang and Chenglong Li},
      year={2025},
      eprint={2504.10018},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.10018}, 
}

@misc{jin2024MSP60Kbenchmark,
      title={Pedestrian Attribute Recognition: A New Benchmark Dataset and A Large Language Model Augmented Framework}, 
      author={Jiandong Jin and Xiao Wang and Qian Zhu and Haiyang Wang and Chenglong Li},
      year={2024},
      eprint={2408.09720},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.09720}, 
}

@misc{wang2024SNNPAR,
      title={SNN-PAR: Energy Efficient Pedestrian Attribute Recognition via Spiking Neural Networks}, 
      author={Haiyang Wang and Qian Zhu and Mowen She and Yabo Li and Haoyu Song and Minghe Xu and Xiao Wang},
      year={2024},
      eprint={2410.07857},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.07857}, 
}


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
