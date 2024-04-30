<div align="center">
      
<img src="https://github.com/Event-AHU/OpenPAR/blob/main/OpenPAR_logo.png" width="600">

**An open-source framework for Pedestrian Attribute Recognition, based on PyTorch**

------

</div>


## :dart: Supported Datasets
```
PETA, PA100K, RAPv1, RAPv2, WIDER, PETA-ZS, RAP-ZS
```


## :dart: [VTFPAR++](https://github.com/Event-AHU/OpenPAR/blob/main/VTFPAR%2B%2B/README.md) 
**Spatio-Temporal Side Tuning Pre-trained Foundation Models for Video-based Pedestrian Attribute Recognition**, arXiv:2404.17929 
Xiao Wang, Qian Zhu, Jiandong Jin, Jun Zhu, Futian Wang, Bo Jiang, Yaowei Wang, Yonghong Tian 
[[Paper](https://arxiv.org/abs/2404.17929)] 
[[Code](https://github.com/Event-AHU/OpenPAR)] 
      
TO BE UPDATE ...







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
