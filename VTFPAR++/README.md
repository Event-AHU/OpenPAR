
# VTFPAR++: Spatio-Temporal Side Tuning Pre-trained Foundation Models for Video-based Pedestrian Attribute Recognition
> **Spatio-Temporal Side Tuning Pre-trained Foundation Models for Video-based Pedestrian Attribute Recognition**, Xiao Wang, Qian Zhu, Jiandong Jin, Jun Zhu, Futian Wang*, Bo Jiang*, Yaowei Wang, Yonghong Tian
[[arXiv](https://arxiv.org/abs/2404.17929)]
 
## Abstract 
Existing pedestrian attribute recognition (PAR) algorithms are mainly developed based on a static image, however, the performance is unreliable in challenging scenarios, such as heavy occlusion, motion blur, etc. In this work, we propose to understand human attributes using video frames that can fully use temporal information by fine-tuning a pre-trained multi-modal foundation model efficiently. Specifically, we formulate the video-based PAR as a vision-language fusion problem and adopt a pre-trained foundation model CLIP to extract the visual features. More importantly, we propose a novel spatiotemporal side-tuning strategy to achieve parameter-efficient optimization of the pre-trained vision foundation model. To better utilize the semantic information, we take the full attribute list that needs to be recognized as another input and transform the attribute words/phrases into the corresponding sentence via split, expand, and prompt operations. Then, the text encoder of CLIP is utilized for embedding processed attribute descriptions. The averaged visual tokens and text tokens are concatenated and fed into a fusion Transformer for multi-modal interactive learning. The enhanced tokens will be fed into a classification head for pedestrian attribute prediction. Extensive experiments on two large-scale video-based PAR datasets fully validated the effectiveness of our proposed framework.

## Requirements
we use single RTX3090 24G GPU for training and evaluation.

**Basic Environment**
```
Python 3.9.16
pytorch 1.12.1
torchvision 0.13.1
```
**Installation**
```
pip install -r requirements.txt
```
## Datasets and Pre-trained Models 

**Download from BaiduYun:**

* **MARS Dataset**:
```
链接：https://pan.baidu.com/s/16Krv3AAlBhB9JPa1EKDbLw 提取码：zi08
```

## Training and Testing 
Use the following code to learn a model for MARS Dataset:

Training
```
python ./dataset/preprocess/mars.py
python train.py MARS
```
Testing
```
python eval.py MARS
```



## Demo Video 
A demo video can be found by clicking the image below: 
<p align="center">
<a href="[https://youtu.be/U4uUjci9Gjc](https://youtu.be/yaeLMrr8MxU?si=ZFha5XZsIG4g8E56)">
<img src="https://github.com/Event-AHU/OpenPAR/blob/main/VTFPAR%2B%2B/figures/VTFPAR%2B%2Bdemo.mp4_20240607_113607.505.jpg" alt="VTFPAR++_DemoVideo" width="700"/>
</a>
</p>






If you have any questions about this work, please submit an issue or contact me via **Email**: wangxiaocvpr@foxmail.com or xiaowang@ahu.edu.cn. Thanks for your attention! 
