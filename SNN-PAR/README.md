<h2 align="center"> SNN-PAR: Energy Efficient Pedestrian Attribute Recognition via Spiking Neural Networks </h2>
> **Energy Efficient Pedestrian Attribute Recognition via Spiking Neural Networks**, Haiyang Wang, Qian Zhu, Mowen She, Yabo Li, Haoyu Song,Minghe Xu, and Xiao Wang*
<p align="center">
  <img src="figures/SNNPAR_framework.png" width="75%">
</p>

## 🔧Requirements
### Installation
```
pip install -r requirements.txt
```
### Data Preparation
```
cd dataset/preprocess
python rap.py
```
### Pre-trained Model
ImageNet pre-trained [ViT-Base](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth) need to be download for training.

## 🚀Training
```
python train.py RAP
```

## 📌Citation
If you found this code/work to be useful in your own research, please consider citing the following:
```
@article{cheng2022simple,
  title={A Simple Visual-Textual Baseline for Pedestrian Attribute Recognition},
  author={Cheng, Xinhua and Jia, Mengxi and Wang, Qian and Zhang, Jian},
  journal={IEEE Transactions on Circuits and Systems for Video Technology (TCSVT)},
  year={2022}
}
```

## 👍Acknowledgements
This code is based on [Rethinking_of_PAR](https://github.com/valencebond/Rethinking_of_PAR) and [TransReID](https://github.com/damo-cv/TransReID). Thanks for their efforts.
