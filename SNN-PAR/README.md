<h2 align="center"> SNN-PAR: Energy Efficient Pedestrian Attribute Recognition via Spiking Neural Networks </h2>
 **Energy Efficient Pedestrian Attribute Recognition via Spiking Neural Networks"**, Haiyang Wang, Qian Zhu, Mowen She, Yabo Li, Haoyu Song,Minghe Xu, and Xiao Wang*
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
python pa100k.py
```
###  Checkpoint Download

Dataset  | BaiduYun | Extracted code
```
|:-------------|:---------:|:---------:|
| PA100k  | [BaiduYun](https://pan.baidu.com/s/1ITe5kgk_smaLWMVQZ9WAzA)  | BICS 
```


need to be download for training.If you want to use the weights from the RAP and PETA datasets, you can get them by running:
```
python train.py PETA
python train.py RAP
```

## 🚀Training
```
python train.py PA1000k --only_feats_kl  --only_logits_kl 
```
## Abstract 
Artificial neural network based Pedestrian Attribute Recognition (PAR) has been widely studied in recent years, despite many progresses, however, the energy consumption is still high. To address this issue, in this paper, we propose a Spiking Neural Network (SNN) based framework for energy-efficient attribute recognition. Specifically, we first adopt a spiking tokenizer module to transform the given pedestrian image into spiking feature representations. Then, the output will be fed into the spiking Transformer backbone networks for energy-efficient feature extraction. We feed the enhanced spiking features into a set of feedforward networks for pedestrian attribute recognition. In addition to the widely used binary cross-entropy loss function, we also exploit knowledge distillation from the artificial neural network to the spiking Transformer network for more accurate attribute recognition. Extensive experiments on three widely used PAR benchmark datasets fully validated the effectiveness of our proposed SNN-PAR framework.


## Experimental Results 

<img src="https://github.com/Event-AHU/OpenPAR/edit/main/SNN-PAR/figures/RightPred.jpg" width="500">

<img src="https://github.com/Event-AHU/OpenPAR/edit/main/SNN-PAR/figures/heatmap.jpg" width="500">

## 👍Acknowledgements
This code is based on [VTB](https://github.com/cxh0519/VTB/tree/main) and [Spikingformer](https://github.com/zhouchenlin2096/Spikingformer). Thanks for their efforts.
