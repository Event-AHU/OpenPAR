<h2> SNN-PAR: Energy Efficient Pedestrian Attribute Recognition via Spiking Neural Networks </h2>
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
###  Teacher Checkpoint 
You can get the weights of the teacher model by training the [VTB](https://github.com/cxh0519/VTB/tree/main) separately

## 🚀Training
```
python train.py PA100k --only_feats_kl  --only_logits_kl 
```
## Abstract 
Artificial neural network based Pedestrian Attribute Recognition (PAR) has been widely studied in recent years, despite many progresses, however, the energy consumption is still high. To address this issue, in this paper, we propose a Spiking Neural Network (SNN) based framework for energy-efficient attribute recognition. Specifically, we first adopt a spiking tokenizer module to transform the given pedestrian image into spiking feature representations. Then, the output will be fed into the spiking Transformer backbone networks for energy-efficient feature extraction. We feed the enhanced spiking features into a set of feedforward networks for pedestrian attribute recognition. In addition to the widely used binary cross-entropy loss function, we also exploit knowledge distillation from the artificial neural network to the spiking Transformer network for more accurate attribute recognition. Extensive experiments on three widely used PAR benchmark datasets fully validated the effectiveness of our proposed SNN-PAR framework.





## 👍Acknowledgements
This code is based on [VTB](https://github.com/cxh0519/VTB/tree/main) and [Spikingformer](https://github.com/zhouchenlin2096/Spikingformer). Thanks for their efforts.
