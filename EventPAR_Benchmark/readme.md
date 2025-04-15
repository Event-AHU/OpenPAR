### EventPAR Benchmark Dataset for Pedestrian Attribute Recognition
The first benchmark dataset for the RGB-Event based pedestrian attribute recognition 

### Abstract 
Existing pedestrian attribute recognition methods are generally developed based on RGB frame cameras. However, these approaches are constrained by the limitations of RGB cameras, such as sensitivity to lighting conditions and motion blur, which hinder their performance. Furthermore, current attribute recognition primarily focuses on analyzing pedestrians’ external appearance and clothing, lacking an exploration of emotional dimensions. In this paper, we revisit these issues and propose a novel multi-modal RGB-Event attribute recognition task by drawing inspiration from the advantages of event cameras in low-light, high-speed, and low-power consumption. Specifically, we introduce the first large-scale multimodal pedestrian attribute recognition dataset, termed EventPAR, comprising 100K paired RGB-Event samples that cover 50 attributes related to both appearance and six human emotions, diverse scenes, and various seasons. By retraining and evaluating mainstream PAR models on this dataset, we establish a comprehensive benchmark and provide a solid foundation for future research in terms of data and algorithmic baselines. In addition, we propose a novel RWKV-based multi-modal pedestrian attribute recognition framework, featuring an RWKV visual encoder and an asymmetric RWKV fusion module. Extensive experiments are conducted on our proposed dataset as well as two simulated datasets (MARS-Attribute and DukeMTMC-VID-Attribute), achieving state-of-the-art results.
<p align="center">
  <img src="figures/firstIMG.png" width="80%">
</p>

#### EventPAR Dataset 
Our EventPAR dataset consists of 100K paired RGB-Event samples and 50 attribute annotations across two seasons. The original and degraded versions of our EventPAR dataset have been released. Specifically, `rgb` represents the original RGB samples, `rgbv3` denotes the degraded samples, `event` corresponds to the event frames, and `stream` refers to the event streams.
<p align="center">
  <img src="figures/dataset_sample.jpg" width="80%">
</p>
<p align="center">
  <img src="figures/attr_distribution.jpg" width="80%">
</p>
<p align="center">
  <img src="figures/attr_predict.jpg" width="80%">
</p>
<p align="center">
  <img src="figures/token_select_visual.jpg" width="80%">
</p>

#### Benchmark Results 


#### Newly Proposed LLM-PAR Framework 
<p align="center">
  <img src="figures/framework.jpg" width="80%">
</p>
<p align="center">
    <img src="figures/OTNmodule.jpg" width="80%">
</p>

#### Environment Configure 

#### Training  

#### Anknowledgement 

#### Citation 
