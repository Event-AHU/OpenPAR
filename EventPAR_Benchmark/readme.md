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
<p align="center">
  <img src="figures/EventPAR.png" width="80%">
</p>
<p align="center">
  <img src="figures/Mars_duke.png" width="80%">
</p>


#### Newly Proposed EPAR Framework 
<p align="center">
  <img src="figures/framework.jpg" width="80%">
</p>
<p align="center">
    <img src="figures/OTNmodule.jpg" width="80%">
</p>

#### Environment Configure 
We use a single NVIDIA RTX 3090 GPU for training and evaluation. Create Environment
```
conda create -n EventPAR python=3.9
conda activate EventPAR
pip install -r requirements.txt
```
Dataset Preparation Refer To [README](https://github.com/Event-AHU/OpenPAR/blob/main/README.md), and get the pkl file
```
python EventPAR.py
```
Modify the dataset's PKL file and sample data in [AttrDataset.py](https://github.com/Event-AHU/OpenPAR/EventPAR_Benchmark/VRWKV_PAR/dataset/AttrDataset.py)
We use the [RWKV](https://github.com/OpenGVLab/Vision-RWKV) as our visual encoder, these weight path can be changed in [config.py](https://github.com/Event-AHU/OpenPAR/EventPAR_Benchmark/VRWKV_PAR/config.py).
#### Training  
```
bash train.sh
```
#### Anknowledgement 
Our code is extended from the following repositories. We sincerely appreciate for their contributions.
* [VTB](https://github.com/cxh0519/VTB/tree/main)
* [Vision-RWKV](https://github.com/OpenGVLab/Vision-RWKV)
  
#### Citation 
If you find this work helps your research, please star this GitHub and cite the following papers: 
```bibtex
@misc{wang2025rgbeventbasedpedestrianattribute,
      title={RGB-Event based Pedestrian Attribute Recognition: A Benchmark Dataset and An Asymmetric RWKV Fusion Framework}, 
      author={Xiao Wang and Haiyang Wang and Shiao Wang and Qiang Chen and Jiandong Jin and Haoyu Song and Bo Jiang and Chenglong Li},
      year={2025},
      eprint={2504.10018},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.10018}, 
}
```
