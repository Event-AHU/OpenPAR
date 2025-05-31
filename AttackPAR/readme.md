## Adversarial Attack Meets PAR 

> **[Adversarial Semantic and Label Perturbation Attack for Pedestrian Attribute Recognition]()**, Weizhe Kong, Xiao Wang*, Ruichong Gao, Chenglong Li*, Yu Zhang, Xing Yang, Yaowei Wang, Jin Tang 



### :rocket: News 

### :bookmark_tabs: Abastract 
Pedestrian Attribute Recognition (PAR) is an indispensable task in human-centered research and has made great progress in recent years with the development of deep neural networks. However, the potential vulnerability and anti-interference ability have still not been fully explored. To bridge this gap, this paper proposes the first adversarial attack and defense framework for pedestrian attribute recognition. Specifically, we exploit both global- and patch-level attacks on the pedestrian images, based on the pre-trained CLIP-based PAR framework. It first divides the input pedestrian image into non-overlapping patches and embeds them into feature embeddings using a projection layer. Meanwhile, the attribute set is expanded into sentences using prompts and embedded into attribute features using a pre-trained CLIP text encoder. A multi-modal Transformer is adopted to fuse the obtained vision and text tokens, and a feed-forward network is utilized for attribute recognition. Based on the aforementioned PAR framework, we adopt the adversarial semantic and label-perturbation to generate the adversarial noise, termed ASL-PAR. We also design a semantic offset defense strategy to suppress the influence of adversarial attacks. Extensive experiments conducted on both digital domains (i.e., PETA, PA100K, MSP60K, RAPv2) and physical domains fully validated the effectiveness of our proposed adversarial attack and defense strategies for the pedestrian attribute recognition. 

### :mag_right: Our Proposed Approach


### :wrench: Environment Requirements 


### :hammer:  Training and Testing 


### :chart_with_upwards_trend: Experimental Results 
#### Main Experimental Results:
![image](https://github.com/user-attachments/assets/ff3edba5-51d2-4d3d-82f8-e81e4db90209)
![image](https://github.com/user-attachments/assets/4270add9-c550-4009-a533-1e7369eab2a2)

#### Ablation experiment:
![image](https://github.com/user-attachments/assets/f383d1b9-0698-4c4f-819e-ffc53cc46bf6)

#### Experiments on Cross-dataset attacks:
![image](https://github.com/user-attachments/assets/4a80261b-4710-4e64-b396-cf60b03fac40)

### :art: Visualization 


### :gift_heart: Acknowledgement 


### :cupid: Citation 
```
@misc{kong2025adversarialsemanticlabelperturbation,
      title={Adversarial Semantic and Label Perturbation Attack for Pedestrian Attribute Recognition}, 
      author={Weizhe Kong and Xiao Wang and Ruichong Gao and Chenglong Li and Yu Zhang and Xing Yang and Yaowei Wang and Jin Tang},
      year={2025},
      eprint={2505.23313},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.23313}, 
}
```


