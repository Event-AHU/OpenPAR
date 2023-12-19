#SequencePAR

<div align="center">
 
<img src="https://github.com/Event-AHU/OpenPAR/blob/main/SequencePAR/figures/SequencePAR_logo.png" width="600">

 **Official PyTorch implementation of "SequencePAR: Understanding Pedestrian Attributes via A Sequence Generation Paradigm"**

 ------
 
</div>

> **[SequencePAR: Understanding Pedestrian Attributes via A Sequence Generation Paradigm]()**, Jiandong Jin, Xiao Wang *, Chenglong Li, Lili Huang, and Jin Tang  


 



## News: 


## Abstract 
Current pedestrian attribute recognition (PAR) algorithms are developed based on multi-label or multi-task learning frameworks, which aim to discriminate the attributes using specific classification heads. However, these discriminative models are easily influenced by imbalanced data or noisy samples. Inspired by the success of generative models, we rethink the pedestrian attribute recognition scheme and believe the generative models may perform better on modeling dependencies and complexity between human attributes. In this paper, we propose a novel sequence generation paradigm for pedestrian attribute recognition, termed SequencePAR. It extracts the pedestrian features using a pre-trained CLIP model and embeds the attribute set into query tokens under the guidance of text prompts. Then, a Transformer decoder is proposed to generate the human attributes by incorporating the visual features and attribute query tokens. The masked multi-head attention layer is introduced into the decoder module to prevent the model from remembering the next attribute while making attribute predictions during training. Extensive experiments on multiple widely used pedestrian attribute recognition datasets fully validated the effectiveness of our proposed SequencePAR. 

<img src="https://github.com/Event-AHU/OpenPAR/blob/main/SequencePAR/figures/frontImage.jpg" width="800">



## Environment 


## Our Proposed Approach 
<img src="https://github.com/Event-AHU/OpenPAR/blob/main/SequencePAR/figures/pipeline.jpg" width="800">


## Dataset 



## Experimental Results 

<img src="https://github.com/Event-AHU/OpenPAR/blob/main/SequencePAR/figures/featuremap_vis.png" width="800">

<img src="https://github.com/Event-AHU/OpenPAR/blob/main/SequencePAR/figures/attResults_vis.jpg" width="800">
