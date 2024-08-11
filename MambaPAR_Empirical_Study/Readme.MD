## MambaPAR Empirical Study

<div align="center">
<img src="./figures/logo.png" width="600">

 **AnEmpirical Study of Mamba-based Pedestrian Attribute Recognition**

------

</div>

> **[AnEmpirical Study of Mamba-based Pedestrian Attribute Recognition](https://arxiv.org/abs/2407.10374)**, Xiao Wang, Weizhe Kong, Jiandong Jin, Shiao Wang, Ruichong Gao, Qingchuan Ma, Chenglong Li*, Jin Tang


## Usage
### Requirements

we use single RTX 3090 24G GPU for training and evaluation. 

First, you need to set up an environment of [Vim](https://github.com/hustvl/Vim)/[VMamba](https://github.com/MzeroMiko/VMamba).

After that, please run the code below  in your environment:

```
pip install -r requirements.txt
```

## Data Preparation
Download the PETA dataset from [here](http://mmlab.ie.cuhk.edu.hk/projects/PETA.html), PA100k dataset from [here](https://github.com/xh-liu/HydraPlus-Net#pa-100k-dataset) and RAP1 and RAP2 dataset form [here](https://www.rapdataset.com/).



Organize them in `your dataset root dir` folder as follows:
```
|-- your dataset root dir/
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
```

 Run dataset/preprocess/peta.py to get the dataset pkl file
 ```python
python dataset/preprocess/peta.py
 ```

## Training

For MambaPAR:

```linux
sh train.sh
```

For Hybrid:
```python
sh train_hybrid.sh
```
## Config
|Parameters |Implication|
|:---------------------|:---------:|
| only_img | Only the visual branch is used |
| use_Vis_model |  Sets the model used by the vision branch  |
| Hybrid | Select the hybrid architecture to be used |


## News: 


## Abstract 
Current strong pedestrian attribute recognition models are developed based on Transformer networks, which are computationally heavy. Recently proposed models with linear complexity (e.g., Mamba) have garnered significant attention and have achieved a good balance between accuracy and computational cost across a variety of visual tasks. Relevant review articles also suggest that while these models can perform well on some pedestrian attribute recognition datasets, they are generally weaker than the corresponding Transformer models. To further tap into the potential of the novel Mamba architecture for PAR tasks, this paper designs and adapts Mamba into two typical PAR frameworks, i.e., the text-image fusion approach and pure vision Mamba multi-label recognition framework. It is found that interacting with attribute tags as additional input does not always lead to an improvement, specifically, Vim can be enhanced, but VMamba cannot. This paper further designs various hybrid Mamba-Transformer variants and conducts thorough experimental validations. These experimental results indicate that simply enhancing Mamba with a Transformer does not always lead to performance improvements but yields better results under certain settings. We hope this empirical study can further inspire research in Mamba for PAR, and even extend into the domain of multi-label recognition, through the design of these network structures and comprehensive experimentation.



## Our Proposed Approach 
<img src="https://github.com/Event-AHU/OpenPAR/tree/main/MambaPAR_Empirical_Study/figures/MambaPAR.jpg" width="800">



<img src="https://github.com/Event-AHU/OpenPAR/tree/main/MambaPAR_Empirical_Study/figures/HybridMambaFormer.jpg" width="800">


## Experimental Results 

<img src="https://github.com/Event-AHU/OpenPAR/tree/main/MambaPAR_Empirical_Study/figures/efficiencyAnalysis.jpg" width="800">

<img src="https://github.com/Event-AHU/OpenPAR/tree/main/MambaPAR_Empirical_Study/figures/PARresults_VIS.jpg" width="800">

### Acknowledgments

Our code is extended from the following repositories. We sincerely appreciate for their contributions.

* [VTB](https://github.com/cxh0519/VTB)
* [Rethink of PAR](https://github.com/valencebond/Rethinking_of_PAR)
* [Mamba](https://github.com/state-spaces/mamba)
* [Vim](https://github.com/hustvl/Vim)
* [VMamba](https://github.com/MzeroMiko/VMamba)

