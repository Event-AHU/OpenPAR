### MSP60K Benchmark Dataset for Pedestrian Attribute Recognition 

#### Abstract 
Pedestrian Attribute Recognition (PAR) is one of the indispensable tasks in human-centered research. However, existing datasets neglect different domains (e.g., environments, times, populations, and data sources), only conducting simple random splits, and the performance of these datasets has already approached saturation. In the past five years, no large-scale dataset has been opened to the public. To address this issue, this paper proposes a new large-scale, cross-domain pedestrian attribute recognition dataset to fill the data gap, termed MSP60K. It consists of 60,122 images and 57 attribute annotations across eight scenarios. Synthetic degradation is also conducted to further narrow the gap between the dataset and real-world challenging scenarios. To establish a more rigorous benchmark, we evaluate 17 representative PAR models under both random and cross-domain split protocols on our dataset. Additionally, we propose an innovative Large Language Model (LLM) augmented PAR framework, named LLM-PAR. This framework processes pedestrian images through a Vision Transformer (ViT) backbone to extract features and introduces a multi-embedding query Transformer to learn partial-aware features for attribute classification. Significantly, we enhance this framework with LLM for ensemble learning and visual feature augmentation. Comprehensive experiments across multiple PAR benchmark datasets have thoroughly validated the efficacy of our proposed framework. 
<img src="https://github.com/Event-AHU/OpenPAR/blob/main/MSP60K_Benchmark_Dataset/figures/dataset_comparison.jpg" width="800">

#### MSP60K Dataset 
Our MSP60K dataset consists of 60,122 images and 57 attribute annotations across eight scenarios.

<img src="https://github.com/Event-AHU/OpenPAR/blob/main/MSP60K_Benchmark_Dataset/figures/statics.jpg" width="800">


The original images of our MSP60k dataset were released on [BaiduYun](https://pan.baidu.com/s/1LW-iBwr26cCikR9u82e7UA?pwd=msp6) [GoogleDriver](https://drive.google.com/file/d/1KBHFPJHtY6V-XrdLo3r-s2I6aDvWoTz4/view?usp=sharing), the degraded images and code have found in `/degrade_images` and `/degrade`.

<img src="https://github.com/Event-AHU/OpenPAR/blob/main/MSP60K_Benchmark_Dataset/figures/dataset_detail3.jpg" width="800">

<img src="https://github.com/Event-AHU/OpenPAR/blob/main/MSP60K_Benchmark_Dataset/figures/dataset_tsne.jpg" width="800">

#### Benchmark Results 

<img src="https://github.com/Event-AHU/OpenPAR/blob/main/MSP60K_Benchmark_Dataset/figures/MSP60k_result.png" width="800">
<img src="https://github.com/Event-AHU/OpenPAR/blob/main/MSP60K_Benchmark_Dataset/figures/LLMPAR_Result.png" width="800">

#### Newly Proposed LLM-PAR Framework 

<img src="https://github.com/Event-AHU/OpenPAR/blob/main/MSP60K_Benchmark_Dataset/figures/LLMPAR_frameworkv2.jpg" width="800">

#### Environment Configure 
We use a single RTX A800 80G GPU for training and evaluation.
Create Environment
```
conda create -n llmpar python=3.8
conda activate llmpar
bash install.sh
```
Dataset Preparation Refer To [README](https://github.com/Event-AHU/OpenPAR/blob/main/README.md).


Change the dataset `PKL` and the `Dataset Image` in [local.py](https://github.com/Event-AHU/OpenPAR/blob/main/MSP60K_Benchmark_Dataset/LLM-PAR/local.py)


We use the [EVA-CLIP-Gint](https://github.com/baaivision/EVA/blob/master/EVA-CLIP) as our visual encoder, the [Vicuna-7b](https://github.com/lm-sys/FastChat) as LLM, and using [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) pre-trained weights, these weight path can be changed in [local.py](https://github.com/Event-AHU/OpenPAR/blob/main/MSP60K_Benchmark_Dataset/LLM-PAR/local.py).
#### Training and Testing 
Training
```
python train.py --dataset PETA --exp expname
```

Inference
```
python train.py --dataset PETA --exp expname --ckpt_path ./logs/PETA/expname/Epoch60.pth
```

#### Checkpoint Download
Dataset  | BaiduYun | Extracted code|
|:-------------|:---------:|:---------:|
| RAP  | [BaiduYun](https://pan.baidu.com/s/1DNMtdvpTr-dUyESxyecETg?pwd=3n97) | 3n97 |
| PETA  | [BaiduYun](https://pan.baidu.com/s/1ury7JR82QNt1MYrl9ly2fQ?pwd=4mfp) | 4mfp |

#### Anknowledgement 
Our code is extended from the following repositories. We sincerely appreciate for their contributions.
* [Vicuna-7b](https://github.com/lm-sys/FastChat)
* [EVA-CLIP-Gint](https://github.com/baaivision/EVA/blob/master/EVA-CLIP)
* [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4)
* [BLIP2](https://github.com/salesforce/LAVIS)

#### Citation 
If you find this work helps your research, please star this GitHub and cite the following papers: 
```bibtex
@article{jin2024pedestrian,
  title={Pedestrian Attribute Recognition: A New Benchmark Dataset and A Large Language Model Augmented Framework},
  author={Jin, Jiandong and Wang, Xiao and Zhu, Qian and Wang, Haiyang and Li, Chenglong},
  journal={arXiv preprint arXiv:2408.09720},
  year={2024}
}
```
