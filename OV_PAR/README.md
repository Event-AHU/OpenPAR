# SRENet: Open-Vocabulary Pedestrian Attribute Recognition

PyTorch implementation of **Structural Relation Enhancement for Open-Vocabulary Pedestrian Attribute Recognition** by Jiandong Jin, Chenglong Li, Xiao Wang, and Jin Tang (IEEE Transactions on Image Processing, 2026).

[IEEE paper](https://doi.org/10.1109/TIP.2026.3749411) · [Manuscript PDF](TIP-OVPAR-MajorRevision_0922.pdf)

SRENet combines spatial-structural enhancement (SpaSE), semantic-structural enhancement (SemSE), and attribute-structural enhancement (AttSE) for zero-shot and generalized zero-shot pedestrian attribute recognition.

## Installation

```bash
git clone https://github.com/Event-AHU/OpenPAR.git
cd OpenPAR/OV_PAR
pip install -r requirements.txt
```

Use Python 3.9 and a CUDA-enabled PyTorch installation with a matching torchvision version. CLIP ViT-B/16 weights are downloaded automatically on the first run.

## Data preparation

The code supports MSPOV and RAPOV2; `RAPOV2` is the name used for RAPOV in the paper. Dataset split files will be released separately.

Set `DATASET_PKL` to the local annotation file and `IMAGE_ROOT` to the image directory. The annotation file contains the base/novel attribute names and indices, together with image names and labels for each partition. Training labels contain base attributes; test labels contain all attributes.

Group-wise sentence banks are included in `instruction/` and selected automatically for each dataset. Training and evaluation require local images and split annotations. Standalone evaluation also requires a trained Stage-2 checkpoint.

## Two-stage training

Stage 1 trains the model with base-attribute supervision. Stage 2 loads a selected Stage-1 checkpoint and optimizes the prompt learners. The following commands run both stages and evaluate the final checkpoint.

### MSPOV

```bash
DATASET_PKL=/path/to/msp_ov.pkl \
IMAGE_ROOT=/path/to/MSP60K/images \
GPU=0 bash scripts/train_mspov.sh
```

### RAPOV2

```bash
DATASET_PKL=/path/to/rap_ov2.pkl \
IMAGE_ROOT=/path/to/RAPv2/RAP_dataset \
GPU=0 bash scripts/train_rapov2.sh
```

## Evaluation

```bash
DATASET_PKL=/path/to/msp_ov.pkl \
IMAGE_ROOT=/path/to/MSP60K/images \
CHECKPOINT=/path/to/mspov_stage2.pth \
GPU=0 bash scripts/test_mspov.sh

DATASET_PKL=/path/to/rap_ov2.pkl \
IMAGE_ROOT=/path/to/RAPv2/RAP_dataset \
CHECKPOINT=/path/to/rapov2_stage2.pth \
GPU=0 bash scripts/test_rapov2.sh
```

Evaluation reports ZSL/GZSL mAP, OF3, and OF5. Console values are percentages; JSON and CSV values are fractions in `[0, 1]`.


## Citation

If you use SRENet or the OV-PAR benchmarks in your research, please cite the paper.

```bibtex
@article{jin2026srenet,
  title   = {Structural Relation Enhancement for Open-Vocabulary Pedestrian Attribute Recognition},
  author  = {Jin, Jiandong and Li, Chenglong and Wang, Xiao and Tang, Jin},
  journal = {IEEE Transactions on Image Processing},
  year    = {2026},
  doi     = {10.1109/TIP.2026.3749411}
}
```

## Acknowledgements

- [RAM (Recover and Match)](https://github.com/EricTan7/RAM)
- [CLIP Surgery](https://github.com/xmed-lab/CLIP_Surgery)
- **OpenAI CLIP** provides the pretrained ViT-B/16 weights and tokenizer. [Official code](https://github.com/openai/CLIP)

Third-party notices and license texts are provided in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [licenses/](licenses/).
