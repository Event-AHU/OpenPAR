# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py PETA --gpus=0,1,2,3
python train.py EventPAR  --multi --backbones rwkv