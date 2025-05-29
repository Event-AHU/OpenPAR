CUDA_VISIBLE_DEVICES=7 python train_defense.py \
    --lr 8e-4 \
    --clip_lr 4e-4 \
    --dataset PETA \
    --use_textprompt \
    --use_div \
    --use_vismask \
    --use_GL \
    --use_mm_former \
    --batchsize 32