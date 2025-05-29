CUDA_VISIBLE_DEVICES=2 python train_attack.py \
    --dataset PA100k \
    --use_textprompt \
    --use_div \
    --use_vismask \
    --use_GL \
    --use_mm_former \
    --lr 3e-3  \
    --batchsize 48 \
    --epoch 100