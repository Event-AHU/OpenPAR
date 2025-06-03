CUDA_VISIBLE_DEVICES=1 python test_ourattack.py \
    --dataset PETA \
    --use_textprompt \
    --use_div \
    --use_vismask \
    --use_GL \
    --use_mm_former \
    --batchsize 64