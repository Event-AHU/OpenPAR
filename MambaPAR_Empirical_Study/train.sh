# --only_img  # 只用图片分支，且使用VMamba
# --only_img --use_Vis_model Vit # 只使用图片分支，且使用Vit
# --only_img --use_Vis_model Vim # 只使用图片分支，且使用Vim

# --use_Vis_model Vim # 使用mamba做为文本分支，Vim为视觉分支

# 'PA100k', 'RAPv1',"PETA", 'RAPv2', "PETAzs", "RAPv2zs","Wider" 可选数据集


python train.py PA100k \
    --batchsize 64 --lr 0.00025 \
    --gpu 3 --epoch 60 \
    --epoch_save_ckpt 100 \
    --height 224 --width 224 \
    --use_Vis_model Vim