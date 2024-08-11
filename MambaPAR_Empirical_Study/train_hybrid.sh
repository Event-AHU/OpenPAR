# hydrid 
# 1 PaFusion
# 2 N-ASF
# 3 ASF
# 4 MaFormer
# 5 MaHDFT
# 6 AdaMTF
# 7 KDTM
# 8 MaKDF

python train_hybrid.py PA100k \
    --batchsize 64 --lr 0.00025 \
    --gpu 3 --epoch 60 \
    --height 224 --width 224 \
    --use_Vis_model Vim \
    --hybrid 8
