google_bert_path= '/gpfs/home/16051/data1/jinjiandong/model/google-bertbert-base-uncased'
minigpt4_path= '/gpfs/home/16051/data1/jinjiandong/model/vicuna-7b/prerained_minigpt4_7b.pth'
vicuna_7b_path= '/gpfs/home/16051/data1/jinjiandong/model/vicuna-7b'
blip2_path = '/gpfs/home/16051/data1/jinjiandong/model/blip2_pretrained_flant5xxl.pth'
eva_vit_g_path = '/gpfs/home/16051/data1/jinjiandong/model/eva_vit_g.pth'

def get_pkl_rootpath(dataset):
    if dataset=="RAPv1":
        root_path='/gpfs/home/16051/data1/jinjiandong/dataset/RAPv1/RAP_dataset'
        pkl_path='/gpfs/home/16051/data1/jinjiandong/dataset/RAPv1/rap1_template.pkl'
    elif dataset=="RAPv2":  
        root_path='/gpfs/home/16051/data1/jinjiandong/dataset/RAPv2/RAP_dataset'
        pkl_path='/gpfs/home/16051/data1/jinjiandong/dataset/RAPv2/rap2_template.pkl'
    elif dataset=="PETA": 
        root_path='/gpfs/home/16051/data1/jinjiandong/dataset/PETA/images'
        pkl_path='/gpfs/home/16051/data1/jinjiandong/dataset/PETA/peta_template.pkl'
    elif dataset=="PA100k": 
        root_path='/gpfs/home/16051/data1/jinjiandong/dataset/PA100k/release_data/release_data'
        pkl_path='/gpfs/home/16051/data1/jinjiandong/dataset/PA100k/pa100k_template.pkl'
    elif dataset=="RAPzs": 
        root_path='/gpfs/home/16051/data1/jinjiandong/dataset/RAPv2/RAP_dataset'
        pkl_path='/gpfs/home/16051/data1/jinjiandong/dataset/RAPv2/rapzs_template.pkl'
    elif dataset=="PETAzs":
        root_path='/gpfs/home/16051/data1/jinjiandong/dataset/PETA/images'
        pkl_path='/gpfs/home/16051/data1/jinjiandong/dataset/PETA/petazs_template.pkl'
    elif dataset=="MSP":
        root_path='/gpfs/home/16051/data1/jinjiandong/dataset/MSP/images'
        pkl_path='/gpfs/home/16051/data1/jinjiandong/dataset/MSP/msp_random_template.pkl'
    elif dataset=="MSPCD":
        root_path='/gpfs/home/16051/data1/jinjiandong/dataset/MSP/images'
        pkl_path='/gpfs/home/16051/data1/jinjiandong/dataset/MSP/msp_cd_template.pkl'
    return pkl_path, root_path
