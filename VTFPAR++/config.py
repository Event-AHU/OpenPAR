import argparse


def argument_parser():
    parser = argparse.ArgumentParser(description="attribute recognition",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("dataset", type=str, default="MARS")
    
    parser.add_argument("--batchsize", type=int, default=16)
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    
    parser.add_argument("--length", type=int, default=15)
    parser.add_argument("--frames", type=int, default=6)
    
    parser.add_argument("--train_split", type=str, default="trainval", choices=['train', 'trainval'])
    parser.add_argument("--valid_split", type=str, default="test", choices=['test', 'valid'])
    
    parser.add_argument('--gpus', default='1', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument("--redirector", action='store_false')
    parser.add_argument("--epoch_save_ckpt", type=int, default=100)

    parser.add_argument("--check_point", type=bool, default=False)
    parser.add_argument("--dir", type=str, default=None)
    parser.add_argument("--avg_frame_extract", action='store_true')
    parser.add_argument("--without_temporal", action='store_true')
    parser.add_argument("--without_spatial", action='store_true')
    parser.add_argument("--clip_model", type=str, default="ViT-B/16", choices=['ViT-B/16', 'ViT-L/14'])
    parser.add_argument("--fusion_type", type=str, default='add')
    parser.add_argument("--spatial_interact_map", type=str, default='default',choices=['default', '0', '1', '2', '3', '4', '5', '6', '7'])
    parser.add_argument("--temporal_interact_map", type=str, default='default',choices=['default', '0', '1', '2', '3', '4', '5', '6', '7'])
    parser.add_argument("--spatial_feat_aggregation", type=str, default='mean_pooling',choices=['mean_pooling', 'LSTM', 'GRU', 'MLP'])
    parser.add_argument("--temporal_feat_aggregation", type=str, default='mean_pooling',choices=['mean_pooling', 'LSTM', 'GRU', 'MLP'])
    parser.add_argument("--mmformer_dim", type=int, default=240)
    parser.add_argument("--spatial_dim", type=int, default=240, choices=[144,240,256,336,432,512,768])
    parser.add_argument("--temporal_dim", type=int, default=240, choices=[144,240,256,336,432,512,768])
    return parser

