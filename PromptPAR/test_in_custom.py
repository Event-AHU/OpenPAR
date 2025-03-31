import os
import pprint
from collections import OrderedDict
import time
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
from batch_engine import valid_trainer
from loss.CE_loss import CEL_Sigmoid
from models.base_block import TransformerClassifier
from tools.function import get_pedestrian_metrics
from tools.utils import time_str, set_seed
from clip import clip
from clip.model import build_model
from torchvision import transforms
from PIL import Image
from config import argument_parser
from utils.train_utils import AverageMeter
set_seed(605)
device = "cuda" if torch.cuda.is_available() else "cpu"
# attr_words的设置取决于你加载哪个数据集训练的checkpoint 例如PETA
attr_words = [
   'head hat',
   'head muffler',
   'head nothing',
   'head sunglasses',
   'head long hair',
   'upper casual', 
   'upper formal', 
   'upper jacket', 
   'upper logo', 
   'upper plaid', 
   'upper short sleeve', 
   'upper thin stripes', 
   'upper t-shirt',
   'upper other',
   'upper v-neck',
   'lower Casual', 
   'lower Formal', 
   'lower Jeans', 
   'lower Shorts', 
   'lower Short Skirt',
   'lower Trousers',
   'shoes Leather', 
   'shoes Sandals', 
   'shoes other', 
   'shoes sneaker',
   'attach Backpack', 
   'attach Other', 
   'attach messenger bag', 
   'attach nothing', 
   'attach plastic bags',
   'age less 30',
   'age 30 45',
   'age 45 60',
   'age over 60',
   'male'
] # 54 

class CustomDataset(Dataset):
    def __init__(self, image_root, transform=None):
        self.image_root = image_root
        print(f"image_root: {image_root}")
        self.image_list = sorted([os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith(('.jpg', '.png'))])
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path  # 这里不返回标签，适用于无标签测试

def main(args, image_root):
    print(time_str())
    pprint.pprint(OrderedDict(vars(args)))
    print('-' * 60)

    # 设置你的数据集属性
    attr_num = len(attr_words) # 手动设置属性数量
    attributes = attr_words # 手动设置属性名称列表

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    dataset = CustomDataset(image_root, transform=transform)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=8, pin_memory=True)

    print(f'Test set size: {len(dataset)}, attr_num: {attr_num}')

    print("Start loading model")
    if os.path.exists(args.dir):
        checkpoint = torch.load(args.dir, map_location=device)
        clip_model = build_model(checkpoint['ViT_model'])
        model = TransformerClassifier(clip_model, attr_num, attributes)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model = model.to(device)
        clip_model = clip_model.to(device)
    else:
        print(f"Warning: Checkpoint {args.dir} not found, skipping model loading.")
        return
    print("Starting evaluation...")
    start = time.time()

    model.eval()
    # loss_meter = AverageMeter() # ----> used nowhere
    preds_probs = []
    imgnames= []
    # gt_list = [] # ----> used nowhere
    with torch.no_grad():
        for step, (imgs, imgname) in enumerate(data_loader):
            imgs = imgs.cuda()
            valid_logits, _ = model(imgs, clip_model=clip_model)
            valid_probs = torch.sigmoid(valid_logits)
            preds_probs.append(valid_probs.cpu().numpy())
            imgnames.append(imgname)

    preds_attrs = {}
    for batch_preds, batch_imgs in zip(preds_probs, imgnames):
        for i, sample_preds in enumerate(batch_preds):
            sample_attrs = []
            # Now iterate over each attribute score in the sample
            for aidx, attr_score in enumerate(sample_preds):
                if attr_score > 0.45:  # now attr_score is a scalar value
                    sample_attrs.append(attributes[aidx])
                    
            
            preds_attrs[batch_imgs[i]]= sample_attrs
        
    end = time.time()
    print(f'Total test time: {end - start:.2f} seconds')
    
    with open("preds.json" , "w") as f:
        json.dump(preds_attrs, f, indent=4)



if __name__ == '__main__':
    parser = argument_parser()
    image_root = ''
    args = parser.parse_args()
    main(args, image_root)

