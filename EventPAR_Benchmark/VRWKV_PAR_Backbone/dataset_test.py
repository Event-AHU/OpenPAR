from config import argument_parser
from torch.utils.data import DataLoader
from dataset.AttrDataset import MultiModalAttrDataset, get_transform, custom_collate_fn
parser = argument_parser()
args = parser.parse_args()


train_tsfm, valid_tsfm = get_transform(args)
valid_set = MultiModalAttrDataset(args=args, split=args.valid_split, transform=valid_tsfm) 

valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

def get_shape(lst):
    if isinstance(lst, list):
        return [len(lst)] + get_shape(lst[0])
    else:
        return []
    
for step, (imgs, gt_label, imgname, imgtemps) in enumerate(valid_loader):
    print(imgs.shape)
    print(gt_label.shape)
    print(get_shape(imgname)) 
    print(type(imgtemps))  
    print(get_shape(imgtemps))
    print(imgtemps[0].shape)
    break

# for batch in valid_loader:
#     break
# for i in batch:
#     if isinstance(i, list):
#         print(get_shape(i))
#     else:
#         print(i.shape)
#     print('---')
# imgs, gt_labels, imgnames, imgtemps = zip(*batch)
# print(get_shape(imgtemps))