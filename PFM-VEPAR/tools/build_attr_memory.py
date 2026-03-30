import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import models.backbone


@torch.no_grad()
def extract_feat(model, image_tensor, reduce='avg'):
    feat_map = model(image_tensor)  # expect [B, N, D] or [B, C, H, W]
    if feat_map.dim() == 4:
        B, C, H, W = feat_map.shape
        feat_map = feat_map.flatten(2).transpose(1, 2)  # [B, N, D]
    if reduce == 'avg':
        return feat_map.mean(dim=1)  # [B, D]
    elif reduce == 'cls':
        return feat_map[:, 0, :]
    else:
        raise ValueError('reduce must be avg or cls')


def kmeans_centers(mat: torch.Tensor, k: int) -> torch.Tensor:
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, n_init='auto', random_state=0)
    km.fit(mat.numpy())
    return torch.from_numpy(km.cluster_centers_).float()


def build_attr_memory(
    backbone_type,
    num_attr=50,
    per_attr_sample=1000,
    per_attr_prototype=100,
    feat_reduce='avg',
    save_path='assets/memory/attr_memory.pt',
    batch_size=64,
    device='cuda',
    modality='rgb',
    use_weighted_sampling=True,
    min_samples_per_attr=50,
    use_train_transform=True
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    from dataset.pedes_attr.pedes import PedesAttr
    from models.model_factory import build_backbone
    from configs import default as cfg_default
    from dataset.augmentation import get_transform

    cfg = cfg_default._C.clone()
    cfg.DATASET.NAME = 'EventPAR'
    # cfg.DATASET.TRAIN_SPLIT = 'train'
    cfg.DATASET.TRAIN_SPLIT = 'trainval'
    train_tsfm, valid_tsfm = get_transform(cfg)

    transform_to_use = train_tsfm if use_train_transform else valid_tsfm
    dataset = PedesAttr(cfg=cfg, split=cfg.DATASET.TRAIN_SPLIT, transform=transform_to_use, target_transform=None)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    import models.backbone.vit

    try:
        backbone, _ = build_backbone(backbone_type, False)
    except KeyError:
        if backbone_type == 'vit_b':
            from models.backbone.vit import vit_base_patch16_224
            backbone = vit_base_patch16_224()
        elif backbone_type == 'vit_s':
            from models.backbone.vit import vit_small_patch16_224
            backbone = vit_small_patch16_224()
        else:
            raise
    model = backbone.to(device).eval()
    attr_counts = [0] * num_attr
    for batch in tqdm(loader, desc="Counting"):
        rgb_imgs, event_imgs, labels, _ = batch
        if not torch.is_tensor(labels):
            labels = torch.from_numpy(labels)
        
        for i in range(labels.size(0)):
            attr_ids = torch.nonzero(labels[i]).flatten().tolist()
            for a in attr_ids:
                if 0 <= a < num_attr:
                    attr_counts[a] += 1
    if use_weighted_sampling:
        total_samples = sum(attr_counts)
        attr_weights = {}
        for a in range(num_attr):
            if attr_counts[a] > 0:
                weight = total_samples / (num_attr * attr_counts[a])
                attr_weights[a] = min(weight, 3.0)
            else:
                attr_weights[a] = 0.0
    else:
        attr_weights = {a: 1.0 for a in range(num_attr)}

    buckets = [[] for _ in range(num_attr)]
    for batch in tqdm(loader, desc="Building memory"):
        rgb_imgs, event_imgs, labels, _ = batch
        if not torch.is_tensor(labels):
            labels = torch.from_numpy(labels)

        if modality == 'rgb':
            rgb_in = rgb_imgs.squeeze(1).to(device)
            feats = extract_feat(model, rgb_in, reduce=feat_reduce).cpu()  # [B, D]
        elif modality == 'event':
            B, F, C, H, W = event_imgs.shape
            event_in = event_imgs.view(B * F, C, H, W).to(device)
            feat_all = extract_feat(model, event_in, reduce=feat_reduce).cpu()  # [B*F, D]
            feats = feat_all.view(B, F, -1).mean(dim=1)  # [B, D]
        else:
            raise ValueError('modality must be rgb or event')
        
        for i in range(feats.size(0)):
            attr_ids = torch.nonzero(labels[i]).flatten().tolist()
            for a in attr_ids:
                if 0 <= a < num_attr:
                    current_count = len(buckets[a])
                    max_samples = int(per_attr_sample * attr_weights.get(a, 1.0))
                    min_samples = min_samples_per_attr

                    if current_count < max_samples or (current_count < min_samples and attr_counts[a] >= min_samples):
                        buckets[a].append(feats[i])

    per_attr_memory = []
    for a in range(num_attr):
        if len(buckets[a]) == 0:
            continue
        mat = torch.stack(buckets[a])  # [m, D]
        if per_attr_prototype and per_attr_prototype > 0 and len(buckets[a]) > per_attr_prototype:
            k = min(per_attr_prototype, len(buckets[a]))
            centers = kmeans_centers(mat, k)     # [k, D]
            per_attr_memory.append(centers)
        else:
            per_attr_memory.append(mat.mean(dim=0, keepdim=True))  # [1, D]

    if len(per_attr_memory) == 0:
        raise RuntimeError('No attribute memory could be constructed. Check dataset labels.')

    memory = torch.cat(per_attr_memory, dim=0)  # [M, D]
    torch.save(memory, save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='vit_b')
    parser.add_argument('--num_attr', type=int, default=50)
    parser.add_argument('--per_attr_sample', type=int, default=100)
    parser.add_argument('--per_attr_prototype', type=int, default=1000)
    parser.add_argument('--feat_reduce', type=str, default='avg', choices=['avg', 'cls'])
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--modality', type=str, default='rgb', choices=['rgb', 'event'])
    parser.add_argument('--use_weighted_sampling', action='store_true', default=False)
    parser.add_argument('--min_samples_per_attr', type=int, default=50)
    parser.add_argument('--use_train_transform', action='store_true', default=True)
    args = parser.parse_args()

    build_attr_memory(
        backbone_type=args.backbone,
        num_attr=args.num_attr,
        per_attr_sample=args.per_attr_sample,
        per_attr_prototype=args.per_attr_prototype,
        feat_reduce=args.feat_reduce,
        save_path=args.save_path,
        batch_size=args.batch_size,
        modality=args.modality,
        use_weighted_sampling=args.use_weighted_sampling,
        min_samples_per_attr=args.min_samples_per_attr,
        use_train_transform=args.use_train_transform
    )


if __name__ == '__main__':
    main()


