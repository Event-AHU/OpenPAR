import torch

def compute_AP(predictions, labels):
    num_class = predictions.size(1)
    ap = torch.zeros(num_class).to(predictions.device)
    empty_class = 0
    for idx_cls in range(num_class):
        prediction = predictions[:, idx_cls]
        label = labels[:, idx_cls]
        mask = label != -1
        if (label > 0).sum() == 0:
            empty_class += 1
            continue
        binary_label = torch.clamp(label[mask], min=0, max=1)
        sorted_pred, sort_idx = prediction[mask].sort(descending=True)
        sorted_label = binary_label[sort_idx]
        tmp = (sorted_label == 1).float()
        tp = tmp.cumsum(0)
        fp = (sorted_label != 1).float().cumsum(0)
        num_pos = binary_label.sum()
        rec = tp / num_pos
        prec = tp / (tp + fp)
        ap_cls = (tmp * prec).sum() / num_pos
        ap[idx_cls].copy_(ap_cls)
    return ap

def compute_F1(predictions, labels, mode_F1, k_val):
    idx = predictions.topk(dim=1, k=k_val)[1]
    predictions.fill_(0)
    predictions.scatter_(dim=1, index=idx, src=torch.ones(predictions.size(0), k_val, device=predictions.device, dtype=predictions.dtype))
    mask = predictions == 1
    TP = (labels[mask] == 1).sum().float()
    tpfp = mask.sum().float()
    tpfn = (labels == 1).sum().float()
    p = TP / tpfp
    r = TP / tpfn
    f1 = 2 * p * r / (p + r)
    return (f1, p, r)

def harmonic_mean(first, second):
    first = float(first)
    second = float(second)
    if first <= 0 or second <= 0:
        return 0.0
    return 2.0 * first * second / (first + second)

def _as_float(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)

def compute_metrics(predictions, labels, novel_predictions, novel_labels, base_count):
    if predictions.ndim != 2 or labels.shape != predictions.shape:
        raise ValueError('predictions and labels must be equal two-dimensional tensors')
    if novel_predictions.ndim != 2 or novel_labels.shape != novel_predictions.shape:
        raise ValueError('novel predictions and labels must have equal shapes')
    if base_count <= 0 or base_count >= predictions.shape[1]:
        raise ValueError('base_count must split the all-attribute dimension')
    if novel_predictions.shape[1] < 5 or predictions.shape[1] < 5:
        raise ValueError('OF5 metrics require at least five attributes')
    novel_ap = compute_AP(novel_predictions, novel_labels)
    zsl_of3, _, _ = compute_F1(novel_predictions.clone(), novel_labels, 'overall', k_val=3)
    zsl_of5, _, _ = compute_F1(novel_predictions.clone(), novel_labels, 'overall', k_val=5)
    total_ap = compute_AP(predictions, labels)
    gzsl_of3, _, _ = compute_F1(predictions.clone(), labels, 'overall', k_val=3)
    gzsl_of5, _, _ = compute_F1(predictions.clone(), labels, 'overall', k_val=5)
    base_map = _as_float(total_ap[:base_count].mean())
    novel_gzsl_map = _as_float(total_ap[base_count:].mean())
    return {'zsl_map': _as_float(novel_ap.mean()), 'zsl_of3': _as_float(zsl_of3), 'zsl_of5': _as_float(zsl_of5), 'gzsl_map': _as_float(total_ap.mean()), 'gzsl_of3': _as_float(gzsl_of3), 'gzsl_of5': _as_float(gzsl_of5), 'base_map': base_map, 'novel_gzsl_map': novel_gzsl_map, 'base_novel_hmean': harmonic_mean(base_map, novel_gzsl_map)}
