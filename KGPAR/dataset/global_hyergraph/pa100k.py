import os
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from torchvision import transforms
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from clip import clip
from collections import defaultdict
import random
import cv2

def unified_clip_feature_extraction(
    pkl_path,
    entity_text_path,
    output_path,
    clip_model_name="ViT-L/14",
    download_root='/media/amax/KGPAR',
    samples_per_attribute=10,
    random_seed=42,
    blur_threshold=50.0,
    enable_blur_detection=True
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    random.seed(random_seed)
    np.random.seed(random_seed)

    if not os.path.exists(entity_text_path):
        raise FileNotFoundError(f"文件不存在: {entity_text_path}")
    with open(entity_text_path, "r", encoding="utf-8") as f:
        text_lines = [line.strip() for line in f if line.strip()]
    
    if len(text_lines) != 26:
        raise ValueError(f"文本行数为 {len(text_lines)}")

    attribute_keys = [
        'female',
        'age over 60', 'age 18 to 60', 'age less 18',
        'front', 'side', 'back',
        'hat', 'glasses', 
        'hand bag', 'shoulder bag', 'backpack', 'hold objects in front', 
        'short sleeve', 'long sleeve', 'upper stride', 'upper logo', 'upper plaid', 'upper splice',
        'lower stripe', 'lower pattern', 'long coat', 'trousers', 'shorts', 'skirt and dress', 'boots'
    ]

    try:
        clip_model, preprocess = clip.load(clip_model_name, device=device, download_root=download_root)
        clip_model.eval()
    except Exception as e:
        return None

    text_tokens = clip.tokenize(text_lines).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens).float()
    feature_dim = text_features.shape[1]

    with open(pkl_path, "rb") as f:
        dataset = pickle.load(f)

    image_names = dataset.image_name
    labels = dataset.label
    root = dataset.root

    attribute_images = defaultdict(list)
    
    for i, (img_name, label) in enumerate(tqdm(zip(image_names, labels), total=len(image_names))):
        img_path = os.path.join(root, img_name)
        if not os.path.exists(img_path):
            continue
            
        for attr_idx in range(len(label)):
            if label[attr_idx] == 1:
                attribute_images[attr_idx].append(i)

    selected_indices_per_attribute = defaultdict(list)
    attribute_sample_counts = {}
    attribute_available_counts = {}
    
    for attr_idx in range(len(text_lines)):
        available_images = attribute_images[attr_idx]
        available_count = len(available_images)
        attribute_available_counts[attr_idx] = available_count

        if available_count <= samples_per_attribute:
            selected_indices = available_images.copy()
            attribute_sample_counts[attr_idx] = available_count
        else:
            selected_indices = random.sample(available_images, samples_per_attribute)
            attribute_sample_counts[attr_idx] = samples_per_attribute
        selected_indices_per_attribute[attr_idx] = selected_indices
    
    attribute_features_dict = {}
    attribute_image_names_dict = {}
    all_image_features_list = []
    all_image_labels_list = []
    all_image_names_list = []
    
    processed_count = 0
    error_count = 0
    blur_rejected_count = 0
    image_replace_count = 0

    for attr_idx in tqdm(range(len(text_lines))):
        selected_indices = selected_indices_per_attribute[attr_idx]
        available_images = attribute_images[attr_idx]
        attr_key = attribute_keys[attr_idx]
        
        if not selected_indices:
            continue

        if len(available_images) <= samples_per_attribute:
            for img_idx in selected_indices:
                img_name = image_names[img_idx]
                label = labels[img_idx]
                img_path = os.path.join(root, img_name)
                
                try:
                    img = Image.open(img_path).convert("RGB")
                    img_tensor = preprocess(img).unsqueeze(0).to(device)

                    with torch.no_grad():
                        visual_output = clip_model.visual(img_tensor.type(clip_model.dtype))
                        
                        if hasattr(visual_output, 'shape'):
                            image_features = visual_output.float()
                            class_token = image_features.squeeze(0).cpu().float()
                        else:
                            clip_image_features, all_class, attnmap, part_patch_tokens, part_indices_info = visual_output
                            if all_class is None:
                                error_count += 1
                                continue
                            class_token = all_class[:, 0, :].squeeze(0).cpu().float()
                        
                        if class_token.shape[0] != feature_dim:
                            error_count += 1
                            continue
                        
                        all_image_features_list.append(class_token)
                        all_image_labels_list.append(torch.tensor(label, dtype=torch.float32))
                        all_image_names_list.append(img_name)
                        processed_count += 1

                        if attr_key not in attribute_features_dict:
                            attribute_features_dict[attr_key] = []
                            attribute_image_names_dict[attr_key] = []
                        attribute_features_dict[attr_key].append(class_token)
                        attribute_image_names_dict[attr_key].append(img_name)

                except Exception as e:
                    error_count += 1
                    continue
        
        else:
            processed_images = set()
            current_processed = 0
            
            for img_idx in selected_indices:
                if current_processed >= samples_per_attribute:
                    break
                    
                img_name = image_names[img_idx]
                label = labels[img_idx]
                img_path = os.path.join(root, img_name)
                
                try:
                    skip_due_to_blur = False
                    if enable_blur_detection:
                        try:
                            img_cv = cv2.imread(img_path)
                            if img_cv is not None:
                                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                                blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()
                                
                                if blur_value < blur_threshold:
                                    skip_due_to_blur = True
                        except:
                            pass
                    
                    if skip_due_to_blur:
                        blur_rejected_count += 1
                        found_replacement = False
                        for alt_idx in available_images:
                            if alt_idx not in processed_images and alt_idx not in selected_indices:
                                alt_img_name = image_names[alt_idx]
                                alt_img_path = os.path.join(root, alt_img_name)

                                alt_blur = False
                                if enable_blur_detection:
                                    try:
                                        alt_img_cv = cv2.imread(alt_img_path)
                                        if alt_img_cv is not None:
                                            alt_gray = cv2.cvtColor(alt_img_cv, cv2.COLOR_BGR2GRAY)
                                            alt_blur_value = cv2.Laplacian(alt_gray, cv2.CV_64F).var()
                                            if alt_blur_value < blur_threshold:
                                                alt_blur = True
                                    except:
                                        pass
                                
                                if not alt_blur:
                                    img_idx = alt_idx
                                    img_name = alt_img_name
                                    img_path = alt_img_path
                                    found_replacement = True
                                    image_replace_count += 1
                                    break
                        
                        if not found_replacement:
                            continue

                    img = Image.open(img_path).convert("RGB")
                    img_tensor = preprocess(img).unsqueeze(0).to(device)

                    with torch.no_grad():
                        visual_output = clip_model.visual(img_tensor.type(clip_model.dtype))
                        
                        if hasattr(visual_output, 'shape'):
                            image_features = visual_output.float()
                            class_token = image_features.squeeze(0).cpu().float()
                        else:
                            clip_image_features, all_class, attnmap, part_patch_tokens, part_indices_info = visual_output
                            if all_class is None:
                                error_count += 1
                                continue
                            class_token = all_class[:, 0, :].squeeze(0).cpu().float()
                        
                        if class_token.shape[0] != feature_dim:
                            error_count += 1
                            continue
                        
                        all_image_features_list.append(class_token)
                        all_image_labels_list.append(torch.tensor(label, dtype=torch.float32))
                        all_image_names_list.append(img_name)
                        processed_count += 1
                        current_processed += 1
                        processed_images.add(img_idx)

                        if attr_key not in attribute_features_dict:
                            attribute_features_dict[attr_key] = []
                            attribute_image_names_dict[attr_key] = []
                        attribute_features_dict[attr_key].append(class_token)
                        attribute_image_names_dict[attr_key].append(img_name)

                except Exception as e:
                    error_count += 1
                    continue

    if len(all_image_features_list) == 0:
        return None

    image_features = torch.stack(all_image_features_list)
    image_labels = torch.stack(all_image_labels_list)

    for attr_key in attribute_features_dict:
        attribute_features_dict[attr_key] = torch.stack(attribute_features_dict[attr_key])

    all_features = torch.cat([image_features, text_features.cpu()], dim=0)
    diag_labels = torch.eye(len(text_lines), dtype=torch.float32)
    all_labels = torch.cat([image_labels, diag_labels], dim=0)

    data = {
        "features": all_features,
        "labels": all_labels,
        "text_features": text_features.cpu(),
        "image_features": image_features,
        "image_labels": image_labels,
        "image_names": all_image_names_list,
        "text_lines": text_lines,
        "attribute_features": attribute_features_dict,
        "attribute_image_names": attribute_image_names_dict,
        "attribute_keys": attribute_keys,
        "attribute_sample_counts": attribute_sample_counts,
        "attribute_available_counts": attribute_available_counts,
        "sampling_config": {
            "samples_per_attribute": samples_per_attribute,
            "random_seed": random_seed,
            "blur_threshold": blur_threshold,
            "enable_blur_detection": enable_blur_detection
        },
        "model_info": {
            "clip_model": clip_model_name,
            "feature_dim": feature_dim,
            "num_images": len(all_image_features_list),
            "num_texts": len(text_lines)
        }
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(data, output_path)
   
    for attr_idx in range(len(text_lines)):
        attr_key = attribute_keys[attr_idx]
        available_count = attribute_available_counts.get(attr_idx, 0)
        sampled_count = attribute_sample_counts.get(attr_idx, 0)
        actual_count = len(attribute_features_dict.get(attr_key, []))
    
    return data


if __name__ == "__main__":
    pkl_path = "/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb61/DATA/pad.pkl"
    entity_text_path = '/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb61/DATA/KGPAR/dataset/hy/pa100k.txt'
    output_path = '/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb61/DATA/KGPAR/dataset/hy/pa100k_unified_clip_features_10.pt'

    result = unified_clip_feature_extraction(
        pkl_path=pkl_path,
        entity_text_path=entity_text_path,
        output_path=output_path,
        clip_model_name="ViT-L/14",
        download_root='/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb61/DATA/KGPAR',
        samples_per_attribute=10,
        random_seed=42,
        blur_threshold=100.0,
        enable_blur_detection=True
    )