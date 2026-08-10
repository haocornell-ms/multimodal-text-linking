import os
import sys
import json
import numpy as np
import random
import re
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from dataset.data_utils import *
from dataset.buildin import DATASET_META
from scipy.spatial import KDTree


def _letterbox_rgb(image, output_height, output_width):
    """Resize an RGB crop without changing its aspect ratio."""
    height, width = image.shape[:2]
    scale = min(output_width / max(width, 1), output_height / max(height, 1))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    canvas = np.full((output_height, output_width, 3), 255, dtype=np.uint8)
    y0 = (output_height - resized_height) // 2
    x0 = (output_width - resized_width) // 2
    canvas[y0:y0 + resized_height, x0:x0 + resized_width] = resized
    return canvas


def _order_quad_points(points):
    """Return quadrilateral corners as top-left, top-right, bottom-right, bottom-left."""
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    coordinate_sum = points.sum(axis=1)
    coordinate_difference = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(coordinate_sum)]
    ordered[2] = points[np.argmax(coordinate_sum)]
    ordered[1] = points[np.argmin(coordinate_difference)]
    ordered[3] = points[np.argmax(coordinate_difference)]
    return ordered


def _extract_oriented_crop(rgb, polygon, context_scale):
    """Rectify a word polygon and return a horizontal, reading-scale crop.

    Using an axis-aligned bounding box makes a near-vertical word occupy only a
    thin sliver after it is fitted into a wide word canvas.  This function uses
    the polygon's minimum-area rectangle, expands it for context, and applies a
    perspective transform before rotating portrait results to landscape.
    """
    rect = cv2.minAreaRect(polygon.astype(np.float32))
    center, (rect_width, rect_height), angle = rect
    rect_width = max(float(rect_width), 1.0) * (1.0 + 2.0 * context_scale)
    rect_height = max(float(rect_height), 1.0) * (1.0 + 2.0 * context_scale)
    expanded_rect = (center, (rect_width, rect_height), angle)
    source = _order_quad_points(cv2.boxPoints(expanded_rect))

    # Keep enough source pixels for small text instead of downsampling early.
    warp_width = max(1, int(round(max(
        np.linalg.norm(source[1] - source[0]),
        np.linalg.norm(source[2] - source[3]),
    ))))
    warp_height = max(1, int(round(max(
        np.linalg.norm(source[3] - source[0]),
        np.linalg.norm(source[2] - source[1]),
    ))))
    destination = np.asarray([
        [0, 0],
        [warp_width - 1, 0],
        [warp_width - 1, warp_height - 1],
        [0, warp_height - 1],
    ], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(source, destination)
    crop = cv2.warpPerspective(
        rgb,
        transform,
        (warp_width, warp_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    if crop.shape[0] > crop.shape[1]:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    return crop, rect


def extract_word_visual_features(image, polygon, output_height, output_width, context_scale=0.25):
    """Return a high-resolution word crop, appearance statistics, and geometry.

    The crop is taken from the original map rather than LayoutLMv3's resized image.
    Descriptors are deliberately simple and deterministic so they can supplement
    the learned crop encoder on small training sets.
    """
    rgb = np.asarray(image, dtype=np.uint8)
    image_height, image_width = rgb.shape[:2]
    polygon = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    x0, y0 = polygon.min(axis=0)
    x1, y1 = polygon.max(axis=0)
    width = max(float(x1 - x0), 1.0)
    height = max(float(y1 - y0), 1.0)
    raw_crop, rect = _extract_oriented_crop(rgb, polygon, context_scale)
    if raw_crop.size == 0:
        raw_crop = np.full((1, 1, 3), 255, dtype=np.uint8)
    crop = _letterbox_rgb(raw_crop, output_height, output_width)

    # Compute statistics before letterboxing so padding does not masquerade as background.
    raw_crop_float = raw_crop.astype(np.float32) / 255.0
    gray = cv2.cvtColor(raw_crop, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    edges = cv2.Canny(raw_crop, 80, 160).astype(np.float32) / 255.0
    border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    raw_height, raw_width = gray.shape
    center = gray[raw_height // 4:max(raw_height // 4 + 1, 3 * raw_height // 4),
                  raw_width // 4:max(raw_width // 4 + 1, 3 * raw_width // 4)]
    angle = float(rect[2])
    if rect[1][0] < rect[1][1]:
        angle += 90.0
    angle = np.deg2rad(angle)

    # 15 appearance/scale descriptors. Size values are map-normalized.
    style = np.asarray([
        *raw_crop_float.mean(axis=(0, 1)).tolist(),
        *raw_crop_float.std(axis=(0, 1)).tolist(),
        float(gray.std()),
        float(edges.mean()),
        float(abs(center.mean() - border.mean())),
        float((gray < max(0.05, border.mean() - 0.15)).mean()),
        width / max(image_width, 1),
        height / max(image_height, 1),
        float(np.log(max(width / height, 1e-4))),
        float(np.sin(angle)),
        float(np.cos(angle)),
    ], dtype=np.float32)
    geometry = np.asarray([
        ((x0 + x1) * 0.5) / max(image_width, 1),
        ((y0 + y1) * 0.5) / max(image_height, 1),
        width / max(image_width, 1),
        height / max(image_height, 1),
        (width * height) / max(image_width * image_height, 1),
        float(np.sin(angle)),
        float(np.cos(angle)),
    ], dtype=np.float32)
    crop_tensor = torch.from_numpy(crop.astype(np.float32) / 255.0).permute(2, 0, 1)
    return crop_tensor, torch.from_numpy(style), torch.from_numpy(geometry)


def save_word_crop(crop, save_root, dataset_name, image_name, word_index, text):
    """Save the exact letterboxed crop passed to the word-style encoder."""
    if not save_root:
        return
    image_stem = os.path.splitext(os.path.basename(image_name))[0]
    safe_text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")[:40] or "empty"
    crop_dir = os.path.join(save_root, dataset_name, image_stem)
    os.makedirs(crop_dir, exist_ok=True)
    crop_array = (
        crop.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1) * 255
    ).round().astype(np.uint8)
    Image.fromarray(crop_array, mode="RGB").save(
        os.path.join(crop_dir, f"{word_index:04d}_{safe_text}.png")
    )


def synthetic_map_data_processor(anno_data, thre=3):    
    out_anno_data = []
    if dataset_name == 'SynthMap_train':
        for anno in anno_data:
            if len(anno['groups']) > thre:
                out_anno_data.append(anno)
    elif dataset_name == 'SynthMap_test':
        out_anno_data = anno_data[1000:1200] # this is just for demoing
    return out_anno_data

def hiertext_data_processor(anno_data):
    out_anno_data = []
    for anno in anno_data:
        if anno['image'] not in ['62aec264ad248f1e', 'd5e1b07d3bf588d7', 'dc30c1e1f7bd87d1']:
            anno['image'] += '.jpg'
            out_anno_data.append(anno)
    return out_anno_data
            
    
class LinkingDataset():
    def __init__(self, dataset_name, anno_path, img_dir, return_labels=False):
        self.dataset_name = dataset_name
        self.anno_path = anno_path
        self.img_dir = img_dir
        self.return_labels = return_labels
        with open(anno_path, 'r') as f:
            anno_data = json.load(f)

        if "SynthMap" in dataset_name:
            anno_data = synthetic_map_data_processor(anno_data)
        if "HierText" in dataset_name:
            anno_data = hiertext_data_processor(anno_data)
            
        self.anno_data = anno_data
        self.generate_group_labels()
        print(f"Dataset `{dataset_name}` contains {len(self.anno_data)} samples.")
        self.unused_indices = list(range(len(self.anno_data)))
        
    def reset(self):
        self.unused_indices = list(range(len(self.anno_data)))

    def generate_group_labels(self):
        group_id_list = [i for i in range(1, 10000) if '0' not in str(i)]
        for anno in self.anno_data:
            for i, group in enumerate(anno['groups']):
                group_id = group_id_list[i]
                seq_id = 1
                for item in group:
                    if item.get('illegible', False) or item.get('truncated', False):                  
                        continue    
                    if self.return_labels: 
                        item['label'] = int(f"{group_id}000{seq_id}")  
                    else:
                        item['label'] = 0
                    seq_id += 1

    def get_length(self):
        return len(self.anno_data)

    def get_item(self, idx, is_random=False):
        if is_random:
            idx = random.choice(self.unused_indices)
            self.unused_indices.remove(idx)

        item = self.anno_data[idx]
        
        if '/' in item['image']:
            image_name = os.path.basename(item['image'])
        elif '.jpg' not in item['image']:
            image_name = item['image'] + '.jpg'
        else:
            image_name = item['image']
            
        image_path = os.path.join(self.img_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        anno = item['groups']
        return image, anno, image_name
    
##################################################################################
##################################################################################
##################################################################################
    
def process_one_sample(
    tokenizer,
    image_processor,
    anno,
    image,
    is_shuffle=True,
    args=None,
    image_name="unknown",
    dataset_name="dataset",
):
    ###
    ### image processor
    image_width, image_height = image.size
    image_features = image_processor(image, return_tensors="pt")
    pixel_values = image_features['pixel_values'][0]

    ###
    ### tokenizer
    max_length = getattr(args, "token_padding_max_length", 1280)
    ##################################################################################
    words, bboxes, labels, polygons, ori_polygons = [], [], [], [], []
    word_crops, word_styles, word_geometries = [], [], []
    for group in anno:
        for item in group:
            if item.get('label') is None: continue;
            poly = np.array(item['vertices']).reshape(-1, 2).astype(float)
            bbox = create_bounding_box(poly)
            if bbox[0] == bbox[2] or bbox[1] == bbox[3]: continue;
                
            bbox_scaled = scale_bounding_box(bbox, 1000/image_width, 1000/image_height)
            bboxes.append(bbox_scaled)
            
            ori_polygons.append(item['vertices'])
            padded_poly = compute_polygon(poly, image_width, image_height, sampling=False)
            padded_poly = pad_sequence(padded_poly, max_length=args.max_len, 
                                       pad_value=args.padding_token_id)
            
            if args.poly_only or len(item["text"]) == 0:
                text = 'NaN'
            else:
                text = item["text"].replace(' ', '')                

            if args.text_only:
                padded_poly = np.zeros_like(padded_poly)         

            poly_tensor = torch.tensor(padded_poly, dtype=torch.float)
            polygons.append(poly_tensor)
            words.append(text)
            labels.append(item['label'])
            crop, style, geometry = extract_word_visual_features(
                image, poly,
                output_height=getattr(args, "word_crop_height", 32),
                output_width=getattr(args, "word_crop_width", 128),
                context_scale=getattr(args, "word_crop_context", 0.25),
            )
            save_word_crop(
                crop,
                getattr(args, "word_crop_save_dir", ""),
                dataset_name,
                image_name,
                len(word_crops),
                text,
            )
            word_crops.append(crop)
            word_styles.append(style)
            word_geometries.append(geometry)
            
    if len(words) == 0 or len(polygons) == 0:
        return None, None
    ##################################################################################
    
    ###
    ### shuffle
    if is_shuffle:
        indices = [i for i in range(len(labels))]
        random.shuffle(indices)
    else:
        pts = [(poly[0], poly[1]) for poly in polygons]
        indices = sorted(range(len(pts)), key=lambda i: (pts[i][1], pts[i][0]))

    words = [words[i] for i in indices]
    labels = [labels[i] for i in indices]
    polygons = [polygons[i] for i in indices]
    bboxes = [bboxes[i] for i in indices]
    ori_polygons = [ori_polygons[i] for i in indices]    
    word_crops = [word_crops[i] for i in indices]
    word_styles = [word_styles[i] for i in indices]
    word_geometries = [word_geometries[i] for i in indices]
    
    ##################################################################################
    ori_data_dict = {
            "image": image,
            "words": words,
            "bboxes": bboxes,
            "polygons": ori_polygons,
            "labels": labels
    }    
    encoded_inputs = tokenizer(
        text=words,
        boxes=bboxes,
        word_labels=labels,
        return_special_tokens_mask=False,
        padding='max_length',
        max_length=max_length,
        truncation=True,
    )
    ##################################################################################
    # input_ids is the token ids starting with CLS token, typically longer than labels
    input_ids = torch.tensor(encoded_inputs['input_ids'], dtype=torch.long)
    ###################################################################################
    # bbox is the text bbox, duplicated for tokens belonging to the same word
    bbox = torch.tensor(encoded_inputs['bbox'], dtype=torch.long)
    bbox = torch.clip(bbox, 0, 1000)
    ###################################################################################
    # labels are at the first token location
    labels = torch.tensor(encoded_inputs['labels'], dtype=torch.long)
    attention_mask = torch.tensor(encoded_inputs['attention_mask'], dtype=torch.long)
    ###################################################################################
    # encoded_inputs.word_ids(): [None, 0, 0, 1, 1, 2, 3, 3, 3, ..., None, None, ...]
    # get first token locations; word_id maps the first token to word indices
    word_ids, first_token_indices = [], [] 
    new_polygons = torch.zeros((input_ids.shape[0], 32), dtype=torch.float)
    
    prev_word_id = -1
    for i, word_id in enumerate(encoded_inputs.word_ids()):
        if word_id is None:
            continue
        if word_id != prev_word_id:
            word_ids.append(word_id)
            first_token_indices.append(i)
            new_polygons[i] = polygons[word_id]
            prev_word_id = word_id
        else:
            new_polygons[i] = polygons[prev_word_id]
        
    first_token_indices = torch.tensor(first_token_indices, dtype=torch.long)
    first_token_indices = F.pad(first_token_indices, (0, max_length-first_token_indices.shape[0]), value=-999)
    ori_data_dict['word_ids'] = word_ids
    max_style_words = getattr(args, "max_style_words", 256)
    crop_height = getattr(args, "word_crop_height", 32)
    crop_width = getattr(args, "word_crop_width", 128)
    padded_crops = torch.zeros((max_style_words, 3, crop_height, crop_width), dtype=torch.float32)
    padded_styles = torch.zeros((max_style_words, 15), dtype=torch.float32)
    padded_geometries = torch.zeros((max_style_words, 7), dtype=torch.float32)
    word_visual_mask = torch.zeros(max_style_words, dtype=torch.bool)
    retained_word_ids = word_ids[:max_style_words]
    for output_index, word_id in enumerate(retained_word_ids):
        padded_crops[output_index] = word_crops[word_id]
        padded_styles[output_index] = word_styles[word_id]
        padded_geometries[output_index] = word_geometries[word_id]
        word_visual_mask[output_index] = True
    output = {
        'input_ids': input_ids,
        'first_token_indices': first_token_indices,
        'bbox': bbox,
        'labels': labels,
        'attention_mask': attention_mask,
        'pixel_values': pixel_values,
        'polygons': new_polygons,
        'word_crops': padded_crops,
        'word_styles': padded_styles,
        'word_geometries': padded_geometries,
        'word_visual_mask': word_visual_mask,
    }
    return output, ori_data_dict


class LinkingTrainDataset(Dataset):
    def __init__(self, tokenizer, image_processor, args):
        assert len(args.train_datasets) == len(args.train_data_probabilities)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.args = args
        self.probabilities = args.train_data_probabilities
        self.shuffle = args.train_data_shuffle
        self.dataset_names = args.train_datasets

        self.datasets = {}
        for i, name in enumerate(self.dataset_names):
            print(f"Load training data: {name}")
            dataset = LinkingDataset(dataset_name=name,
                                     anno_path=DATASET_META[name]['anno_path'], 
                                     img_dir=DATASET_META[name]['img_dir'],
                                     return_labels=True)
            self.datasets[name] = dataset
    
    def __len__(self):
        if 'MapText' in self.dataset_names[0]:
            return 200
        else:
            return 500 

    def reset(self):
        for _, dataset in self.datasets.items():
            dataset.reset()

    def __getitem__(self, idx):
        dataset_name = random.choices(self.dataset_names, self.probabilities)[0]
        image, anno, image_name = self.datasets[dataset_name].get_item(idx, is_random=True)
        output, _ = process_one_sample(self.tokenizer, self.image_processor, 
                                       anno, image, 
                                       self.shuffle, self.args,
                                       image_name=image_name,
                                       dataset_name=dataset_name)
        return output

    
class LinkingTestDataset(Dataset):
    def __init__(self, tokenizer, image_processor, args, mode, return_ori=False):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.args = args
        self.mode = mode
        if self.mode == 'val':
            self.dataset_name = args.val_dataset
            self.shuffle = args.val_data_shuffle
            self.return_labels = True
        else:
            self.dataset_name = args.test_dataset
            self.shuffle = args.test_data_shuffle
            self.return_labels = True

        if self.mode == 'val':
            self.dataset = LinkingDataset(dataset_name=self.dataset_name,
                                          anno_path=DATASET_META[self.dataset_name]['anno_path'], 
                                          img_dir=DATASET_META[self.dataset_name]['img_dir'],
                                          return_labels=self.return_labels)
        else:
            print(f"Load annotations from {args.anno_path}")
            self.dataset = LinkingDataset(dataset_name='test',
                                          anno_path=args.anno_path,
                                          img_dir=args.img_dir,
                                          return_labels=self.return_labels)
        self.return_ori = return_ori
    
    def __len__(self):
        return self.dataset.get_length()
        
    def __getitem__(self, idx):
        image, anno, image_name = self.dataset.get_item(idx)
        output, ori_data_dict = process_one_sample(self.tokenizer, self.image_processor, 
                                                   anno, image, 
                                                   self.shuffle, self.args,
                                                   image_name=image_name,
                                                   dataset_name=self.dataset_name)
        if output is None: return None;
        if self.return_ori:
            output['image_name'] = image_name
            output['ori_words'] = ori_data_dict["words"]
            output['ori_polygons'] = ori_data_dict["polygons"]
            output['ori_word_ids'] = ori_data_dict["word_ids"]
        return output
