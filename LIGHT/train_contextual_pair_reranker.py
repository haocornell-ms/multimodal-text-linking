import argparse
import json
import math
import os
from functools import lru_cache

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from analyze_topk_headroom import build_ground_truth, image_key, match_item
from train_candidate_verifier import candidate_target_index, text_features


def polygon_array(vertices):
    return np.asarray(vertices, dtype=np.float32).reshape(-1, 2)


def long_axis_angle(polygon):
    (_, _), (width, height), angle = cv2.minAreaRect(
        polygon.astype(np.float32)
    )
    if width < height:
        angle += 90.0
    return math.radians(angle)


def polygon_height(polygon):
    (_, _), (width, height), _ = cv2.minAreaRect(polygon.astype(np.float32))
    return max(min(float(width), float(height)), 1.0)


def extract_context_pair_crop(image, source_vertices, target_vertices,
                              output_height=96, output_width=256,
                              context_scale=1.5):
    """Smallest landscape crop covering both words without rotating pixels."""
    rgb = np.asarray(image.convert('RGB'), dtype=np.uint8)
    source = polygon_array(source_vertices)
    target = polygon_array(target_vertices)
    union = np.concatenate((source, target), axis=0)
    x0, y0 = union.min(axis=0)
    x1, y1 = union.max(axis=0)
    margin = context_scale * max(
        polygon_height(source), polygon_height(target)
    )
    x0, y0, x1, y1 = x0 - margin, y0 - margin, x1 + margin, y1 + margin
    width, height = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    desired_ratio = output_width / output_height
    if width / height < desired_ratio:
        required_width = desired_ratio * height
        center_x = 0.5 * (x0 + x1)
        x0, x1 = center_x - 0.5 * required_width, center_x + 0.5 * required_width

    left, top = int(math.floor(x0)), int(math.floor(y0))
    right, bottom = int(math.ceil(x1)), int(math.ceil(y1))
    crop_width, crop_height = max(right - left, 1), max(bottom - top, 1)
    crop = np.full((crop_height, crop_width, 3), 255, dtype=np.uint8)
    source_mask = np.zeros((crop_height, crop_width), dtype=np.uint8)
    target_mask = np.zeros((crop_height, crop_width), dtype=np.uint8)
    image_height, image_width = rgb.shape[:2]
    src_left, src_top = max(left, 0), max(top, 0)
    src_right, src_bottom = min(right, image_width), min(bottom, image_height)
    if src_right > src_left and src_bottom > src_top:
        crop[
            src_top - top:src_bottom - top,
            src_left - left:src_right - left,
        ] = rgb[src_top:src_bottom, src_left:src_right]
    offset = np.asarray((left, top), dtype=np.float32)
    cv2.fillPoly(source_mask, [np.round(source - offset).astype(np.int32)], 255)
    cv2.fillPoly(target_mask, [np.round(target - offset).astype(np.int32)], 255)

    scale = min(output_width / crop_width, output_height / crop_height)
    resized_width = max(1, int(round(crop_width * scale)))
    resized_height = max(1, int(round(crop_height * scale)))
    resized_rgb = cv2.resize(
        crop, (resized_width, resized_height), interpolation=cv2.INTER_AREA
    )
    resized_source = cv2.resize(
        source_mask, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST
    )
    resized_target = cv2.resize(
        target_mask, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST
    )
    rgb_canvas = np.full((output_height, output_width, 3), 255, dtype=np.uint8)
    source_canvas = np.zeros((output_height, output_width), dtype=np.uint8)
    target_canvas = np.zeros((output_height, output_width), dtype=np.uint8)
    paste_x = (output_width - resized_width) // 2
    paste_y = (output_height - resized_height) // 2
    region = np.s_[paste_y:paste_y + resized_height, paste_x:paste_x + resized_width]
    rgb_canvas[region] = resized_rgb
    source_canvas[region] = resized_source
    target_canvas[region] = resized_target
    channels = np.concatenate((
        rgb_canvas.astype(np.float32).transpose(2, 0, 1) / 255.0,
        source_canvas[None].astype(np.float32) / 255.0,
        target_canvas[None].astype(np.float32) / 255.0,
    ), axis=0)
    return torch.from_numpy(channels)


def context_numeric_features(source_word, target_word, candidate, rank):
    source_polygon = polygon_array(source_word['source_vertices'])
    target_polygon = polygon_array(target_word['source_vertices'])
    source_center = source_polygon.mean(axis=0)
    target_center = target_polygon.mean(axis=0)
    source_height = polygon_height(source_polygon)
    target_height = polygon_height(target_polygon)
    source_angle = long_axis_angle(source_polygon)
    target_angle = long_axis_angle(target_polygon)
    delta = target_center - source_center
    mean_height = max(0.5 * (source_height + target_height), 1.0)
    ndx, ndy = delta / mean_height
    distance = math.hypot(float(ndx), float(ndy))
    direction_angle = math.atan2(float(delta[1]), float(delta[0]))
    probability = max(float(candidate['probability']), 1e-8)
    reverse_probability = max(float(candidate['reverse_probability']), 1e-8)
    numeric = np.asarray((
        math.log(probability), math.log(reverse_probability),
        probability, reverse_probability, float(rank),
        float(candidate.get('is_self', False)),
        float(ndx), float(ndy), distance,
        math.log(max(target_height / source_height, 1e-6)),
        math.sin(2 * source_angle), math.cos(2 * source_angle),
        math.sin(2 * target_angle), math.cos(2 * target_angle),
        math.sin(2 * (target_angle - source_angle)),
        math.cos(2 * (target_angle - source_angle)),
        math.cos(direction_angle - source_angle),
        math.sin(direction_angle - source_angle),
        math.cos(direction_angle - target_angle),
        math.sin(direction_angle - target_angle),
    ), dtype=np.float32)
    return np.concatenate((
        numeric,
        text_features(source_word['source_text'], target_word['source_text']),
    ))


class ContextPairReranker(nn.Module):
    def __init__(self, numeric_dim):
        super().__init__()
        self.crop_encoder = nn.Sequential(
            nn.Conv2d(5, 24, 5, stride=2, padding=2), nn.BatchNorm2d(24), nn.GELU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.BatchNorm2d(48), nn.GELU(),
            nn.Conv2d(48, 96, 3, stride=2, padding=1), nn.BatchNorm2d(96), nn.GELU(),
            nn.Conv2d(96, 128, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, 96), nn.GELU(),
        )
        self.numeric_encoder = nn.Sequential(
            nn.Linear(numeric_dim, 96), nn.LayerNorm(96), nn.GELU(),
            nn.Linear(96, 64), nn.GELU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(160, 96), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(96, 1),
        )

    def forward(self, crops, numeric):
        return self.scorer(torch.cat((
            self.crop_encoder(crops), self.numeric_encoder(numeric)
        ), dim=-1)).squeeze(-1)


@lru_cache(maxsize=16)
def load_image(path):
    return Image.open(path).convert('RGB')


class ContextCandidateDataset(torch.utils.data.Dataset):
    def __init__(self, groups, images, numeric_mean, numeric_std,
                 crop_height, crop_width, context_scale):
        self.groups = groups
        self.images = images
        self.numeric_mean = numeric_mean
        self.numeric_std = numeric_std
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.context_scale = context_scale

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        group = self.groups[index]
        image_record = self.images[group['image_index']]
        image = load_image(image_record['path'])
        source = image_record['words'][group['source_index']]
        crops, numeric = [], []
        for candidate in group['candidates']:
            target = image_record['words'][candidate['target_index']]
            crops.append(extract_context_pair_crop(
                image, source['source_vertices'], target['source_vertices'],
                self.crop_height, self.crop_width, self.context_scale,
            ))
            numeric.append(candidate['numeric'])
        numeric = (np.asarray(numeric) - self.numeric_mean) / self.numeric_std
        return (
            torch.stack(crops), torch.from_numpy(numeric.astype(np.float32)),
            torch.tensor(group['target_rank'], dtype=torch.long),
            torch.tensor(group['baseline_correct'], dtype=torch.bool),
        )


def build_groups(prediction_path, annotation_path, image_dir, training=False,
                 hard_margin=0.15):
    with open(prediction_path) as file:
        predictions = json.load(file)
    with open(annotation_path) as file:
        truth = build_ground_truth(json.load(file))
    images, groups, numeric_values = [], [], []
    for prediction in predictions:
        words = prediction['words']
        key = image_key(prediction['image'])
        image_index = len(images)
        images.append({'path': os.path.join(image_dir, key), 'words': words})
        image_truth = truth[key]
        for source_index, word in enumerate(words):
            source_truth = match_item(
                word['source_text'], word['source_vertices'], image_truth
            )
            true_center = np.asarray(source_truth['target_center'])
            candidates, target_rank = [], -1
            for rank, candidate in enumerate(word['top_successors'][:3]):
                target_index = candidate_target_index(candidate, words)
                target_truth = match_item(
                    candidate['target_text'], candidate['target_vertices'], image_truth
                )
                if np.linalg.norm(np.asarray(target_truth['center']) - true_center) < 1e-3:
                    target_rank = rank
                numeric = context_numeric_features(
                    word, words[target_index], candidate, rank
                )
                numeric_values.append(numeric)
                candidates.append({
                    'target_index': target_index,
                    'numeric': numeric,
                })
            if len(candidates) != 3:
                continue
            probability_margin = (
                float(word['top_successors'][0]['probability'])
                - float(word['top_successors'][1]['probability'])
            )
            keep = not training or (
                target_rank >= 0 and (
                    target_rank != 0 or probability_margin < hard_margin
                    or (source_index + image_index) % 10 == 0
                )
            )
            if keep:
                groups.append({
                    'image_index': image_index,
                    'source_index': source_index,
                    'candidates': candidates,
                    'target_rank': target_rank,
                    'baseline_correct': target_rank == 0,
                })
    return images, groups, np.asarray(numeric_values, dtype=np.float32)


def score_dataset(model, loader, device):
    scores, targets, baseline_correct = [], [], []
    model.eval()
    with torch.no_grad():
        for crops, numeric, target, baseline in loader:
            batch_size, candidates = crops.shape[:2]
            logits = model(
                crops.flatten(0, 1).to(device),
                numeric.flatten(0, 1).to(device),
            ).view(batch_size, candidates)
            scores.append(torch.softmax(logits, dim=1).cpu())
            targets.append(target)
            baseline_correct.append(baseline)
    return (
        torch.cat(scores).numpy(), torch.cat(targets).numpy(),
        torch.cat(baseline_correct).numpy(),
    )


def calibrate(scores, targets, baseline_correct):
    best = None
    selected = scores.argmax(axis=1)
    rows = np.arange(len(scores))
    for threshold in np.linspace(0.4, 0.9, 11):
        for margin in np.linspace(0.0, 0.4, 17):
            intervene = (
                (selected != 0)
                & (scores[rows, selected] >= threshold)
                & (scores[rows, selected] - scores[:, 0] >= margin)
            )
            choices = np.where(intervene, selected, 0)
            accuracy = np.mean(choices == targets)
            changes = int(intervene.sum())
            result = (accuracy, -changes, threshold, margin, changes)
            if best is None or result > best:
                best = result
    baseline_accuracy = baseline_correct.mean()
    return {
        'baseline_accuracy': float(baseline_accuracy),
        'accuracy': float(best[0]),
        'threshold': float(best[2]),
        'margin': float(best[3]),
        'interventions': int(best[4]),
        'groups': int(len(targets)),
    }


def main():
    parser = argparse.ArgumentParser()
    for split in ('train', 'val'):
        parser.add_argument(f'--{split}-predictions', required=True)
        parser.add_argument(f'--{split}-gt', required=True)
        parser.add_argument(f'--{split}-images', required=True)
    parser.add_argument('--model-out', required=True)
    parser.add_argument('--report-out', required=True)
    parser.add_argument('--crop-height', type=int, default=96)
    parser.add_argument('--crop-width', type=int, default=256)
    parser.add_argument('--context-scale', type=float, default=1.5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    train_images, train_groups, train_numeric = build_groups(
        args.train_predictions, args.train_gt, args.train_images, training=True
    )
    val_images, val_groups, _ = build_groups(
        args.val_predictions, args.val_gt, args.val_images
    )
    numeric_mean = train_numeric.mean(axis=0)
    numeric_std = train_numeric.std(axis=0).clip(min=1e-5)
    train_dataset = ContextCandidateDataset(
        train_groups, train_images, numeric_mean, numeric_std,
        args.crop_height, args.crop_width, args.context_scale,
    )
    val_dataset = ContextCandidateDataset(
        val_groups, val_images, numeric_mean, numeric_std,
        args.crop_height, args.crop_width, args.context_scale,
    )
    generator = torch.Generator().manual_seed(1234)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, generator=generator, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(1234)
    model = ContextPairReranker(train_numeric.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    best_state, best_report, stale = None, None, 0
    print(f'Training groups: {len(train_groups):,}; validation groups: {len(val_groups):,}')
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        batches = 0
        for crops, numeric, target, _ in train_loader:
            valid = target >= 0
            if not valid.any():
                continue
            batch_size, candidates = crops.shape[:2]
            logits = model(
                crops.flatten(0, 1).to(device),
                numeric.flatten(0, 1).to(device),
            ).view(batch_size, candidates)
            loss = F.cross_entropy(logits[valid.to(device)], target[valid].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss)
            batches += 1
        scores, targets, baseline = score_dataset(model, val_loader, device)
        report = calibrate(scores, targets, baseline)
        print(
            f'Epoch {epoch + 1}: loss={epoch_loss / max(batches, 1):.6f}, '
            f'baseline={report["baseline_accuracy"]:.4%}, '
            f'accuracy={report["accuracy"]:.4%}, '
            f'interventions={report["interventions"]:,}'
        )
        if best_report is None or report['accuracy'] > best_report['accuracy'] + 1e-5:
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_report = report
            stale = 0
        else:
            stale += 1
        if stale >= 6:
            break
    model.load_state_dict(best_state)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'numeric_dim': int(train_numeric.shape[1]),
        'numeric_mean': numeric_mean,
        'numeric_std': numeric_std,
        'crop_height': args.crop_height,
        'crop_width': args.crop_width,
        'context_scale': args.context_scale,
        'report': best_report,
    }
    torch.save(checkpoint, args.model_out)
    with open(args.report_out, 'w') as file:
        json.dump(best_report, file, indent=2)
    print(json.dumps(best_report, indent=2))


if __name__ == '__main__':
    main()
