import argparse
import json
import math
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from analyze_topk_headroom import build_ground_truth, image_key, match_item
from dataset.dataset import extract_word_visual_features


class PairVerifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(64, 1),
        )

    def forward(self, features):
        return self.network(features).squeeze(-1)


def center(vertices):
    points = np.asarray(vertices, dtype=np.float32).reshape(-1, 2)
    return points.mean(axis=0)


def text_features(source, target):
    source = str(source)
    target = str(target)
    functions = (
        str.isupper, str.islower, str.isdigit,
    )
    values = [len(source), len(target), abs(len(source) - len(target))]
    for function in functions:
        values.extend((float(function(source)), float(function(target))))
    values.extend((
        float(source[-1:].isalnum()), float(target[:1].isalnum()),
        float(source[-1:] in '.-'), float(target[-1:] in '.-'),
    ))
    return np.asarray(values, dtype=np.float32)


def pair_geometry(source, target):
    eps = 1e-6
    dx = target[0] - source[0]
    dy = target[1] - source[1]
    mean_height = max(0.5 * (source[3] + target[3]), eps)
    ndx, ndy = dx / mean_height, dy / mean_height
    distance = math.sqrt(ndx * ndx + ndy * ndy + eps)
    direction_x, direction_y = ndx / distance, ndy / distance
    return np.asarray((
        ndx, ndy, distance, direction_x, direction_y,
        math.log((target[2] + eps) / (source[2] + eps)),
        math.log((target[3] + eps) / (source[3] + eps)),
        math.log((target[4] + eps) / (source[4] + eps)),
        source[5] * target[5] + source[6] * target[6],
        direction_x * source[6] + direction_y * source[5],
        direction_x * target[6] + direction_y * target[5],
    ), dtype=np.float32)


def load_word_features(image_path, words, crop_height=32, crop_width=128):
    image = Image.open(image_path).convert('RGB')
    features = []
    for word in words:
        _, style, geometry = extract_word_visual_features(
            image, word['source_vertices'], crop_height, crop_width
        )
        features.append((style.numpy(), geometry.numpy()))
    return features


def candidate_target_index(candidate, words):
    target_center = center(candidate['target_vertices'])
    same_text = [
        index for index, word in enumerate(words)
        if str(word['source_text']).lower() == str(candidate['target_text']).lower()
    ]
    choices = same_text or range(len(words))
    return min(
        choices,
        key=lambda index: np.linalg.norm(
            center(words[index]['source_vertices']) - target_center
        ),
    )


def candidate_vector(source_index, target_index, candidate, rank, words, word_features):
    source_style, source_geometry = word_features[source_index]
    target_style, target_geometry = word_features[target_index]
    probability = max(float(candidate['probability']), 1e-8)
    reverse_probability = max(float(candidate['reverse_probability']), 1e-8)
    return np.concatenate((
        np.asarray((
            math.log(probability), math.log(reverse_probability),
            probability, reverse_probability, rank,
            float(candidate.get('is_self', source_index == target_index)),
        ), dtype=np.float32),
        source_style, target_style,
        np.abs(source_style - target_style), source_style * target_style,
        source_geometry, target_geometry, np.abs(source_geometry - target_geometry),
        pair_geometry(source_geometry, target_geometry),
        text_features(words[source_index]['source_text'], candidate['target_text']),
    ))


def build_dataset(prediction_path, annotation_path, image_dir):
    with open(prediction_path) as file:
        predictions = json.load(file)
    with open(annotation_path) as file:
        truth = build_ground_truth(json.load(file))
    vectors, labels, groups, baseline_correct = [], [], [], [], []
    group_id = 0
    for image in predictions:
        words = image['words']
        image_truth = truth[image_key(image['image'])]
        image_path = os.path.join(image_dir, image_key(image['image']))
        word_features = load_word_features(image_path, words)
        for source_index, word in enumerate(words):
            source_truth = match_item(
                word['source_text'], word['source_vertices'], image_truth
            )
            true_center = np.asarray(source_truth['target_center'])
            group_labels = []
            for rank, candidate in enumerate(word['top_successors']):
                target_index = candidate_target_index(candidate, words)
                target_truth = match_item(
                    candidate['target_text'], candidate['target_vertices'], image_truth
                )
                is_correct = np.linalg.norm(
                    np.asarray(target_truth['center']) - true_center
                ) < 1e-3
                vectors.append(candidate_vector(
                    source_index, target_index, candidate, rank, words, word_features
                ))
                labels.append(is_correct)
                groups.append(group_id)
                group_labels.append(is_correct)
            baseline_correct.append(bool(group_labels[0]))
            group_id += 1
    return (
        np.asarray(vectors, dtype=np.float32),
        np.asarray(labels, dtype=np.int8),
        np.asarray(groups, dtype=np.int64),
        np.asarray(baseline_correct, dtype=bool),
    )


def evaluate_interventions(probabilities, labels, groups, baseline_correct):
    grouped = defaultdict(list)
    for index, group in enumerate(groups):
        grouped[int(group)].append(index)
    best = None
    for threshold in np.linspace(0.35, 0.9, 12):
        for margin in np.linspace(0.0, 0.3, 13):
            correct = 0
            changes = 0
            for group, indices in grouped.items():
                scores = probabilities[indices]
                selected_local = int(scores.argmax())
                selected = indices[selected_local]
                baseline = indices[0]
                intervene = (
                    selected != baseline
                    and probabilities[selected] >= threshold
                    and probabilities[selected] - probabilities[baseline] >= margin
                )
                choice = selected if intervene else baseline
                correct += int(labels[choice])
                changes += int(intervene)
            accuracy = correct / len(grouped)
            result = (accuracy, -changes, threshold, margin, changes)
            if best is None or result > best:
                best = result
    baseline_accuracy = baseline_correct.mean()
    print(f"Validation groups: {len(grouped):,}")
    print(f"Baseline top-1 accuracy: {baseline_accuracy:.4%}")
    print(f"Best verifier accuracy: {best[0]:.4%}")
    print(f"Absolute change: {best[0] - baseline_accuracy:+.4%}")
    print(
        f"Threshold={best[2]:.3f}, margin={best[3]:.3f}, "
        f"interventions={best[4]:,} ({best[4] / len(grouped):.2%})"
    )
    return {
        'baseline_accuracy': float(baseline_accuracy),
        'accuracy': float(best[0]),
        'threshold': float(best[2]),
        'margin': float(best[3]),
        'interventions': int(best[4]),
    }


def main():
    parser = argparse.ArgumentParser()
    for split in ('train', 'val'):
        parser.add_argument(f'--{split}-predictions', required=True)
        parser.add_argument(f'--{split}-gt', required=True)
        parser.add_argument(f'--{split}-images', required=True)
    parser.add_argument('--model-out', required=True)
    parser.add_argument('--report-out', required=True)
    args = parser.parse_args()

    train = build_dataset(args.train_predictions, args.train_gt, args.train_images)
    val = build_dataset(args.val_predictions, args.val_gt, args.val_images)
    torch.manual_seed(1234)
    feature_mean = train[0].mean(axis=0)
    feature_std = train[0].std(axis=0).clip(min=1e-5)
    train_features = torch.from_numpy((train[0] - feature_mean) / feature_std)
    train_labels = torch.from_numpy(train[1].astype(np.float32))
    val_features = torch.from_numpy((val[0] - feature_mean) / feature_std)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PairVerifier(train_features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    positive_weight = max((train[1] == 0).sum() / max((train[1] == 1).sum(), 1), 1.0)
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, device=device)
    )
    generator = torch.Generator().manual_seed(1234)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_features, train_labels),
        batch_size=2048, shuffle=True, generator=generator,
    )
    best_state, best_accuracy, stale_epochs = None, -1.0, 0
    for epoch in range(100):
        model.train()
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_function(model(features), labels)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            probabilities = torch.sigmoid(model(val_features.to(device))).cpu().numpy()
        report = evaluate_interventions(probabilities, val[1], val[2], val[3])
        if report['accuracy'] > best_accuracy + 1e-5:
            best_accuracy = report['accuracy']
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_report = report
            stale_epochs = 0
        else:
            stale_epochs += 1
        print(f"Epoch {epoch + 1}: verifier best accuracy={best_accuracy:.4%}")
        if stale_epochs >= 10:
            break
    model.load_state_dict(best_state)
    report = best_report
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': train_features.shape[1],
        'feature_mean': feature_mean,
        'feature_std': feature_std,
        'report': report,
    }, args.model_out)
    with open(args.report_out, 'w') as file:
        json.dump(report, file, indent=2)


if __name__ == '__main__':
    main()
