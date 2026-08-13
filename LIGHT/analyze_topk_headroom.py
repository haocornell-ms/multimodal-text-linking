import argparse
import json
import os
import math
from collections import Counter


def polygon_key(vertices, precision=2):
    """Stable polygon identity robust to JSON integer/float representation."""
    points = vertices
    if points and not isinstance(points[0], (list, tuple)):
        points = list(zip(points[::2], points[1::2]))
    return tuple(
        coordinate
        for point in points
        for coordinate in (round(float(point[0]), precision), round(float(point[1]), precision))
    )


def image_key(name):
    return os.path.basename(name)


def centroid(vertices):
    points = vertices
    if points and not isinstance(points[0], (list, tuple)):
        points = list(zip(points[::2], points[1::2]))
    return (
        sum(float(point[0]) for point in points) / len(points),
        sum(float(point[1]) for point in points) / len(points),
    )


def normalized_text(text):
    return ''.join(str(text).lower().split())


def build_ground_truth(annotation):
    images = {}
    for image in annotation:
        items = []
        for group in image['groups']:
            retained = [item for item in group if not item.get('illegible', False)]
            for index, item in enumerate(retained):
                target_item = retained[index + 1] if index + 1 < len(retained) else item
                items.append({
                    'text': normalized_text(item.get('text', '')),
                    'center': centroid(item['vertices']),
                    'target_center': centroid(target_item['vertices']),
                    'is_stop': target_item is item,
                })
        images[image_key(image['image'])] = items
    return images


def match_item(text, vertices, items):
    center = centroid(vertices)
    normalized = normalized_text(text)
    same_text = [item for item in items if item['text'] == normalized]
    candidates = same_text or items
    return min(candidates, key=lambda item: math.dist(center, item['center']))


def analyze(annotation, predictions, max_k):
    truth = build_ground_truth(annotation)
    counts = Counter()
    ranks = Counter()
    missing_images = set()
    for image in predictions:
        image_truth = truth.get(image_key(image['image']))
        if image_truth is None:
            missing_images.add(image_key(image['image']))
            continue
        for word in image['words']:
            source_item = match_item(word['source_text'], word['source_vertices'], image_truth)
            target = source_item['target_center']
            is_stop = source_item['is_stop']
            category = 'stop' if is_stop else 'continue'
            counts['total'] += 1
            counts[category] += 1
            candidate_items = [
                match_item(candidate['target_text'], candidate['target_vertices'], image_truth)
                for candidate in word['top_successors'][:max_k]
            ]
            candidate_keys = [item['center'] for item in candidate_items]
            try:
                rank = candidate_keys.index(target) + 1
            except ValueError:
                rank = None
            if rank is not None:
                ranks[('all', rank)] += 1
                ranks[(category, rank)] += 1
            predicted_stop = bool(word['top_successors'][0].get('is_self', False))
            if predicted_stop == is_stop:
                counts['stop_continue_correct'] += 1
            if candidate_keys and candidate_keys[0] == target:
                counts['top1_correct'] += 1

    print(f"Matched words: {counts['total']:,}")
    print(f"Unmatched sources: {counts['unmatched_source']:,}")
    if missing_images:
        print(f"Missing images: {len(missing_images)}")
    print(f"STOP words: {counts['stop']:,}; CONTINUE words: {counts['continue']:,}")
    print(f"Top-1 target accuracy: {counts['top1_correct'] / max(counts['total'], 1):.4%}")
    print(
        "STOP/CONTINUE accuracy: "
        f"{counts['stop_continue_correct'] / max(counts['total'], 1):.4%}"
    )
    for category, denominator in (
        ('all', counts['total']), ('stop', counts['stop']), ('continue', counts['continue'])
    ):
        cumulative = 0
        values = []
        for k in range(1, max_k + 1):
            cumulative += ranks[(category, k)]
            values.append(f"top-{k}={cumulative / max(denominator, 1):.4%}")
        print(f"{category.upper()} target coverage: " + ", ".join(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', required=True)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--max-k', type=int, default=3)
    args = parser.parse_args()
    with open(args.gt) as file:
        annotation = json.load(file)
    with open(args.predictions) as file:
        predictions = json.load(file)
    analyze(annotation, predictions, args.max_k)


if __name__ == '__main__':
    main()
