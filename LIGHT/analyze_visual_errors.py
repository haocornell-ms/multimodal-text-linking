import argparse
import csv
import json
import math
import os
from collections import Counter

from PIL import Image, ImageDraw


def image_key(name):
    return os.path.basename(name)


def normalized_text(text):
    return ''.join(str(text).lower().split())


def points(vertices):
    if vertices and not isinstance(vertices[0], (list, tuple)):
        return list(zip(vertices[::2], vertices[1::2]))
    return vertices


def centroid(vertices):
    polygon = points(vertices)
    return (
        sum(float(point[0]) for point in polygon) / len(polygon),
        sum(float(point[1]) for point in polygon) / len(polygon),
    )


def build_truth(annotation):
    truth = {}
    for image in annotation:
        nodes = []
        edges = set()
        for group_index, group in enumerate(image['groups']):
            retained = [item for item in group if not item.get('illegible', False)]
            group_ids = []
            for item_index, item in enumerate(retained):
                node_id = f'{group_index}:{item_index}'
                center = centroid(item['vertices'])
                nodes.append({
                    'id': node_id,
                    'text': normalized_text(item.get('text', '')),
                    'raw_text': item.get('text', ''),
                    'center': center,
                    'vertices': item['vertices'],
                })
                group_ids.append(node_id)
            edges.update(
                frozenset((group_ids[index], group_ids[index + 1]))
                for index in range(len(group_ids) - 1)
            )
        truth[image_key(image['image'])] = {
            'nodes': nodes,
            'node_by_id': {node['id']: node for node in nodes},
            'edges': edges,
        }
    return truth


def match_node(item, nodes):
    center = centroid(item['vertices'])
    text = normalized_text(item.get('text', ''))
    same_text = [node for node in nodes if node['text'] == text]
    candidates = same_text or nodes
    return min(candidates, key=lambda node: math.dist(center, node['center']))


def prediction_edges(prediction, truth_image):
    edges = set()
    matched_groups = []
    for group in prediction['groups']:
        node_ids = [
            match_node(item, truth_image['nodes'])['id'] for item in group
        ]
        matched_groups.append(node_ids)
        edges.update(
            frozenset((node_ids[index], node_ids[index + 1]))
            for index in range(len(node_ids) - 1)
            if node_ids[index] != node_ids[index + 1]
        )
    return edges, matched_groups


def edge_metrics(predicted, target):
    true_positives = len(predicted & target)
    false_positives = len(predicted - target)
    false_negatives = len(target - predicted)
    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        'tp': true_positives,
        'fp': false_positives,
        'fn': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def edge_geometry(edge, node_by_id):
    first, second = (node_by_id[node_id] for node_id in edge)
    dx = second['center'][0] - first['center'][0]
    dy = second['center'][1] - first['center'][1]
    distance = math.hypot(dx, dy)
    angle = abs(math.degrees(math.atan2(dy, dx))) % 180
    if angle > 90:
        angle = 180 - angle
    return {'distance': distance, 'angle': angle}


def summarize_geometry(edges, node_by_id):
    values = [edge_geometry(edge, node_by_id) for edge in edges]
    if not values:
        return {'count': 0, 'mean_distance': 0.0, 'mean_angle': 0.0}
    return {
        'count': len(values),
        'mean_distance': sum(value['distance'] for value in values) / len(values),
        'mean_angle': sum(value['angle'] for value in values) / len(values),
    }


def draw_edge(draw, edge, nodes, color, width):
    first, second = (nodes[node_id] for node_id in edge)
    draw.line((first['center'], second['center']), fill=color, width=width)


def render_case(image_path, output_path, record, truth_image):
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    nodes = truth_image['node_by_id']
    for edge in truth_image['edges']:
        draw_edge(draw, edge, nodes, (150, 150, 150), 2)
    for edge in record['removed_false_positives']:
        draw_edge(draw, edge, nodes, (40, 120, 255), 5)
    for edge in record['fixed_true_edges']:
        draw_edge(draw, edge, nodes, (30, 200, 70), 6)
    for edge in record['broken_true_edges']:
        draw_edge(draw, edge, nodes, (230, 30, 30), 6)
    for edge in record['added_false_positives']:
        draw_edge(draw, edge, nodes, (255, 145, 20), 6)
    legend = [
        ('GT', (150, 150, 150)),
        ('fixed', (30, 200, 70)),
        ('broken', (230, 30, 30)),
        ('removed FP', (40, 120, 255)),
        ('added FP', (255, 145, 20)),
    ]
    y = 8
    for label, color in legend:
        draw.rectangle((8, y, 28, y + 12), fill=color)
        draw.text((34, y), label, fill=(0, 0, 0), stroke_width=2,
                  stroke_fill=(255, 255, 255))
        y += 18
    image.thumbnail((1800, 1800))
    image.save(output_path, quality=92)


def serializable_edges(edges):
    return [sorted(edge) for edge in sorted(edges, key=lambda value: sorted(value))]


def analyze(annotation, baseline_predictions, visual_predictions, image_dir,
            output_dir, max_images):
    os.makedirs(output_dir, exist_ok=True)
    case_dir = os.path.join(output_dir, 'cases')
    os.makedirs(case_dir, exist_ok=True)
    truth = build_truth(annotation)
    baseline_by_image = {
        image_key(image['image']): image for image in baseline_predictions
    }
    visual_by_image = {
        image_key(image['image']): image for image in visual_predictions
    }
    totals = Counter()
    records = []
    geometry_edges = {
        'fixed_true_edges': [],
        'broken_true_edges': [],
        'removed_false_positives': [],
        'added_false_positives': [],
    }
    for name, truth_image in truth.items():
        if name not in baseline_by_image or name not in visual_by_image:
            totals['missing_images'] += 1
            continue
        baseline_edges, _ = prediction_edges(
            baseline_by_image[name], truth_image
        )
        visual_edges, _ = prediction_edges(visual_by_image[name], truth_image)
        target_edges = truth_image['edges']
        fixed = (visual_edges & target_edges) - baseline_edges
        broken = (baseline_edges & target_edges) - visual_edges
        removed_fp = (baseline_edges - target_edges) - visual_edges
        added_fp = (visual_edges - target_edges) - baseline_edges
        categories = {
            'fixed_true_edges': fixed,
            'broken_true_edges': broken,
            'removed_false_positives': removed_fp,
            'added_false_positives': added_fp,
        }
        for category, edges in categories.items():
            totals[category] += len(edges)
            geometry_edges[category].extend(
                (edge, truth_image['node_by_id']) for edge in edges
            )
        baseline_metrics = edge_metrics(baseline_edges, target_edges)
        visual_metrics = edge_metrics(visual_edges, target_edges)
        records.append({
            'image': name,
            'baseline': baseline_metrics,
            'visual': visual_metrics,
            'delta_f1': visual_metrics['f1'] - baseline_metrics['f1'],
            'net_correct_edges': len(fixed) + len(removed_fp)
                                 - len(broken) - len(added_fp),
            **{key: serializable_edges(value) for key, value in categories.items()},
        })

    geometry_summary = {}
    for category, edge_records in geometry_edges.items():
        values = [edge_geometry(edge, nodes) for edge, nodes in edge_records]
        geometry_summary[category] = {
            'count': len(values),
            'mean_distance': (
                sum(value['distance'] for value in values) / len(values)
                if values else 0.0
            ),
            'mean_angle': (
                sum(value['angle'] for value in values) / len(values)
                if values else 0.0
            ),
        }

    changed = [record for record in records if record['net_correct_edges'] != 0]
    improved = sorted(
        changed, key=lambda record: (record['net_correct_edges'], record['delta_f1']),
        reverse=True,
    )[:max_images]
    worsened = sorted(
        changed, key=lambda record: (record['net_correct_edges'], record['delta_f1'])
    )[:max_images]
    selected = [('improved', record) for record in improved]
    selected += [('worsened', record) for record in worsened]
    for rank, (category, record) in enumerate(selected, 1):
        path = os.path.join(image_dir, record['image'])
        if not os.path.exists(path):
            continue
        output_name = f'{category}_{rank:03d}_{record["image"]}.jpg'
        render_record = {
            key: [frozenset(edge) for edge in record[key]]
            for key in geometry_edges
        }
        render_case(
            path, os.path.join(case_dir, output_name), render_record,
            truth[record['image']],
        )

    with open(os.path.join(output_dir, 'visual_error_analysis.json'), 'w') as file:
        json.dump({
            'totals': dict(totals),
            'geometry': geometry_summary,
            'images': records,
        }, file, indent=2)
    with open(os.path.join(output_dir, 'per_image.csv'), 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=[
            'image', 'delta_f1', 'net_correct_edges', 'baseline_f1', 'visual_f1',
            'fixed', 'broken', 'removed_fp', 'added_fp',
        ])
        writer.writeheader()
        for record in sorted(records, key=lambda value: value['delta_f1']):
            writer.writerow({
                'image': record['image'],
                'delta_f1': record['delta_f1'],
                'net_correct_edges': record['net_correct_edges'],
                'baseline_f1': record['baseline']['f1'],
                'visual_f1': record['visual']['f1'],
                'fixed': len(record['fixed_true_edges']),
                'broken': len(record['broken_true_edges']),
                'removed_fp': len(record['removed_false_positives']),
                'added_fp': len(record['added_false_positives']),
            })
    improved_count = sum(record['delta_f1'] > 0 for record in records)
    worsened_count = sum(record['delta_f1'] < 0 for record in records)
    unchanged_count = len(records) - improved_count - worsened_count
    with open(os.path.join(output_dir, 'README.md'), 'w') as file:
        file.write('# Visual error analysis\n\n')
        file.write(f'- Images analyzed: {len(records)}\n')
        file.write(f'- Images improved: {improved_count}\n')
        file.write(f'- Images worsened: {worsened_count}\n')
        file.write(f'- Images unchanged: {unchanged_count}\n')
        for category in geometry_edges:
            file.write(f'- {category.replace("_", " ")}: {totals[category]}\n')
        file.write('\nColors: gray=ground truth, green=fixed true edge, '
                   'red=broken true edge, blue=removed false positive, '
                   'orange=added false positive.\n')
    print(json.dumps({
        'images': len(records),
        'improved_images': improved_count,
        'worsened_images': worsened_count,
        'unchanged_images': unchanged_count,
        'totals': dict(totals),
        'geometry': geometry_summary,
        'output_dir': output_dir,
    }, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--visual', required=True)
    parser.add_argument('--image-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-images', type=int, default=30)
    args = parser.parse_args()
    with open(args.gt) as file:
        annotation = json.load(file)
    with open(args.baseline) as file:
        baseline_predictions = json.load(file)
    with open(args.visual) as file:
        visual_predictions = json.load(file)
    analyze(
        annotation, baseline_predictions, visual_predictions, args.image_dir,
        args.output_dir, args.max_images,
    )


if __name__ == '__main__':
    main()
