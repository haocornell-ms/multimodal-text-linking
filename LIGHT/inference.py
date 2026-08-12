import os
import sys
import json
import argparse
import numpy as np
import yaml
from typing import Optional, Tuple, Union
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset.dataset import LinkingTestDataset
from dataset.buildin import DATASET_META
from models.text_linking import LightTextLinking
from models.model_utils import get_processors


def group_successors(elements, successors):
    """
    Merge the elements that point to the same successor (ordered by indices)
    """
    ele2suc_mapping = {}
    suc2ele_mapping = {}
    for element, successor in zip(elements, successors):
        ele2suc_mapping[element] = successor
        if element != successor:
            suc2ele_mapping[successor] = element
    
    groups = []
    seen = set()
    def find_path(x):
        path = []
        local_seen = set()
        path = [x]
        while x not in local_seen and x not in seen:  # Keep following until self-loop
            if suc2ele_mapping.get(x) is not None:
                path.insert(0, suc2ele_mapping[x])
                local_seen.add(x)
                x = suc2ele_mapping[x]
            else:
                break
        return path

    # Iterate through all elements and assign them to groups
    for elem in elements:
        if elem not in seen:
            if ele2suc_mapping[elem] == elem:            
                path = find_path(elem)
                groups.append(path)
                for p in path:
                    seen.add(p)

    return groups


def group_successors_with_probability(words, probabilities, bi_probabilities):
    """
    Groups elements based on successor relationships based on probabilities.
    """
    word2succ = {}
    succ2word = {}
    
    iteration = 0
    while len(word2succ) < len(words):
        for i, word in enumerate(words):
            
            prob = probabilities[i]
            sorted_indices = np.argsort(prob)[::-1]
            j = sorted_indices[0]

            if succ2word.get(j) is None:
                word2succ[i] = j
                succ2word[j] = i
            else:
                old_i = succ2word[j]
                if old_i == i: continue;
                elif old_i == j:
                    word2succ[i] = j
                    succ2word[j] = i
                elif i == j:
                    word2succ[i] = j
                elif bi_probabilities[j][i] > bi_probabilities[j][old_i]:
                    word2succ[i] = j
                    succ2word[j] = i
                    word2succ.pop(old_i)
                    probabilities[old_i][j] = 0.
                    bi_probabilities[j][old_i] = 0.
                else:
                    probabilities[i][j] = 0.
                    bi_probabilities[j][i] = 0.
        
    new_successors = [word2succ[i] for i, word in enumerate(words)]
    return group_successors(words, new_successors)


def top_k_successor_predictions(words, polygons, word_ids, probabilities,
                                bi_probabilities, top_k=3):
    """Return top successor candidates with source and target polygons."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    # ``word_ids`` contains only words retained after tokenizer truncation;
    # ``words`` is the original, potentially longer list.
    num_words = len(word_ids)
    if len(words) != len(polygons):
        raise ValueError(
            f"Expected one polygon per word, got {len(words)} words and "
            f"{len(polygons)} polygons"
        )
    if probabilities.shape != (num_words, num_words):
        raise ValueError(
            f"Expected a {num_words}x{num_words} probability matrix, "
            f"got {probabilities.shape}"
        )

    top_k = min(top_k, num_words)
    predictions = []
    for source_index in range(num_words):
        target_indices = np.argsort(probabilities[source_index])[::-1][:top_k]
        candidates = []
        for target_index in target_indices:
            target_index = int(target_index)
            target_word_id = int(word_ids[target_index])
            candidates.append({
                "target_text": words[target_word_id],
                "target_vertices": polygons[target_word_id],
                "probability": float(probabilities[source_index, target_index]),
                "reverse_probability": float(
                    bi_probabilities[target_index, source_index]
                ),
                "is_self": target_index == source_index,
            })

        source_word_id = int(word_ids[source_index])
        predictions.append({
            "source_text": words[source_word_id],
            "source_vertices": polygons[source_word_id],
            "top_successors": candidates,
        })

    return predictions


def collect_visual_contributions(sample_data, full_logits, no_visual_logits):
    """Compare each successor decision with and without word-level visuals."""
    full_probabilities = torch.softmax(full_logits, dim=-1)
    no_visual_probabilities = torch.softmax(no_visual_logits, dim=-1)
    full_targets = full_logits.argmax(dim=-1)
    no_visual_targets = no_visual_logits.argmax(dim=-1)
    word_ids = sample_data['ori_word_ids']
    words = sample_data['ori_words']
    visual_mask = sample_data['word_visual_mask'].bool()

    records = []
    for source_index in range(full_logits.shape[0]):
        target_index = int(full_targets[source_index])
        no_visual_target_index = int(no_visual_targets[source_index])
        source_word_id = int(word_ids[source_index])
        target_word_id = int(word_ids[target_index])
        no_visual_target_word_id = int(word_ids[no_visual_target_index])
        records.append({
            "image": sample_data['image_name'],
            "source_index": source_index,
            "source_text": words[source_word_id],
            "has_visual": bool(visual_mask[source_index]) if source_index < len(visual_mask) else False,
            "full_target_index": target_index,
            "full_target_text": words[target_word_id],
            "no_visual_target_index": no_visual_target_index,
            "no_visual_target_text": words[no_visual_target_word_id],
            "target_changed": target_index != no_visual_target_index,
            # Positive values mean visual inputs increased support for the
            # target selected by the full model.
            "selected_logit_delta": float(
                full_logits[source_index, target_index]
                - no_visual_logits[source_index, target_index]
            ),
            "selected_probability_delta": float(
                full_probabilities[source_index, target_index]
                - no_visual_probabilities[source_index, target_index]
            ),
            "full_probability": float(full_probabilities[source_index, target_index]),
            "no_visual_probability": float(
                no_visual_probabilities[source_index, target_index]
            ),
        })
    return records


def print_visual_contribution_summary(records, max_examples):
    if not records:
        print("Visual contribution stats: no word predictions were available.")
        return
    changed = sum(record["target_changed"] for record in records)
    visual_records = [record for record in records if record["has_visual"]]
    mean_abs_logit_delta = np.mean([
        abs(record["selected_logit_delta"]) for record in records
    ])
    mean_abs_probability_delta = np.mean([
        abs(record["selected_probability_delta"]) for record in records
    ])
    print("\nVisual contribution summary (full model vs word visuals disabled)")
    print(f"  words: {len(records)} ({len(visual_records)} with a valid visual crop)")
    print(f"  top-1 target changed: {changed}/{len(records)} ({changed / len(records):.2%})")
    print(f"  mean |selected logit delta|: {mean_abs_logit_delta:.6f}")
    print(f"  mean |selected probability delta|: {mean_abs_probability_delta:.6f}")

    ranked_records = sorted(
        records,
        key=lambda record: abs(record["selected_probability_delta"]),
        reverse=True,
    )[:max_examples]
    print(f"  largest {len(ranked_records)} per-word effects:")
    for record in ranked_records:
        print(
            f"    {record['image']} | {record['source_text']!r}: "
            f"{record['no_visual_target_text']!r} -> {record['full_target_text']!r}; "
            f"delta_logit={record['selected_logit_delta']:+.4f}, "
            f"delta_prob={record['selected_probability_delta']:+.4f}"
        )


def main():
    # python inference.py --test_dataset test --out_file lithium.json --model_dir _runs/best/ --anno_path /home/yaoyi/shared/critical-maas/12month-text-extraction/spot/lithium.json  --img_dir /home/yaoyi/shared/critical-maas/12month-text-extraction/img_crops/lithium
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--test_dataset', type=str)
    parser.add_argument('--out_file', type=str)
    parser.add_argument('--anno_path', type=str)
    parser.add_argument('--img_dir', type=str)
    parser.add_argument(
        '--prob_out_file', type=str, default=None,
        help='Optional JSON file for per-word top-k successor probabilities.'
    )
    parser.add_argument(
        '--top_k', type=int, default=3,
        help='Number of successor candidates dumped per word (default: 3).'
    )
    parser.add_argument(
        '--visual_contribution_stats', action='store_true',
        help='Compare predictions with word-level visual features enabled and disabled.'
    )
    parser.add_argument(
        '--visual_contribution_examples', type=int, default=20,
        help='Number of strongest visual effects printed (default: 20).'
    )
    parser.add_argument(
        '--disable_word_visuals', action='store_true',
        help='Disable word-crop/style inputs for the main saved predictions.'
    )
    args, remaining_args = parser.parse_known_args()
    
    with open(os.path.join(args.model_dir, 'config.yaml'), 'r') as f:
        config = yaml.safe_load(f)

    for key, value in config.items():
        parser.add_argument(f'--{key}', type=type(value), default=value)
    args = parser.parse_args()
    print(
        "Inference visual settings: "
        f"use_word_style={getattr(args, 'use_word_style', False)}, "
        f"use_pairwise_relations={getattr(args, 'use_pairwise_relations', False)}, "
        f"preserve_text_stop_scores={getattr(args, 'preserve_text_stop_scores', False)}, "
        f"use_factorized_linking={getattr(args, 'use_factorized_linking', False)}, "
        f"disable_word_visuals={args.disable_word_visuals}"
    )

    ### Load model ###
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = os.path.join(args.model_dir, 'best_model.pth')
    state_dict = torch.load(checkpoint_path, map_location=device)

    model = LightTextLinking(args)
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)
    model.to(device)
    
    ### Load data ###
    args.test_data_shuffle=False
    tokenizer, image_processor = get_processors(args.pretrained_model_name)
    test_dataset = LinkingTestDataset(tokenizer, image_processor, args, mode='test', return_ori=True)

    model.eval()
    with torch.no_grad():
        result_list = []
        probability_result_list = []
        visual_contribution_records = []
        for i in tqdm(range(len(test_dataset)), total=len(test_dataset)):
            sample_data = test_dataset[i]
            if sample_data is None:
                continue
                
            image_name = sample_data['image_name']
            input_data = sample_data.copy()
            input_data.pop('image_name')
            input_data.pop('ori_words')
            input_data.pop('ori_polygons')
            input_data.pop('ori_word_ids')
            input_data = {k: v.unsqueeze(0).to(device) for k, v in input_data.items()}
            if args.disable_word_visuals:
                input_data['word_visual_mask'] = torch.zeros_like(
                    input_data['word_visual_mask']
                )
            all_logits, _ = model(input_data, return_loss=False)

            if args.visual_contribution_stats:
                no_visual_input_data = input_data.copy()
                no_visual_input_data['word_visual_mask'] = torch.zeros_like(
                    input_data['word_visual_mask']
                )
                no_visual_logits, _ = model(no_visual_input_data, return_loss=False)
                visual_contribution_records.extend(collect_visual_contributions(
                    sample_data,
                    all_logits['logits'][0],
                    no_visual_logits['logits'][0],
                ))
            
            probabilities = torch.softmax(all_logits['logits'][0], dim=-1)
            bi_probabilities = torch.softmax(all_logits['bi_logits'][0], dim=-1)
            probabilities = probabilities.detach().cpu().numpy()
            bi_probabilities = bi_probabilities.detach().cpu().numpy()

            if args.prob_out_file:
                probability_result_list.append({
                    "image": image_name,
                    "words": top_k_successor_predictions(
                        sample_data['ori_words'],
                        sample_data['ori_polygons'],
                        sample_data['ori_word_ids'],
                        probabilities,
                        bi_probabilities,
                        args.top_k,
                    ),
                })
            
            groups = group_successors_with_probability(sample_data['ori_word_ids'], 
                                                       probabilities.copy(),
                                                       bi_probabilities.copy())
                
            ori_words = sample_data['ori_words']
            ori_polygons = sample_data['ori_polygons']
            ori_word_ids = sample_data['ori_word_ids']

            if 'MapText' in args.test_dataset:
                result = {"image": "rumsey/test/" + sample_data['image_name'], "groups": []}
            # elif 'HierText' in args.test_dataset:
            #     result = {"image": sample_data['image_name'].split('.')[0], "groups": []}
            # elif 'IGN' in args.test_dataset:
            #     result = {"image": "ign/test/" + sample_data['image_name'], "groups": []}
            else:
                result = {"image": image_name, "groups": []}
                
            for group in groups:
                group_items = []
                for idx in group:
                    item_dict = {'text': ori_words[idx], 'vertices': ori_polygons[idx]}
                    group_items.append(item_dict)
                result['groups'].append(group_items)

            result_list.append(result)    

        with open(os.path.join(args.model_dir, args.out_file), 'w') as f:
            json.dump(result_list, f, indent=4)

        if args.prob_out_file:
            probability_output_path = os.path.join(
                args.model_dir, args.prob_out_file
            )
            with open(probability_output_path, 'w') as f:
                json.dump(probability_result_list, f, indent=4)
            print(f"Saved top-{args.top_k} probabilities to {probability_output_path}")

        if args.visual_contribution_stats:
            print_visual_contribution_summary(
                visual_contribution_records,
                args.visual_contribution_examples,
            )

    if 'MapText' in args.test_dataset:
        gt_path = DATASET_META['MapText_test']['anno_path']
    # if 'HierText' in args.test_dataset:
    #     gt_path = DATASET_META['HierText_test']['anno_path']
    # if 'IGN' in args.test_dataset:
    #     gt_path = DATASET_META['IGN_test']['anno_path']
    print(f"python evaluation/eval.py --gt {gt_path} --task detrecedges --pred {os.path.join(args.model_dir, args.out_file)}")
    os.system(f"python evaluation/eval.py --gt {gt_path} --task detrecedges --pred {os.path.join(args.model_dir, args.out_file)}")

if __name__ == '__main__':
    main()
