from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
from transformers import LayoutLMv3Config
from transformers import BertModel, BertConfig
from transformers import BeitForMaskedImageModeling

from models.losses import NCELoss, FocalLoss
from models.model_utils import MLP, TokenEncoder
from models.light import LightModel


class WordStyleEncoder(nn.Module):
    """Encode an original-resolution word crop and explicit style statistics."""
    def __init__(self, output_dim=256, descriptor_dim=15):
        super().__init__()
        self.crop_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.GroupNorm(4, 32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GroupNorm(8, 128), nn.GELU(),
            nn.Conv2d(128, 192, 3, stride=2, padding=1), nn.GroupNorm(8, 192), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.descriptor_encoder = nn.Sequential(
            nn.LayerNorm(descriptor_dim),
            nn.Linear(descriptor_dim, 64),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(192 + 64, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )

    def forward(self, crops, descriptors):
        crop_features = self.crop_encoder(crops).flatten(1)
        descriptor_features = self.descriptor_encoder(descriptors)
        return self.fusion(torch.cat((crop_features, descriptor_features), dim=-1))


class LightTextLinking(nn.Module):
    def __init__(self, args):
        super(LightTextLinking, self).__init__()
        self.emb_dim = 768
        self.aux_losses = args.aux_losses
        self.embedding_components = args.embedding_components
        self.max_position_embeddings = getattr(args, "max_position_embeddings", 1280)
        self.token_padding_max_length = getattr(args, "token_padding_max_length", 1000)
        self.use_word_style = getattr(args, "use_word_style", False)
        self.use_pairwise_relations = getattr(args, "use_pairwise_relations", False)
        self.style_contrastive_weight = getattr(args, "style_contrastive_weight", 0.1)
        self.style_temperature = getattr(args, "style_temperature", 0.1)
        self.style_dim = getattr(args, "style_embedding_dim", 256)

        self.config = LayoutLMv3Config(max_position_embeddings=self.max_position_embeddings)
        self.light = LightModel(self.config)
        self.token_encoder = TokenEncoder(self.emb_dim)

        if self.use_word_style:
            self.word_style_encoder = WordStyleEncoder(output_dim=self.style_dim)
            self.style_projection = nn.Linear(self.style_dim, self.emb_dim)
            # Preserve pretrained LIGHT behavior at initialization.
            nn.init.zeros_(self.style_projection.weight)
            nn.init.zeros_(self.style_projection.bias)

        if self.use_pairwise_relations:
            self.relation_mlp = nn.Sequential(
                nn.LayerNorm(13),
                nn.Linear(13, 128), nn.GELU(),
                nn.Linear(128, 64), nn.GELU(),
                nn.Linear(64, 1),
            )
            nn.init.zeros_(self.relation_mlp[-1].weight)
            nn.init.zeros_(self.relation_mlp[-1].bias)
        
        self.predecessor_mlp = MLP(self.emb_dim, self.emb_dim, 
                                   [self.emb_dim, self.emb_dim], 
                                   nonlinearity='relu')
        self.successor_mlp = MLP(self.emb_dim, self.emb_dim, 
                                 [self.emb_dim, self.emb_dim], 
                                 nonlinearity='relu')
            
        self.loss_fn = nn.CrossEntropyLoss() 
        self.focal_loss_fn = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)

    def forward(self, input_data, return_loss=True):
        B = input_data['labels'].shape[0]
        attn_mask = input_data['attention_mask']

        outputs, poly_embedding = self.light(
            input_ids=input_data['input_ids'],
            bbox=input_data['bbox'],
            attention_mask=attn_mask,
            pixel_values=input_data['pixel_values'],
            polygons=input_data['polygons']
        )

        token_embeddings = outputs.last_hidden_state[:, :self.token_padding_max_length]
            
        style_embeddings = self.encode_styles(input_data) if self.use_word_style else None
        all_losses = defaultdict(int)
        all_logits = {'logits': [], 'bi_logits': []}
        for batch_idx in range(B):
            first_token_indices = input_data['first_token_indices'][batch_idx]
            first_token_indices = first_token_indices[first_token_indices != -999]

            embeddings = token_embeddings[batch_idx][first_token_indices]
            embeddings = self.token_encoder(embeddings)

            num_words = embeddings.shape[0]
            sample_styles = None
            if style_embeddings is not None:
                available_words = min(num_words, style_embeddings.shape[1])
                sample_styles = embeddings.new_zeros((num_words, self.style_dim))
                sample_styles[:available_words] = style_embeddings[batch_idx, :available_words]
                visual_mask = torch.zeros(num_words, dtype=torch.bool, device=embeddings.device)
                visual_mask[:available_words] = input_data['word_visual_mask'][batch_idx, :available_words]
                style_delta = self.style_projection(sample_styles)
                style_delta = style_delta * visual_mask.unsqueeze(-1).to(style_delta.dtype)
                embeddings = embeddings + style_delta
            
            pred_embeddings = self.predecessor_mlp(embeddings)
            succ_embeddings = self.successor_mlp(embeddings)
        
            dot_products = torch.matmul(pred_embeddings, succ_embeddings.T)
            bi_dot_products = torch.matmul(succ_embeddings, pred_embeddings.T)
            if self.use_pairwise_relations:
                available_words = min(num_words, input_data['word_geometries'].shape[1])
                geometry = embeddings.new_zeros((num_words, 7))
                geometry[:available_words] = input_data['word_geometries'][batch_idx, :available_words]
                relation_bias = self.compute_relation_bias(geometry, sample_styles)
                dot_products = dot_products + relation_bias
                bi_dot_products = bi_dot_products + relation_bias.T
            all_logits['logits'].append(dot_products)
            all_logits['bi_logits'].append(bi_dot_products)

            if return_loss:
                lables_B = [input_data['labels'][batch_idx][i] for i in first_token_indices]
                losses = self.compute_sample_losses(lables_B, dot_products, bi_dot_products)
                all_losses['base_loss'] += losses['base_loss']
                if 'bidirection' in self.aux_losses:
                    all_losses['bidirection'] += losses['bidirection']
                if 'focal' in self.aux_losses:
                    all_losses['focal_loss'] += losses['focal_loss']
                if (self.use_word_style and self.style_contrastive_weight > 0
                        and ('style_contrastive' in self.aux_losses or 'style' in self.aux_losses)):
                    valid_styles = sample_styles[visual_mask]
                    valid_labels = torch.stack(lables_B).to(visual_mask.device)[visual_mask]
                    style_loss = self.compute_style_contrastive_loss(valid_styles, valid_labels)
                    all_losses['style_contrastive'] += self.style_contrastive_weight * style_loss
            
        return all_logits, all_losses

    def encode_styles(self, input_data):
        """Encode only real crops, avoiding CNN work on padded word slots."""
        crops = input_data['word_crops']
        descriptors = input_data['word_styles']
        mask = input_data['word_visual_mask'].bool()
        batch_size, max_words = mask.shape
        output = crops.new_zeros((batch_size, max_words, self.style_dim))
        if mask.any():
            output[mask] = self.word_style_encoder(crops[mask], descriptors[mask])
        return output

    def compute_relation_bias(self, geometry, style_embeddings=None):
        """Build directional geometry/style features for every candidate edge."""
        eps = 1e-6
        source = geometry[:, None, :]
        target = geometry[None, :, :]
        dx = target[..., 0] - source[..., 0]
        dy = target[..., 1] - source[..., 1]
        mean_height = 0.5 * (source[..., 3] + target[..., 3]).clamp_min(eps)
        ndx, ndy = dx / mean_height, dy / mean_height
        distance = torch.sqrt(ndx.square() + ndy.square() + eps)
        direction_x, direction_y = ndx / distance, ndy / distance
        log_width_ratio = torch.log((target[..., 2] + eps) / (source[..., 2] + eps))
        log_height_ratio = torch.log((target[..., 3] + eps) / (source[..., 3] + eps))
        log_area_ratio = torch.log((target[..., 4] + eps) / (source[..., 4] + eps))
        orientation_similarity = source[..., 5] * target[..., 5] + source[..., 6] * target[..., 6]
        source_alignment = direction_x * source[..., 6] + direction_y * source[..., 5]
        target_alignment = direction_x * target[..., 6] + direction_y * target[..., 5]

        if style_embeddings is None:
            style_cosine = torch.zeros_like(dx)
            style_difference = torch.zeros_like(dx)
        else:
            normalized_style = F.normalize(style_embeddings, dim=-1)
            style_cosine = normalized_style @ normalized_style.T
            style_difference = (style_embeddings[:, None] - style_embeddings[None, :]).abs().mean(-1)

        features = torch.stack((
            ndx, ndy, distance, direction_x, direction_y,
            log_width_ratio, log_height_ratio, log_area_ratio,
            orientation_similarity, source_alignment, target_alignment,
            style_cosine, style_difference,
        ), dim=-1)
        return self.relation_mlp(features).squeeze(-1)

    def compute_style_contrastive_loss(self, embeddings, labels):
        """Supervised contrastive loss: same text group is positive."""
        if embeddings is None or embeddings.shape[0] < 2:
            return embeddings.new_zeros(()) if embeddings is not None else torch.tensor(0.0)
        if not isinstance(labels, torch.Tensor):
            labels = torch.stack(labels)
        labels = labels.to(embeddings.device)
        group_ids = torch.div(labels, 1000, rounding_mode='floor')
        same_group = group_ids[:, None].eq(group_ids[None, :])
        eye = torch.eye(len(labels), dtype=torch.bool, device=embeddings.device)
        positives = same_group & ~eye
        valid_anchors = positives.any(dim=1)
        if not valid_anchors.any():
            return embeddings.new_zeros(())
        logits = F.normalize(embeddings, dim=-1) @ F.normalize(embeddings, dim=-1).T
        logits = logits / self.style_temperature
        logits = logits.masked_fill(eye, float('-inf'))
        log_probability = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        positive_log_probability = log_probability.masked_fill(~positives, 0.0)
        mean_positive_log_probability = positive_log_probability.sum(dim=1) / (
            positives.sum(dim=1).clamp_min(1).to(logits.dtype)
        )
        return -mean_positive_log_probability[valid_anchors].mean()


    def compute_sample_losses(self, labels, dot_products, bi_dot_products):
        losses = {}
        label_to_index = {label.item(): idx for idx, label in enumerate(labels)}
        target_next_indices, targer_prev_indices = [], []

        target_matrix_succ = torch.zeros_like(dot_products, dtype=torch.float32)
        target_matrix_prev = torch.zeros_like(dot_products, dtype=torch.float32)

        for i, label in enumerate(labels):
            label = label.item()
            a, b = str(label).split('000', 1)
            a, b = int(a), int(b)
            next_label = int(f"{a}000{b + 1}")
            prev_label = int(f"{a}000{b - 1}")
    
            if label_to_index.get(next_label) is not None:
                target_next_indices.append(label_to_index[next_label])
                target_matrix_succ[i, label_to_index[next_label]] = 1.0 
            else:
                target_next_indices.append(label_to_index[label])

            if label_to_index.get(prev_label) is not None:
                targer_prev_indices.append(label_to_index[prev_label])
                target_matrix_prev[i, label_to_index[prev_label]] = 1.0 
            else:
                targer_prev_indices.append(label_to_index[label])
            
        target_next_indices = torch.tensor(target_next_indices, dtype=torch.long).to(dot_products.device)
        loss = self.loss_fn(dot_products, target_next_indices)        
        losses['base_loss'] = loss

        if 'bidirection' in self.aux_losses:
            targer_prev_indices = torch.tensor(targer_prev_indices, dtype=torch.long).to(dot_products.device)
            loss = self.loss_fn(bi_dot_products, targer_prev_indices)
            losses['bidirection'] = loss

        if 'focal' in self.aux_losses:
            losses['focal_loss'] = 0
            focal_loss = self.focal_loss_fn(dot_products, target_matrix_succ)
            losses['focal_loss'] += focal_loss.mean() * 100

            if 'bidirection' in self.aux_losses:
                focal_loss = self.focal_loss_fn(bi_dot_products, target_matrix_prev)
                losses['focal_loss'] += focal_loss.mean() * 100

        return losses
