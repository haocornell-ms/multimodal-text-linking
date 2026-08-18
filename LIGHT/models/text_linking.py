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


class GatedStyleFusion(nn.Module):
    """Fuse word-level visual style into text with a gated residual."""
    def __init__(self, text_dim=768, style_dim=256, hidden_dim=768):
        super().__init__()
        self.text_norm = nn.LayerNorm(text_dim)
        self.style_norm = nn.LayerNorm(style_dim)
        input_dim = text_dim + style_dim
        self.delta_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, text_dim),
        )
        self.gate = nn.Linear(input_dim, text_dim)

        # Start close to the text-only model while retaining gradients through
        # both the visual branch and the gate from the first update.
        nn.init.normal_(self.delta_mlp[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.delta_mlp[-1].bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, text_embeddings, style_embeddings):
        fused_input = torch.cat((
            self.text_norm(text_embeddings),
            self.style_norm(style_embeddings),
        ), dim=-1)
        style_delta = self.delta_mlp(fused_input)
        return text_embeddings + torch.sigmoid(self.gate(fused_input)) * style_delta


class LateVisualEdgeScorer(nn.Module):
    """Score visual compatibility without modifying text token embeddings."""
    def __init__(self, style_dim=256, pair_dim=32, hidden_dim=128):
        super().__init__()
        self.style_projection = nn.Sequential(
            nn.LayerNorm(style_dim), nn.Linear(style_dim, pair_dim), nn.GELU()
        )
        self.edge_mlp = nn.Sequential(
            nn.LayerNorm(pair_dim * 4 + 11),
            nn.Linear(pair_dim * 4 + 11, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.normal_(self.edge_mlp[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.edge_mlp[-1].bias)

    def forward(self, styles, geometry, visual_mask):
        num_words = len(styles)
        scores = styles.new_zeros((num_words, num_words))
        valid_indices = visual_mask.nonzero(as_tuple=False).flatten()
        if len(valid_indices) < 2:
            return scores
        style = self.style_projection(styles[valid_indices])
        valid_geometry = geometry[valid_indices]
        num_visual_words = len(style)
        source_style = style[:, None, :].expand(-1, num_visual_words, -1)
        target_style = style[None, :, :].expand(num_visual_words, -1, -1)
        geometry_features = LightTextLinking.geometry_pair_features(valid_geometry)
        pair_features = torch.cat((
            source_style,
            target_style,
            (source_style - target_style).abs(),
            source_style * target_style,
            geometry_features,
        ), dim=-1)
        valid_scores = self.edge_mlp(pair_features).squeeze(-1)
        off_diagonal = ~torch.eye(
            num_visual_words, dtype=torch.bool, device=styles.device
        )
        valid_scores = valid_scores * off_diagonal.to(valid_scores.dtype)
        scores[valid_indices[:, None], valid_indices[None, :]] = valid_scores
        return scores


class VisualEntityAffinityScorer(nn.Module):
    """Predict same-entity probability and an affinity-aware link residual."""
    def __init__(self, style_dim=256, pair_dim=32, hidden_dim=128):
        super().__init__()
        self.style_projection = nn.Sequential(
            nn.LayerNorm(style_dim), nn.Linear(style_dim, pair_dim), nn.GELU()
        )
        # Symmetric inputs make P(i,j) == P(j,i), as required for entity identity.
        affinity_dim = pair_dim * 3 + 9
        self.affinity_mlp = nn.Sequential(
            nn.LayerNorm(affinity_dim),
            nn.Linear(affinity_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        # Linking remains directional and can combine affinity with map geometry.
        link_dim = pair_dim * 4 + 11 + 1
        self.link_mlp = nn.Sequential(
            nn.LayerNorm(link_dim),
            nn.Linear(link_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.normal_(self.affinity_mlp[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.affinity_mlp[-1].bias)
        nn.init.normal_(self.link_mlp[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.link_mlp[-1].bias)

    @staticmethod
    def symmetric_geometry_features(geometry):
        eps = 1e-6
        source, target = geometry[:, None, :], geometry[None, :, :]
        mean_height = 0.5 * (source[..., 3] + target[..., 3]).clamp_min(eps)
        ndx = (target[..., 0] - source[..., 0]) / mean_height
        ndy = (target[..., 1] - source[..., 1]) / mean_height
        distance = torch.sqrt(ndx.square() + ndy.square() + eps)
        direction_x, direction_y = ndx / distance, ndy / distance
        width_ratio = torch.log(
            (target[..., 2] + eps) / (source[..., 2] + eps)
        ).abs()
        height_ratio = torch.log(
            (target[..., 3] + eps) / (source[..., 3] + eps)
        ).abs()
        area_ratio = torch.log(
            (target[..., 4] + eps) / (source[..., 4] + eps)
        ).abs()
        orientation_similarity = (
            source[..., 5] * target[..., 5]
            + source[..., 6] * target[..., 6]
        ).abs()
        return torch.stack((
            ndx.abs(), ndy.abs(), distance,
            direction_x.abs(), direction_y.abs(),
            width_ratio, height_ratio, area_ratio, orientation_similarity,
        ), dim=-1)

    def forward(self, styles, geometry, visual_mask):
        num_words = len(styles)
        affinity_logits = styles.new_zeros((num_words, num_words))
        link_scores = styles.new_zeros((num_words, num_words))
        valid_pair_mask = torch.zeros(
            (num_words, num_words), dtype=torch.bool, device=styles.device
        )
        valid_indices = visual_mask.nonzero(as_tuple=False).flatten()
        if len(valid_indices) < 2:
            return affinity_logits, link_scores, valid_pair_mask

        style = self.style_projection(styles[valid_indices])
        valid_geometry = geometry[valid_indices]
        num_visual_words = len(style)
        source_style = style[:, None, :].expand(-1, num_visual_words, -1)
        target_style = style[None, :, :].expand(num_visual_words, -1, -1)
        symmetric_features = torch.cat((
            source_style + target_style,
            (source_style - target_style).abs(),
            source_style * target_style,
            self.symmetric_geometry_features(valid_geometry),
        ), dim=-1)
        valid_affinity_logits = self.affinity_mlp(
            symmetric_features
        ).squeeze(-1)
        # Numerical averaging guarantees exact symmetry despite finite precision.
        valid_affinity_logits = 0.5 * (
            valid_affinity_logits + valid_affinity_logits.T
        )
        affinity_probability = torch.sigmoid(valid_affinity_logits)

        directional_geometry = LightTextLinking.geometry_pair_features(
            valid_geometry
        )
        directional_features = torch.cat((
            source_style,
            target_style,
            (source_style - target_style).abs(),
            source_style * target_style,
            directional_geometry,
            affinity_probability.unsqueeze(-1),
        ), dim=-1)
        valid_link_scores = self.link_mlp(directional_features).squeeze(-1)
        off_diagonal = ~torch.eye(
            num_visual_words, dtype=torch.bool, device=styles.device
        )
        valid_link_scores = valid_link_scores * off_diagonal.to(
            valid_link_scores.dtype
        )
        affinity_logits[valid_indices[:, None], valid_indices[None, :]] = (
            valid_affinity_logits
        )
        link_scores[valid_indices[:, None], valid_indices[None, :]] = (
            valid_link_scores
        )
        valid_pair_mask[valid_indices[:, None], valid_indices[None, :]] = (
            off_diagonal
        )
        return affinity_logits, link_scores, valid_pair_mask


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
        self.preserve_text_stop_scores = getattr(args, "preserve_text_stop_scores", False)
        self.style_contrastive_weight = getattr(args, "style_contrastive_weight", 0.1)
        self.edge_soft_f1_weight = getattr(args, "edge_soft_f1_weight", 0.0)
        self.edge_soft_f1_epsilon = getattr(args, "edge_soft_f1_epsilon", 1e-6)
        self.edge_ranking_weight = getattr(args, "edge_ranking_weight", 0.0)
        self.edge_ranking_margin = getattr(args, "edge_ranking_margin", 0.2)
        self.edge_ranking_top_k = getattr(args, "edge_ranking_top_k", 3)
        self.edge_collision_weight = getattr(args, "edge_collision_weight", 0.0)
        self.edge_consistency_weight = getattr(args, "edge_consistency_weight", 0.0)
        self.style_temperature = getattr(args, "style_temperature", 0.1)
        self.style_dim = getattr(args, "style_embedding_dim", 256)
        self.style_fusion_hidden_dim = getattr(args, "style_fusion_hidden_dim", self.emb_dim)
        self.use_factorized_linking = getattr(args, "use_factorized_linking", False)
        self.use_visual_edge_residual = getattr(args, "use_visual_edge_residual", False)
        self.use_visual_entity_affinity = getattr(
            args, "use_visual_entity_affinity", False
        )
        self.use_token_style_fusion = getattr(args, "use_token_style_fusion", True)
        self.stop_loss_weight = getattr(args, "stop_loss_weight", 1.0)
        self.visual_pair_dim = getattr(args, "visual_pair_dim", 32)
        self.visual_edge_max_scale = getattr(args, "visual_edge_max_scale", 1.0)
        self.visual_edge_initial_scale = getattr(args, "visual_edge_initial_scale", 0.05)
        self.entity_affinity_loss_weight = getattr(
            args, "entity_affinity_loss_weight", 0.5
        )
        self.entity_affinity_max_scale = getattr(
            args, "entity_affinity_max_scale", 0.1
        )
        self.entity_affinity_initial_scale = getattr(
            args, "entity_affinity_initial_scale", 0.025
        )
        if (self.use_factorized_linking or self.use_visual_edge_residual
                or self.use_visual_entity_affinity) and not self.use_word_style:
            raise ValueError("Visual edge scoring requires use_word_style=True")
        if sum((
                self.use_factorized_linking,
                self.use_visual_edge_residual,
                self.use_visual_entity_affinity,
        )) > 1:
            raise ValueError(
                "factorized, residual, and entity-affinity linking are mutually exclusive"
            )

        self.config = LayoutLMv3Config(max_position_embeddings=self.max_position_embeddings)
        self.light = LightModel(self.config)
        self.token_encoder = TokenEncoder(self.emb_dim)

        if self.use_word_style:
            self.word_style_encoder = WordStyleEncoder(output_dim=self.style_dim)
            if self.use_factorized_linking or self.use_visual_edge_residual:
                self.visual_edge_scorer = LateVisualEdgeScorer(
                    style_dim=self.style_dim, pair_dim=self.visual_pair_dim
                )
            if self.use_visual_entity_affinity:
                self.visual_entity_affinity_scorer = VisualEntityAffinityScorer(
                    style_dim=self.style_dim, pair_dim=self.visual_pair_dim
                )
                initial_fraction = min(max(
                    self.entity_affinity_initial_scale
                    / self.entity_affinity_max_scale,
                    1e-4,
                ), 1.0 - 1e-4)
                self.entity_affinity_scale_logit = nn.Parameter(torch.tensor(
                    torch.logit(torch.tensor(initial_fraction)).item()
                ))
            if self.use_visual_edge_residual:
                initial_fraction = min(max(
                    self.visual_edge_initial_scale / self.visual_edge_max_scale,
                    1e-4,
                ), 1.0 - 1e-4)
                self.visual_edge_scale_logit = nn.Parameter(torch.tensor(
                    torch.logit(torch.tensor(initial_fraction)).item()
                ))
            if self.use_factorized_linking:
                stop_input_dim = self.emb_dim + self.visual_pair_dim + 4
                self.stop_style_projection = nn.Sequential(
                    nn.LayerNorm(self.style_dim),
                    nn.Linear(self.style_dim, self.visual_pair_dim), nn.GELU(),
                )
                self.successor_stop_head = self._make_stop_head(stop_input_dim)
                self.predecessor_stop_head = self._make_stop_head(stop_input_dim)
            elif self.use_token_style_fusion:
                self.style_fusion = GatedStyleFusion(
                    text_dim=self.emb_dim,
                    style_dim=self.style_dim,
                    hidden_dim=self.style_fusion_hidden_dim,
                )

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

    @property
    def visual_edge_scale(self):
        if not self.use_visual_edge_residual:
            return None
        return self.visual_edge_max_scale * torch.sigmoid(
            self.visual_edge_scale_logit
        )

    @property
    def entity_affinity_scale(self):
        if not self.use_visual_entity_affinity:
            return None
        return self.entity_affinity_max_scale * torch.sigmoid(
            self.entity_affinity_scale_logit
        )

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
        all_logits = {'logits': [], 'bi_logits': [], 'entity_affinity': []}
        for batch_idx in range(B):
            first_token_indices = input_data['first_token_indices'][batch_idx]
            first_token_indices = first_token_indices[first_token_indices != -999]

            embeddings = token_embeddings[batch_idx][first_token_indices]
            embeddings = self.token_encoder(embeddings)
            text_embeddings = embeddings

            num_words = embeddings.shape[0]
            sample_styles = None
            visual_mask = torch.zeros(num_words, dtype=torch.bool, device=embeddings.device)
            if style_embeddings is not None:
                available_words = min(num_words, style_embeddings.shape[1])
                sample_styles = embeddings.new_zeros((num_words, self.style_dim))
                sample_styles[:available_words] = style_embeddings[batch_idx, :available_words]
                visual_mask[:available_words] = input_data['word_visual_mask'][batch_idx, :available_words]
                if not self.use_factorized_linking and self.use_token_style_fusion:
                    fused_embeddings = self.style_fusion(embeddings, sample_styles)
                    embeddings = torch.where(
                        visual_mask.unsqueeze(-1), fused_embeddings, embeddings
                    )

            text_stop_logits = None
            if self.use_word_style and self.preserve_text_stop_scores:
                text_pred_embeddings = self.predecessor_mlp(text_embeddings)
                text_succ_embeddings = self.successor_mlp(text_embeddings)
                text_stop_logits = (
                    text_pred_embeddings * text_succ_embeddings
                ).sum(dim=-1)
            
            pred_embeddings = self.predecessor_mlp(embeddings)
            succ_embeddings = self.successor_mlp(embeddings)
        
            dot_products = torch.matmul(pred_embeddings, succ_embeddings.T)
            bi_dot_products = torch.matmul(succ_embeddings, pred_embeddings.T)
            geometry = None
            if (self.use_factorized_linking or self.use_visual_edge_residual
                    or self.use_visual_entity_affinity
                    or self.use_pairwise_relations):
                available_words = min(num_words, input_data['word_geometries'].shape[1])
                geometry = embeddings.new_zeros((num_words, 7))
                geometry[:available_words] = input_data['word_geometries'][batch_idx, :available_words]
            if self.use_visual_edge_residual:
                edge_delta = self.visual_edge_scorer(sample_styles, geometry, visual_mask)
                edge_scale = self.visual_edge_scale
                dot_products = dot_products + edge_scale * edge_delta
                bi_dot_products = bi_dot_products + edge_scale * edge_delta.T
            entity_affinity_logits = entity_affinity_pair_mask = None
            if self.use_visual_entity_affinity:
                (entity_affinity_logits, entity_link_scores,
                 entity_affinity_pair_mask) = self.visual_entity_affinity_scorer(
                    sample_styles, geometry, visual_mask
                )
                entity_scale = self.entity_affinity_scale
                dot_products = dot_products + entity_scale * entity_link_scores
                bi_dot_products = (
                    bi_dot_products + entity_scale * entity_link_scores.T
                )
                all_logits['entity_affinity'].append(
                    torch.sigmoid(entity_affinity_logits)
                    * entity_affinity_pair_mask.to(entity_affinity_logits.dtype)
                )
            if self.use_factorized_linking:
                edge_delta = self.visual_edge_scorer(sample_styles, geometry, visual_mask)
                dot_products = dot_products + edge_delta
                bi_dot_products = bi_dot_products + edge_delta.T
                successor_stop_logits = self.compute_stop_logits(
                    self.successor_stop_head, text_embeddings, sample_styles,
                    visual_mask, dot_products
                )
                predecessor_stop_logits = self.compute_stop_logits(
                    self.predecessor_stop_head, text_embeddings, sample_styles,
                    visual_mask, bi_dot_products
                )
                dot_products = self.combine_factorized_logits(
                    successor_stop_logits, dot_products
                )
                bi_dot_products = self.combine_factorized_logits(
                    predecessor_stop_logits, bi_dot_products
                )
            if self.use_pairwise_relations and not self.use_factorized_linking:
                relation_bias = self.compute_relation_bias(geometry, sample_styles)
                dot_products = dot_products + relation_bias
                bi_dot_products = bi_dot_products + relation_bias.T
            if text_stop_logits is not None and not self.use_factorized_linking:
                # A diagonal entry means STOP. Keep that decision grounded in
                # the text/layout representation; visual features may only
                # change scores between distinct words.
                diagonal_mask = torch.eye(
                    num_words, dtype=torch.bool, device=embeddings.device
                )
                text_stop_matrix = torch.diag_embed(text_stop_logits)
                dot_products = torch.where(
                    diagonal_mask, text_stop_matrix, dot_products
                )
                bi_dot_products = torch.where(
                    diagonal_mask, text_stop_matrix, bi_dot_products
                )
            all_logits['logits'].append(dot_products)
            all_logits['bi_logits'].append(bi_dot_products)

            if return_loss:
                lables_B = [input_data['labels'][batch_idx][i] for i in first_token_indices]
                if self.use_factorized_linking:
                    losses = self.compute_factorized_losses(
                        lables_B, dot_products, bi_dot_products,
                        successor_stop_logits, predecessor_stop_logits,
                    )
                else:
                    losses = self.compute_sample_losses(
                        lables_B, dot_products, bi_dot_products,
                        entity_affinity_logits, entity_affinity_pair_mask,
                    )
                for loss_name, loss_value in losses.items():
                    all_losses[loss_name] += loss_value
                if (self.use_word_style and self.style_contrastive_weight > 0
                        and ('style_contrastive' in self.aux_losses or 'style' in self.aux_losses)):
                    valid_styles = sample_styles[visual_mask]
                    valid_labels = torch.stack(lables_B).to(visual_mask.device)[visual_mask]
                    style_loss = self.compute_style_contrastive_loss(valid_styles, valid_labels)
                    all_losses['style_contrastive'] += self.style_contrastive_weight * style_loss
            
        return all_logits, all_losses

    @staticmethod
    def _make_stop_head(input_dim):
        return nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, 256), nn.GELU(),
            nn.Linear(256, 1),
        )

    @staticmethod
    def geometry_pair_features(geometry):
        eps = 1e-6
        source, target = geometry[:, None, :], geometry[None, :, :]
        dx = target[..., 0] - source[..., 0]
        dy = target[..., 1] - source[..., 1]
        mean_height = 0.5 * (source[..., 3] + target[..., 3]).clamp_min(eps)
        ndx, ndy = dx / mean_height, dy / mean_height
        distance = torch.sqrt(ndx.square() + ndy.square() + eps)
        direction_x, direction_y = ndx / distance, ndy / distance
        return torch.stack((
            ndx, ndy, distance, direction_x, direction_y,
            torch.log((target[..., 2] + eps) / (source[..., 2] + eps)),
            torch.log((target[..., 3] + eps) / (source[..., 3] + eps)),
            torch.log((target[..., 4] + eps) / (source[..., 4] + eps)),
            source[..., 5] * target[..., 5] + source[..., 6] * target[..., 6],
            direction_x * source[..., 6] + direction_y * source[..., 5],
            direction_x * target[..., 6] + direction_y * target[..., 5],
        ), dim=-1)

    def compute_stop_logits(self, head, text, styles, visual_mask, edge_scores):
        style = self.stop_style_projection(styles)
        style = style * visual_mask.unsqueeze(-1).to(style.dtype)
        num_words = edge_scores.shape[0]
        if num_words > 1:
            non_self = edge_scores.masked_fill(
                torch.eye(num_words, dtype=torch.bool, device=edge_scores.device),
                float('-inf'),
            )
            top_k = min(3, num_words - 1)
            top_values = non_self.topk(top_k, dim=-1).values
            evidence = torch.stack((
                top_values[:, 0], top_values.mean(-1),
                edge_scores.diagonal(),
                top_values[:, 0] - edge_scores.diagonal(),
            ), dim=-1)
        else:
            evidence = edge_scores.new_zeros((1, 4))
        features = torch.cat((F.layer_norm(text, (text.shape[-1],)), style, evidence), dim=-1)
        return head(features).squeeze(-1)

    @staticmethod
    def combine_factorized_logits(stop_logits, edge_scores):
        """Return log P(STOP) on diagonal and log P(CONTINUE,target) elsewhere."""
        num_words = edge_scores.shape[0]
        if num_words == 1:
            return F.logsigmoid(stop_logits).reshape(1, 1)
        diagonal = torch.eye(num_words, dtype=torch.bool, device=edge_scores.device)
        non_stop_scores = edge_scores.masked_fill(diagonal, float('-inf'))
        conditional_log_probs = F.log_softmax(non_stop_scores, dim=-1)
        combined = F.logsigmoid(-stop_logits)[:, None] + conditional_log_probs
        return torch.where(diagonal, F.logsigmoid(stop_logits)[:, None], combined)

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

    @staticmethod
    def build_link_targets(labels):
        label_to_index = {label.item(): index for index, label in enumerate(labels)}
        next_targets, previous_targets = [], []
        for index, label_tensor in enumerate(labels):
            label = label_tensor.item()
            group_id, sequence_id = str(label).split('000', 1)
            next_label = int(f"{int(group_id)}000{int(sequence_id) + 1}")
            previous_label = int(f"{int(group_id)}000{int(sequence_id) - 1}")
            next_targets.append(label_to_index.get(next_label, index))
            previous_targets.append(label_to_index.get(previous_label, index))
        device = labels[0].device
        return (
            torch.tensor(next_targets, dtype=torch.long, device=device),
            torch.tensor(previous_targets, dtype=torch.long, device=device),
        )

    def compute_factorized_direction_loss(self, combined_logits, stop_logits, targets):
        source_indices = torch.arange(len(targets), device=targets.device)
        stop_targets = targets.eq(source_indices)
        stop_loss = F.binary_cross_entropy_with_logits(
            stop_logits, stop_targets.to(stop_logits.dtype)
        )
        continue_mask = ~stop_targets
        if continue_mask.any():
            successor_loss = F.nll_loss(
                combined_logits[continue_mask], targets[continue_mask]
            )
            # Remove the CONTINUE term already present in the combined log
            # probability, leaving only P(target | CONTINUE).
            successor_loss = successor_loss + F.logsigmoid(
                -stop_logits[continue_mask]
            ).mean()
        else:
            successor_loss = stop_logits.new_zeros(())
        return successor_loss, stop_loss

    def compute_factorized_losses(self, labels, logits, bi_logits,
                                  stop_logits, predecessor_stop_logits):
        if not isinstance(labels, torch.Tensor):
            labels = torch.stack(labels)
        next_targets, previous_targets = self.build_link_targets(labels)
        successor_loss, stop_loss = self.compute_factorized_direction_loss(
            logits, stop_logits, next_targets
        )
        losses = {
            'base_loss': successor_loss,
            'stop_loss': self.stop_loss_weight * stop_loss,
        }
        if 'bidirection' in self.aux_losses:
            predecessor_loss, predecessor_stop_loss = self.compute_factorized_direction_loss(
                bi_logits, predecessor_stop_logits, previous_targets
            )
            losses['bidirection'] = predecessor_loss
            losses['predecessor_stop_loss'] = (
                self.stop_loss_weight * predecessor_stop_loss
            )
        return losses


    def compute_sample_losses(self, labels, dot_products, bi_dot_products,
                              entity_affinity_logits=None,
                              entity_affinity_pair_mask=None):
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
        targer_prev_indices = torch.tensor(
            targer_prev_indices, dtype=torch.long, device=dot_products.device
        )
        loss = self.loss_fn(dot_products, target_next_indices)        
        losses['base_loss'] = loss

        if 'bidirection' in self.aux_losses:
            loss = self.loss_fn(bi_dot_products, targer_prev_indices)
            losses['bidirection'] = loss

        if self.edge_soft_f1_weight > 0:
            losses['edge_soft_f1_loss'] = self.edge_soft_f1_weight * (
                self.compute_soft_edge_f1_loss(dot_products, target_matrix_succ)
                + self.compute_soft_edge_f1_loss(
                    bi_dot_products, target_matrix_prev
                )
            )

        if self.edge_ranking_weight > 0:
            losses['edge_ranking_loss'] = self.edge_ranking_weight * (
                self.compute_hard_negative_ranking_loss(
                    dot_products, target_next_indices
                )
                + self.compute_hard_negative_ranking_loss(
                    bi_dot_products, targer_prev_indices
                )
            )

        if self.edge_collision_weight > 0:
            losses['edge_collision_loss'] = self.edge_collision_weight * (
                self.compute_successor_collision_loss(dot_products)
                + self.compute_successor_collision_loss(bi_dot_products)
            )

        if self.edge_consistency_weight > 0:
            losses['edge_consistency_loss'] = self.edge_consistency_weight * (
                self.compute_direction_consistency_loss(
                    dot_products, bi_dot_products
                )
            )

        if (entity_affinity_logits is not None
                and entity_affinity_pair_mask is not None
                and entity_affinity_pair_mask.any()):
            label_tensor = torch.stack(labels).to(dot_products.device)
            group_ids = torch.div(label_tensor, 1000, rounding_mode='floor')
            same_entity_targets = group_ids[:, None].eq(group_ids[None, :])
            valid_logits = entity_affinity_logits[entity_affinity_pair_mask]
            valid_targets = same_entity_targets[entity_affinity_pair_mask]
            positive_logits = valid_logits[valid_targets]
            negative_logits = valid_logits[~valid_targets]
            positive_loss = (
                F.binary_cross_entropy_with_logits(
                    positive_logits, torch.ones_like(positive_logits)
                ) if positive_logits.numel() else valid_logits.new_zeros(())
            )
            negative_loss = (
                F.binary_cross_entropy_with_logits(
                    negative_logits, torch.zeros_like(negative_logits)
                ) if negative_logits.numel() else valid_logits.new_zeros(())
            )
            losses['entity_affinity_loss'] = (
                self.entity_affinity_loss_weight
                * 0.5 * (positive_loss + negative_loss)
            )

        if 'focal' in self.aux_losses:
            losses['focal_loss'] = 0
            focal_loss = self.focal_loss_fn(dot_products, target_matrix_succ)
            losses['focal_loss'] += focal_loss.mean() * 100

            if 'bidirection' in self.aux_losses:
                focal_loss = self.focal_loss_fn(bi_dot_products, target_matrix_prev)
                losses['focal_loss'] += focal_loss.mean() * 100

        return losses

    def compute_soft_edge_f1_loss(self, logits, edge_targets):
        """Differentiable per-image F1 over non-STOP successor edges."""
        probabilities = torch.softmax(logits, dim=-1)
        off_diagonal = ~torch.eye(
            logits.shape[0], dtype=torch.bool, device=logits.device
        )
        edge_probabilities = probabilities[off_diagonal]
        edge_targets = edge_targets[off_diagonal]
        true_positives = (edge_probabilities * edge_targets).sum()
        predicted_positives = edge_probabilities.sum()
        actual_positives = edge_targets.sum()
        soft_f1 = (
            2.0 * true_positives + self.edge_soft_f1_epsilon
        ) / (
            predicted_positives
            + actual_positives
            + self.edge_soft_f1_epsilon
        )
        return 1.0 - soft_f1

    def compute_hard_negative_ranking_loss(self, logits, target_indices):
        """Require the correct successor to outrank the hardest alternatives."""
        num_candidates = logits.shape[1]
        if num_candidates <= 1:
            return logits.new_zeros(())
        target_scores = logits.gather(1, target_indices[:, None])
        negative_mask = torch.ones_like(logits, dtype=torch.bool)
        negative_mask.scatter_(1, target_indices[:, None], False)
        negative_scores = logits.masked_fill(~negative_mask, float('-inf'))
        top_k = min(self.edge_ranking_top_k, num_candidates - 1)
        hard_negatives = negative_scores.topk(top_k, dim=1).values
        return F.relu(
            self.edge_ranking_margin + hard_negatives - target_scores
        ).mean()

    @staticmethod
    def compute_successor_collision_loss(logits):
        """Penalize more than one source assigning mass to the same successor."""
        probabilities = torch.softmax(logits, dim=-1)
        off_diagonal = ~torch.eye(
            logits.shape[0], dtype=torch.bool, device=logits.device
        )
        incoming_edge_mass = (
            probabilities * off_diagonal.to(probabilities.dtype)
        ).sum(dim=0)
        return F.relu(incoming_edge_mass - 1.0).square().mean()

    @staticmethod
    def compute_direction_consistency_loss(logits, reverse_logits):
        """Make forward successor and reverse predecessor probabilities agree."""
        probabilities = torch.softmax(logits, dim=-1)
        reverse_probabilities = torch.softmax(reverse_logits, dim=-1).T
        off_diagonal = ~torch.eye(
            logits.shape[0], dtype=torch.bool, device=logits.device
        )
        return F.smooth_l1_loss(
            probabilities[off_diagonal], reverse_probabilities[off_diagonal]
        )
