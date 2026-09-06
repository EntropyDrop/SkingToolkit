"""Mutually exclusive head identities with paired-view semantic context.

An inner nose cannot simultaneously be a royal crown. A learned facial-hair
identity explicitly distinguishes flat and extruded beard geometry.
"""
import torch
from torch import nn
import torch.nn.functional as F
from SkingToolkit.dense_uv_parser.headwear import HeadwearHead
from SkingToolkit.dense_uv_parser.accessories import HeadAccessoryHead

CLASSES = ('abstain', 'inner_face', 'inner_hair', 'inner_beard', 'outer_hair',
           'outer_beard', 'eyewear', 'headphones', 'inner_hat_body',
           'inner_hat_band', 'outer_hat_body', 'outer_hat_band', 'brim', 'royal_crown')
PROJECTIONS = {
    'headwear': (tuple(range(8)), (8,), (9,), (10,), (11,), (12,), (13,)),
    'ownership': ((0,4,5,8,9,10,11,12,13), (1,3), (2,), (6,), (7,)),
}


def project_semantics(logits, task):
    """Marginalize ONE categorical distribution, preserving its probability."""
    return torch.stack([torch.logsumexp(logits[:, list(ids)].float(), 1)
                        for ids in PROJECTIONS[task]], 1)


class HeadSemanticsHead(HeadwearHead):
    def __init__(self, semantic_dim=768):
        super().__init__(semantic_dim, predict_presence=True)
        self.classifier = nn.Conv2d(24, len(CLASSES), 1)
        self.paired_context = True
        self.view_embedding = nn.Parameter(torch.zeros(2, 96))

    def forward(self, crop, semantic_features):
        logits = HeadAccessoryHead.forward(self, crop, semantic_features)
        headwear = project_semantics(logits, 'headwear').softmax(1)
        spatial = F.adaptive_avg_pool2d(headwear, (8,8)).flatten(1)
        semantic = semantic_features.float().mean((2,3))
        presence = self.presence(torch.cat([spatial, semantic], 1))
        return logits, presence


def apply_joint_head_routing(routing, outputs, foreground, renderer, views):
    """Consume the new model's inner-hair evidence across both visible top views.

    Face/beard evidence already reaches the established ownership router via
    categorical marginalization. This additional top veto requires confident
    inner hair in EVERY observed view, not a single-view colour/occupancy rule.
    Genuine outer hair and uncertain/occluded cells stay under existing routing.
    """
    if len(views) != 2:
        raise ValueError('Joint head routing requires front/back views')
    from SkingToolkit.dense_uv_parser.utils import build_static_surface_routing
    p = outputs['head_semantics_logits'].float().softmax(1)
    groups = foreground.shape[0] // len(views)
    negative = []
    for vi, view in enumerate(views):
        static = build_static_surface_routing(renderer, view, foreground.device)
        sl = slice(vi, foreground.shape[0], len(views))
        valid = foreground[sl] & static['masks'][1] & (static['part'][1]==0) & (static['face'][1]==4)
        index = static['flat_uv'][1].flatten()[None].expand(groups,-1)
        votes = p.new_zeros(groups,4096); counts = torch.zeros_like(votes)
        votes.scatter_add_(1,index,(p[sl,2]*valid).flatten(1))
        counts.scatter_add_(1,index,valid.float().flatten(1))
        negative.append((votes/counts.clamp_min(1) >= .95) & (counts>=8))
    cells = torch.stack(negative).all(0)
    accepted_all = torch.zeros_like(foreground)
    for vi, view in enumerate(views):
        static = build_static_surface_routing(renderer, view, foreground.device)
        sl = slice(vi, foreground.shape[0], len(views))
        accepted = cells.gather(1,static['flat_uv'][1].flatten()[None].expand(groups,-1)).reshape_as(foreground[sl])
        accepted &= foreground[sl] & static['masks'][0] & (static['part'][0]==0)
        accepted_all[sl] = accepted
        for name,value in [('layer',0),('foreground',True),('surface',0),('secondary',False),('secondary_routed',False),('semantic_fallback',False),('consensus_outer_gate_rejected',False)]:
            if name in routing:routing[name][sl]=torch.where(accepted,torch.full_like(routing[name][sl],value),routing[name][sl])
        for name in ('flat_uv','part','face','texel_center_score'):
            if name in routing:routing[name][sl]=torch.where(accepted,static[name][0],routing[name][sl])
        confidence=p[sl,2];margin=(2*confidence-1).clamp_min(0)
        for name,value in [('confidence',confidence),('confidence_margin',margin),('confidence_margin_ratio',margin/confidence.clamp_min(1e-6))]:
            if name in routing:routing[name][sl]=torch.where(accepted,value,routing[name][sl])
    routing['accessory_supported'] &= ~accepted_all
    routing['ownership_inner_supported'] |= accepted_all
    routing['joint_inner_hair_veto'] = accepted_all
    # Clear stale component attribution after a semantic layer correction.
    routing['headwear_supported'] &= ~accepted_all
    routing['headwear_layer'] = routing['headwear_layer'].masked_fill(accepted_all,-1)
    routing['headwear_family'] = routing['headwear_family'].masked_fill(accepted_all,0)
