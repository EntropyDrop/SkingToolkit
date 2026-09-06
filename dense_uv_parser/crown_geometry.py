"""Resolve crown top occupancy through the renderer's actual visible surfaces.

Pixel semantics alone cannot distinguish a primary top hit from a crown wall
behind it. Compare binary UV hypotheses through the same occlusion/fallback
renderer used for the final image. RGB and source filenames are not evidence.
"""
import copy

import torch

from SkingToolkit.dense_uv_parser.accessories import head_bounds
from SkingToolkit.dense_uv_parser.uv_topology import build_simple_uv_topology


def crop_renderer(renderer, views, height, width):
    """Share immutable buffers, slicing only image axes; never mutate renderer."""
    y0, y1, x0, x1 = head_bounds(height, width)
    cropped = copy.copy(renderer)
    cropped._buffers = renderer._buffers.copy()
    for name, value in cropped._buffers.items():
        if not any(name.startswith(view + '_') for view in views):
            continue
        for axis in range(value.ndim - 1):
            if value.shape[axis:axis + 2] == (height, width):
                slices = [slice(None)] * value.ndim
                slices[axis] = slice(y0, y1)
                slices[axis + 1] = slice(x0, x1)
                cropped._buffers[name] = value[tuple(slices)]
                break
    return cropped, (slice(y0, y1), slice(x0, x1))


@torch.no_grad()
def prune_crown_top(uv, target, foreground, renderer, views, crown_uv=None):
    """Delete a top cell only when its removal improves rendered semantics.

    Only existing outer head-top alpha can change. Hidden or ambiguous cells
    stay unchanged: no sparsity penalty, fixed open-top template, or minimum
    crown thickness. All other geometry and material stay frozen. The binary
    search re-evaluates visibility after each deletion, including secondary
    crown walls revealed behind the top plane. Each player is independent.
    """
    if target.shape[0] != uv.shape[0] * len(views) or foreground.shape != target.shape:
        raise ValueError('Crown geometry requires complete aligned view groups')
    topology = build_simple_uv_topology()
    outer = (topology.valid & (topology.part == 0) & (topology.layer == 1)).to(uv.device)
    top = outer & (topology.face.to(uv.device) == 4)
    cropped, region = crop_renderer(renderer, views, *target.shape[-2:])
    target = target[..., region[0], region[1]].float()
    foreground = foreground[..., region[0], region[1]].float()
    result = uv.clone()
    records = []
    for group in range(uv.shape[0]):
        original = uv[group:group + 1]
        semantic = torch.zeros_like(original)
        semantic[:, 3] = original[:, 3]
        # Red minus green cancels the equal RGB renderer background, yielding
        # the visible crown contribution, including bilinear/occlusion effects.
        identity = outer if crown_uv is None else (crown_uv[group] | top) & outer
        semantic[:, 0] = identity.float()
        sl = slice(group * len(views), (group + 1) * len(views))
        expected = target[sl] * foreground[sl]
        fg = foreground[sl]

        def objective(skins):
            rendered = torch.stack([cropped.forward_view(skins, view) for view in views], 1)
            probability = rendered[:, :, 0] - rendered[:, :, 1]
            # The silhouette term also protects crown teeth above bare hair.
            error = (probability - expected[None]).square()
            error += (rendered[:, :, 3] - fg[None]).square()
            return error.sum((1, 2, 3))

        score = float(objective(semantic))
        record = {'before_loss': score, 'removed': []}
        while True:
            active = (top & (semantic[0, 3] > .5)).nonzero()
            if not len(active):
                break
            scores = []
            for chunk in active.split(4):
                batch = semantic.repeat(len(chunk), 1, 1, 1)
                batch[torch.arange(len(chunk), device=uv.device), 3, chunk[:, 0], chunk[:, 1]] = 0
                scores.extend(objective(batch).tolist())
            best = min(range(len(scores)), key=scores.__getitem__)
            # A positive numerical margin prevents deleting unobserved cells
            # because of summation noise. It is not an occupancy prior.
            if scores[best] >= score - .05:
                break
            y, x = active[best].tolist()
            record['removed'].append({'uv': [x, y], 'loss_improvement': score - scores[best]})
            semantic[0, 3, y, x] = 0
            score = scores[best]
        record['after_loss'] = score
        records.append(record)
        result[group, 3] = semantic[0, 3]
    return result, records


@torch.no_grad()
def reconcile_crown_geometry(conditioning, details, renderer, views):
    from SkingToolkit.dense_uv_parser.headwear import headwear_uv_families
    from SkingToolkit.dense_uv_parser.infer import simple_inpaint_uv
    routing, outputs = details['routing'], details['outputs']
    required = ('headwear_presence_logits', 'headwear_logits')
    if len(views) < 2 or any(key not in outputs for key in required) or 'headwear_supported' not in routing:
        return conditioning
    royal = routing['headwear_supported'] & (routing['headwear_family'] == 4)
    active = royal.flatten(1).any(1).reshape(-1, len(views)).all(1)
    if not active.any():
        return conditioning
    # Use the same completed base alpha that the final renderer will see.
    # Material is discarded before comparison; RGB cannot influence geometry.
    uv = torch.stack([simple_inpaint_uv(c[None].cpu())[0] for c in conditioning]).to(conditioning.device)
    families = headwear_uv_families(routing, uv.shape[0], len(views))
    target = outputs['headwear_logits'].float().softmax(1)[:, 6]
    changed = torch.zeros_like(uv[:, 3], dtype=torch.bool)
    records = []
    for group in active.nonzero().flatten().tolist():
        sl = slice(group * len(views), (group + 1) * len(views))
        fixed, record = prune_crown_top(uv[group:group + 1], target[sl], routing['observed_foreground'][sl],
                                       renderer, views, crown_uv=families[group:group + 1] == 4)
        changed[group] = (uv[group, 3] > .5) & (fixed[0, 3] <= .5)
        records.append({'group': group, **record[0]})
    result = conditioning.clone()
    offset = 6 if conditioning.shape[1] == 12 else 5
    result[:, offset:] = result[:, offset:].masked_fill(changed[:, None], 0)
    details['headwear_removed_top_uv'] = changed
    details['crown_geometry'] = records
    return result
