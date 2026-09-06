"""Learned head accessory masks, object losses and geometry-safe routing.

Classes are object identities for one player: none, glasses, hat, outer hair.
RGB colour and projected-cell occupancy never create an accessory label.
"""
import torch
from torch import nn
import torch.nn.functional as F

ACCESSORY_CLASSES = ('none', 'glasses', 'hat', 'outer_hair')
HEAD_CROP = (0.0625, 0.0, 0.9375, 0.34375)  # fixed Steve camera ROI, not a mask


def head_bounds(height, width):
    x0, y0, x1, y1 = HEAD_CROP
    return round(y0*height), round(y1*height), round(x0*width), round(x1*width)


def head_crop(images, foreground=None):
    y0, y1, x0, x1 = head_bounds(*images.shape[-2:])
    rgb = images[:, :3]
    if foreground is None:
        foreground = torch.ones_like(rgb[:, :1])
    elif foreground.dim() == 3:
        foreground = foreground[:, None]
    foreground = foreground.float()
    rgb = rgb * foreground + 0.5 * (1-foreground)
    crop = torch.cat([rgb, foreground], 1)[:, :, y0:y1, x0:x1]
    return F.interpolate(crop.float(), (224, 224), mode='bilinear', align_corners=False)


def restore_logits(logits, size):
    h, w = size
    y0, y1, x0, x1 = head_bounds(h, w)
    patch = F.interpolate(logits, (y1-y0, x1-x0), mode='bilinear', align_corners=False)
    # Outside the crop the new branch is exactly inactive.
    prior = logits.new_tensor([0.]+[-20.]*(logits.shape[1]-1))[None, :, None, None]
    return F.pad(patch-prior, (x0, w-x1, y0, h-y1)) + prior


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout),
                         nn.SiLU(), nn.Conv2d(cout, cout, 3, padding=1),
                         nn.GroupNorm(8, cout), nn.SiLU())


class HeadAccessoryHead(nn.Module):
    def __init__(self, semantic_dim=768, predict_hat_components=False):
        super().__init__()
        self.enc0 = block(4, 24)
        self.enc1 = block(24, 48)
        self.enc2 = block(48, 96)
        self.enc3 = block(96, 96)
        self.semantic = nn.Conv2d(semantic_dim, 96, 1)
        self.context = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(96, 4, 256, 0.05, batch_first=True, norm_first=True),
            2, enable_nested_tensor=False)
        self.dec2 = block(192, 96)
        self.dec1 = block(144, 48)
        self.dec0 = block(72, 24)
        self.classifier = nn.Conv2d(24, 4, 1)
        self.components = nn.Conv2d(24,6,1) if predict_hat_components else None
        if self.components is not None:
            nn.init.zeros_(self.components.weight)
            nn.init.constant_(self.components.bias,-2.)
            with torch.no_grad():self.components.bias[0]=2.
        nn.init.constant_(self.classifier.bias, -2.)
        with torch.no_grad():
            self.classifier.bias[0] = 2.

    def forward(self, crop, semantic_features, return_components=False):
        s0 = self.enc0(crop)
        s1 = self.enc1(F.avg_pool2d(s0, 2))
        s2 = self.enc2(F.avg_pool2d(s1, 2))
        z = self.enc3(F.avg_pool2d(s2, 2))
        z = F.avg_pool2d(z, 2) + self.semantic(semantic_features.to(z.dtype))
        tokens = z.flatten(2).transpose(1, 2)
        if getattr(self, 'paired_context', False):
            # Adjacent front/back views of one player exchange semantic evidence.
            # Never mix players; incomplete groups are an input error.
            if tokens.shape[0] % 2:
                raise ValueError('Paired head semantics requires front/back groups')
            n, length, channels = tokens.shape
            tokens = tokens.reshape(n//2, 2, length, channels)
            tokens = tokens + self.view_embedding[None, :, None].to(tokens.dtype)
            tokens = self.context(tokens.reshape(n//2, 2*length, channels))
            tokens = tokens.reshape(n, length, channels)
        else:
            tokens = self.context(tokens)
        z = tokens.transpose(1, 2).reshape_as(z)
        z = self.dec2(torch.cat([F.interpolate(z, s2.shape[-2:], mode='bilinear', align_corners=False), s2], 1))
        z = self.dec1(torch.cat([F.interpolate(z, s1.shape[-2:], mode='bilinear', align_corners=False), s1], 1))
        z = self.dec0(torch.cat([F.interpolate(z, s0.shape[-2:], mode='bilinear', align_corners=False), s0], 1))
        logits=self.classifier(z)
        return (logits,self.components(z)) if return_components and self.components is not None else logits


def accessory_loss(logits, labels, valid=None):
    """Per-object Dice + class-balanced CE + boundary affinity, preserving holes."""
    if valid is None:
        valid = torch.ones_like(labels, dtype=torch.bool)
    target = labels.masked_fill(~valid, -100)
    ce = F.cross_entropy(logits.float(), target, weight=logits.new_tensor([1., 3., 2., 2.]).float())
    probability = logits.float().softmax(1)
    onehot = F.one_hot(labels.clamp(0, 3), 4).permute(0, 3, 1, 2).float()
    mask = valid[:, None]
    p, g = probability * mask, onehot * mask
    intersection = (p * g).sum((2, 3))
    dice = (1-(2*intersection+1)/(p.sum((2, 3))+g.sum((2, 3))+1))[:, 1:].mean()
    # Neighbour agreement targets also include object/background boundaries;
    # there is no indiscriminate smoothness term that fills lens/frame holes.
    affinity = logits.sum() * 0
    for axis in (2, 3):
        delta = torch.diff(p[:, 1:], dim=axis)
        truth_delta = torch.diff(g[:, 1:], dim=axis)
        affinity = affinity + F.smooth_l1_loss(delta, truth_delta)
    return ce + dice + .2 * affinity


def connected_object_support(probability, identity, valid, seed_threshold=.90, grow_threshold=0.0):
    """Keep a learned object region around confident seeds, across RGB changes.

    Growth is restricted to the SAME predicted object class and foreground.
    A low-confidence region without a seed cannot manufacture an accessory.
    """
    seeds = (probability >= seed_threshold) & (identity > 0) & valid
    grown = seeds.clone()
    for cls in (1,2,3):
        candidate = (identity == cls) & (probability >= grow_threshold) & valid
        current = seeds & (identity == cls)
        if not current.any():
            continue
        while True:
            neighbours = current.clone()
            neighbours[:,1:] |= current[:,:-1]
            neighbours[:,:-1] |= current[:,1:]
            neighbours[:,:,1:] |= current[:,:,:-1]
            neighbours[:,:,:-1] |= current[:,:,1:]
            updated = current | (candidate & neighbours)
            if torch.equal(updated,current):
                break
            current = updated
        grown |= current
    return grown, seeds


def apply_accessory_routing(routing, outputs, foreground, renderer, views, threshold=.80):
    """Promote learned object evidence even when the old route selected inner.

    This is an explicit object-evidence bypass of local texel voting. Actual
    foreground and valid head geometry are required; later silhouette vetoes
    and colour-background safeguards still apply.
    """
    from SkingToolkit.dense_uv_parser.utils import build_static_surface_routing
    threshold = float(outputs.get('accessory_route_threshold', threshold))
    logits = outputs.get('accessory_logits')
    support = torch.zeros_like(foreground)
    if logits is None:
        return support
    probability, identity = logits.float().softmax(1).max(1)
    valid = torch.zeros_like(foreground)
    for vi,view in enumerate(views):
        static = build_static_surface_routing(renderer,view,foreground.device)
        valid[vi::len(views)] = foreground[vi::len(views)] & static['masks'][1] & (static['part'][1] == 0)
    support, seeds = connected_object_support(probability,identity,valid,threshold)
    for vi, view in enumerate(views):
        sl = slice(vi, foreground.shape[0], len(views))
        static = build_static_surface_routing(renderer, view, foreground.device)
        accepted = support[sl]
        for name, value in [('layer', 1), ('foreground', True), ('surface', 1),
                            ('secondary', False), ('secondary_routed', False),
                            ('semantic_fallback', False), ('consensus_outer_gate_rejected', False)]:
            if name in routing:
                routing[name][sl] = torch.where(accepted, torch.full_like(routing[name][sl], value), routing[name][sl])
        for name in ('flat_uv', 'part', 'face', 'texel_center_score'):
            if name in routing:
                routing[name][sl] = torch.where(accepted, static[name][1], routing[name][sl])
        routing['confidence'][sl] = torch.where(accepted, probability[sl], routing['confidence'][sl])
        margin = (2*probability[sl]-1).clamp_min(0)
        for name, value in [('confidence_margin', margin), ('confidence_margin_ratio', margin/probability[sl].clamp_min(1e-6))]:
            if name in routing:
                routing[name][sl] = torch.where(accepted, value, routing[name][sl])
    # Learned inner crown/band evidence can also correct an old outer route.
    # This is restricted to explicitly trained components; no RGB rule is used.
    inner_support=torch.zeros_like(foreground)
    component_logits=outputs.get('hat_component_logits')
    if component_logits is not None:
        cp,ci=component_logits.float().softmax(1).max(1)
        candidate=((ci==1)|(ci==2))&(cp>=.95)&(identity==0)&foreground
        for vi,view in enumerate(views):
            sl=slice(vi,foreground.shape[0],len(views))
            static=build_static_surface_routing(renderer,view,foreground.device)
            accepted=candidate[sl]&static['masks'][0]&(static['part'][0]==0)
            inner_support[sl]=accepted
            for name,value in [('layer',0),('foreground',True),('surface',0),('secondary',False),('secondary_routed',False),('semantic_fallback',False)]:
                if name in routing:routing[name][sl]=torch.where(accepted,torch.full_like(routing[name][sl],value),routing[name][sl])
            for name in ('flat_uv','part','face','texel_center_score'):
                if name in routing:routing[name][sl]=torch.where(accepted,static[name][0],routing[name][sl])
            routing['confidence'][sl]=torch.where(accepted,cp[sl],routing['confidence'][sl])
            cm=(2*cp[sl]-1).clamp_min(0)
            for name,value in [('confidence_margin',cm),('confidence_margin_ratio',cm/cp[sl].clamp_min(1e-6))]:
                if name in routing:routing[name][sl]=torch.where(accepted,value,routing[name][sl])
    routing['hat_inner_supported']=inner_support
    routing['accessory_identity'] = identity
    routing['accessory_probability'] = probability
    routing['accessory_supported'] = support
    routing['accessory_seed'] = seeds
    from SkingToolkit.dense_uv_parser.ownership import apply_head_ownership
    apply_head_ownership(routing, outputs, foreground, renderer, views)
    from SkingToolkit.dense_uv_parser.headwear import apply_headwear_routing
    apply_headwear_routing(routing, outputs, foreground, renderer, views)
    if 'head_semantics_logits' in outputs:
        from SkingToolkit.dense_uv_parser.head_semantics import apply_joint_head_routing
        apply_joint_head_routing(routing, outputs, foreground, renderer, views)
    return routing['accessory_supported']
