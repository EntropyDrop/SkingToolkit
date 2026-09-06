"""The same foreground, routing, completion and renderer path for v101 eval/infer."""
import json
from pathlib import Path
import torch
from SkingToolkit.dense_uv_parser.foreground import build_parser_input,learned_foreground_support
from SkingToolkit.dense_uv_parser.utils import estimate_top_left_flood_foreground, splat_parser_predictions_to_uv_conditioning
from SkingToolkit.dense_uv_parser.infer import simple_inpaint_uv


def load_pipeline(path=None):
    return json.loads(Path(path or Path(__file__).with_name('v101_pipeline.json')).read_text())


@torch.no_grad()
def run_pipeline(model, renderer, images, config=None, complete=False, outputs=None, foreground_probability=None):
    config = config or load_pipeline()
    views = config['routing']['views']
    color_sources=None
    if foreground_probability is None:
        fg = estimate_top_left_flood_foreground(images, config['foreground_flood_tolerance'])
    else:
        if foreground_probability.shape != images.shape[:1]+images.shape[-2:]:
            raise ValueError('Foreground probability must have shape (N,H,W)')
        probability=foreground_probability.to(device=images.device,dtype=torch.float32)
        if not torch.isfinite(probability).all() or probability.min()<0 or probability.max()>1:
            raise ValueError('Foreground probability must be finite in [0,1]')
        fg,color_sources=learned_foreground_support(probability,config.get('foreground_probability_threshold',.5),config.get('foreground_source_threshold',.98),config.get('foreground_source_inset',1))
    actual = build_parser_input(images, fg, background_mode=config['foreground_background'])
    if outputs is None:
        outputs = model(actual, view_ids=torch.arange(images.shape[0], device=images.device) % len(views), semantic_foreground=fg)
    outputs = {**outputs, 'accessory_route_threshold':config['accessory_route_threshold']}
    outputs['headwear_presence_threshold']=config.get('headwear_presence_threshold',.95)
    outputs['headphone_presence_threshold']=config.get('headphone_presence_threshold',.9)
    outputs['headphone_presence_consensus']=config.get('headphone_presence_consensus',False)
    if 'head_color_ownership_logits' in outputs:
        # Preserve the validated phone colour guard when the frozen presence
        # detector accepts real headphones. Geometry uses the joint taxonomy.
        probability=outputs['headphone_presence_logit'].sigmoid().reshape(-1,len(views)).amin(1).repeat_interleave(len(views))
        accepted=probability>=outputs['headphone_presence_threshold']
        outputs['head_ownership_logits']=torch.where(accepted[:,None,None,None],outputs['head_color_ownership_logits'],outputs['head_ownership_logits'])
        outputs['head_color_ownership_logits']=outputs['head_ownership_logits']
        outputs['joint_phone_expert_accepted']=accepted
    mode=config.get('headphone_routing_mode','unrestricted')
    def splat(current):
        return splat_parser_predictions_to_uv_conditioning(
            images,current,renderer=renderer,observed_foreground=fg,observed_color_support=color_sources,return_details=True,**config['routing'])
    if mode=='existing_uv' and 'head_ownership_logits' in outputs:
        # The new headphone head may recover colour ownership inside geometry
        # already accepted by v101. It cannot manufacture new outer occupancy.
        initial,_=splat({**outputs,'headphone_routing_mode':'disabled'})
        outputs={**outputs,'headphone_routing_mode':'existing_uv','headphone_uv_support':(initial[:,9]>.5).flatten(1)}
    else:outputs={**outputs,'headphone_routing_mode':mode}
    cond,details=splat(outputs)
    surface_veto=details['routing'].get('head_surface_uv_veto')
    if surface_veto is not None:
        offset=6 if cond.shape[1]==12 else 5
        cond[:,offset:]=cond[:,offset:].masked_fill(surface_veto[:,None],0)
    if 'head_semantics_logits' in outputs:
        from SkingToolkit.dense_uv_parser.head_semantics import reconcile_beard_alignment
        cond=reconcile_beard_alignment(cond,details,views)
    geometry_mode = config.get('crown_top_geometry_mode', 'legacy_consensus')
    if geometry_mode == 'rendered_semantics':
        from SkingToolkit.dense_uv_parser.crown_geometry import reconcile_crown_geometry
        cond = reconcile_crown_geometry(cond, details, renderer, views)
    elif geometry_mode == 'legacy_consensus':
        from SkingToolkit.dense_uv_parser.headwear import reconcile_crown_top
        cond = reconcile_crown_top(cond, details, renderer, views)
    else:
        raise ValueError('Unknown crown top geometry mode: ' + str(geometry_mode))
    if 'head_semantics_logits' in outputs:
        from SkingToolkit.dense_uv_parser.head_semantics import reconcile_joint_head_geometry
        cond = reconcile_joint_head_geometry(cond, details, renderer, views)
    result = {'foreground_color_sources':color_sources,'foreground_probability':foreground_probability,'conditioning':cond,'details':details,'outputs':outputs,'foreground':fg}
    if complete:
        result['uv'] = torch.stack([simple_inpaint_uv(c[None].cpu())[0] for c in cond]).to(images.device)
        if config.get('head_semantic_inpaint',False):
            from SkingToolkit.dense_uv_parser.head_inpainting import repair_hidden_head_material
            result['uv']=repair_hidden_head_material(result['uv'],cond,details,views)
        from SkingToolkit.dense_uv_parser.headwear import isolate_headwear_material
        result['uv'],headwear_uv=isolate_headwear_material(result['uv'],cond,details,views)
        if config.get('head_material_refine_steps',0):
            from SkingToolkit.dense_uv_parser.material_refine import refine_head_material
            if config.get('head_material_refine_mode')!='isolated':
                from SkingToolkit.dense_uv_parser.material_refine import refine_head_material_legacy
                result['uv']=refine_head_material_legacy(result['uv'],details['rendered'],details['color_source_support'],renderer,views,config['head_material_refine_steps'])
            else:
                result['uv']=refine_head_material(result['uv'],details['rendered'],details['color_source_support'],renderer,views,config['head_material_refine_steps'],ownership=details['outputs'].get('head_color_ownership_logits',details['outputs'].get('head_ownership_logits')),inner_exclusion=details['routing'].get('head_color_boundary_excluded'),inner_texel_support=cond[:,4]>.5,headwear_layer=details['routing'].get('headwear_layer'),headwear_uv=headwear_uv,headwear_family=details['routing'].get('headwear_family'))
        if 'head_semantics_logits' in outputs:
            from SkingToolkit.dense_uv_parser.head_semantics import complete_aligned_beard_material
            result['uv']=complete_aligned_beard_material(result['uv'],details)
        result['render'] = torch.stack([renderer.forward_view(result['uv'], view) for view in views],1).flatten(0,1)
    return result


def cached_real_foreground(images, source, config):
    """Load verified learned foreground for real development evaluation."""
    import hashlib
    import numpy as np
    from PIL import Image
    import torch.nn.functional as F
    cache=config.get('foreground_cache')
    if cache is None:return None
    source=Path(source).resolve();entry=cache.get(str(source))
    if entry is None:raise ValueError('Missing learned foreground for '+str(source))
    data=Path(entry['probability']).read_bytes()
    if hashlib.sha256(source.read_bytes()).hexdigest()!=entry['input_sha256'] or hashlib.sha256(data).hexdigest()!=entry['probability_sha256']:
        raise ValueError('Learned foreground cache changed')
    import io
    p=torch.from_numpy(np.array(Image.open(io.BytesIO(data))).copy()).float().to(images.device)/255.
    return F.interpolate(torch.stack(p.chunk(len(config['routing']['views']),1))[:,None],images.shape[-2:],mode='nearest-exact')[:,0]
