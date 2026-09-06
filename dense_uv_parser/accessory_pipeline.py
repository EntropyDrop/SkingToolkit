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
    cond, details = splat_parser_predictions_to_uv_conditioning(
        images, outputs, renderer=renderer, observed_foreground=fg, observed_color_support=color_sources, return_details=True, **config['routing'])
    result = {'foreground_color_sources':color_sources,'foreground_probability':foreground_probability,'conditioning':cond,'details':details,'outputs':outputs,'foreground':fg}
    if complete:
        result['uv'] = torch.stack([simple_inpaint_uv(c[None].cpu())[0] for c in cond]).to(images.device)
        result['render'] = torch.stack([renderer.forward_view(result['uv'], view) for view in views],1).flatten(0,1)
    return result
