"""The same foreground, routing, completion and renderer path for v101 eval/infer."""
import json
from pathlib import Path
import torch
from SkingToolkit.dense_uv_parser.foreground import build_parser_input
from SkingToolkit.dense_uv_parser.utils import estimate_top_left_flood_foreground, splat_parser_predictions_to_uv_conditioning
from SkingToolkit.dense_uv_parser.infer import simple_inpaint_uv


def load_pipeline(path=None):
    return json.loads(Path(path or Path(__file__).with_name('v101_pipeline.json')).read_text())


@torch.no_grad()
def run_pipeline(model, renderer, images, config=None, complete=False, outputs=None):
    config = config or load_pipeline()
    views = config['routing']['views']
    fg = estimate_top_left_flood_foreground(images, config['foreground_flood_tolerance'])
    actual = build_parser_input(images, fg, background_mode=config['foreground_background'])
    if outputs is None:
        outputs = model(actual, view_ids=torch.arange(images.shape[0], device=images.device) % len(views), semantic_foreground=fg)
    outputs = {**outputs, 'accessory_route_threshold':config['accessory_route_threshold']}
    cond, details = splat_parser_predictions_to_uv_conditioning(
        images, outputs, renderer=renderer, observed_foreground=fg, return_details=True, **config['routing'])
    result = {'conditioning':cond,'details':details,'outputs':outputs,'foreground':fg}
    if complete:
        result['uv'] = torch.stack([simple_inpaint_uv(c[None].cpu())[0] for c in cond]).to(images.device)
        result['render'] = torch.stack([renderer.forward_view(result['uv'], view) for view in views],1).flatten(0,1)
    return result
