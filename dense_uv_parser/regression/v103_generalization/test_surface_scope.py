import sys,torch
sys.path.insert(0,'dense_uv_parser')
from run_local import bind_checkout
bind_checkout()
from SkingToolkit.dense_uv_parser.test_v102 import GeometryTests
from SkingToolkit.dense_uv_parser.head_semantics import apply_head_surface_routing
GeometryTests.setUpClass();g=GeometryTests;h,w=g.static[0]['masks'].shape[-2:];fg=torch.ones(2,h,w,device='cuda',dtype=torch.bool)
def route():
 r={k:torch.stack([s[k][1] for s in g.static]) for k in ['flat_uv','part','face','texel_center_score']}
 r.update(layer=torch.ones_like(fg,dtype=torch.long),surface=torch.ones_like(fg,dtype=torch.long),confidence=torch.zeros_like(fg,dtype=torch.float),accessory_supported=~fg,ownership_inner_supported=~fg)
 return r
for face in range(1,7):
 joint=torch.full((2,14,h,w),-20.,device='cuda');joint[:,4]=20
 surface=torch.full((2,7,h,w),-20.,device='cuda');surface[:,face]=20
 outputs={'head_semantics_logits':joint,'head_surface_logits':surface};default=route();before=default['flat_uv'].clone()
 apply_head_surface_routing(default,outputs,fg,g.renderer,g.views)
 if not (default['flat_uv']!=before).any():continue
 protected=route();apply_head_surface_routing(protected,{**outputs,'head_surface_routing_scope':'beard'},fg,g.renderer,g.views)
 assert torch.equal(protected['flat_uv'],before) and not protected['head_surface_supported'].any() and not protected['head_surface_uv_veto'].any()
 joint[:,4]=-20;joint[:,5]=20;beard=route();part=beard['part'].clone();apply_head_surface_routing(beard,{**outputs,'head_surface_routing_scope':'beard'},fg,g.renderer,g.views)
 assert beard['head_surface_supported'].any() and (beard['flat_uv']!=before).any()
 assert torch.equal(beard['flat_uv'][part!=0],before[part!=0])
 print('passed: preserve hair geometry, retain learned beard reassignment, isolate body');break
else:raise AssertionError('No nontrivial secondary face test found')
