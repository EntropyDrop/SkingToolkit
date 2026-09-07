import unittest,io
import torch
if not __package__:
 from run_local import bind_checkout
 bind_checkout()
from SkingToolkit.dense_uv_parser.final_head_uv import FinalHeadUVDecoder,final_uv_loss
from SkingToolkit.dense_uv_parser.final_head_uv_revision import initialize_revision
from SkingToolkit.dense_uv_parser.head_topology_context import supervised_boundary_loss,augment_connected_uv

class TopologyTests(unittest.TestCase):
 def setUp(self):
  torch.set_num_threads(4);torch.manual_seed(104)
  self.config=dict(width=32,layers=1,revision=2,robust_edits=True,semantic_geometry=True,edit_risk_weight=5,mappings_dir='/home/ds/llms/github/differentiable_minecraft_renderer/mappings_256x512')
  self.model=FinalHeadUVDecoder(**self.config,topology_context=True,boundary_loss_weight=.5).eval()
  self.uv=torch.rand(1,4,64,64);self.uv[:,3]=(self.uv[:,3]>.5).float();self.uv[:,:3]*=self.uv[:,3:4];self.evidence=torch.rand(2,25,56,56)
 def test_v103_initialization_is_identical_and_unknown_missing_weights_fail(self):
  old=FinalHeadUVDecoder(**self.config).eval();checkpoint={'final_head_uv_state':old.state_dict()}
  initialize_revision(self.model,checkpoint)
  with torch.no_grad():
   a=old(self.uv,self.evidence);b=self.model(self.uv,self.evidence)
  for key in ['uv','alpha_logits','semantic_logits','mirror_link_logits','layer_link_logits']:self.assertTrue(torch.equal(a[key],b[key]),key)
  broken=dict(checkpoint['final_head_uv_state']);del broken['edit.weight']
  with self.assertRaises(RuntimeError):initialize_revision(self.model,{'final_head_uv_state':broken})
 def test_graph_crosses_cube_seams_but_not_false_atlas_neighbours(self):
  m=self.model;a=m.topology_adapter.within_face+m.topology_adapter.across_seam
  self.assertGreater(int((m.topology_adapter.across_seam>0).sum()),0)
  left,right=torch.where(a>0);self.assertTrue(torch.equal(m.outer[left],m.outer[right]))
  xyz=m.geometry[:,:3]*5;distance=torch.linalg.vector_norm(xyz[left]-xyz[right],dim=1)
  self.assertTrue(bool((distance<=1.201).all()))
  xy=torch.stack([m.ids%64,m.ids//64],1)
  atlas=(xy[:,None]-xy[None]).abs().sum(2)==1
  far=torch.cdist(xyz,xyz)>1.21
  candidates=atlas&far&(m.outer[:,None]==m.outer[None])
  self.assertTrue(bool(candidates.any()));self.assertFalse(bool((a[candidates]>0).any()))
 def test_true_boundaries_are_not_smoothed_away(self):
  edges=torch.tensor([[0,1,2],[1,2,3]])
  target=torch.tensor([[0.,0.,1.,1.]])
  good=supervised_boundary_loss(target,target,edges)
  erased=supervised_boundary_loss(torch.zeros_like(target),target,edges)
  fragment=supervised_boundary_loss(torch.tensor([[0.,1.,1.,1.]]),target,edges)
  self.assertLess(float(good),1e-4);self.assertGreater(float(erased),1);self.assertGreater(float(fragment),1)
  p=torch.tensor([[.1,.2,.8,.9]],requires_grad=True);loss=supervised_boundary_loss(p,target,edges);loss.backward();self.assertTrue(bool(torch.isfinite(p.grad).all()))
 def test_training_reaches_topology_adapter_and_preserves_body_and_inner_alpha(self):
  m=self.model;target=self.uv.clone();target[:,3,8:12,40:44]=1-target[:,3,8:12,40:44]
  p=m(self.uv,self.evidence);loss,_=final_uv_loss(m,p,self.uv,target,torch.full((1,64,64),-100,dtype=torch.long),torch.tensor([False]));loss.backward()
  self.assertGreater(float(m.topology_adapter.message[-1].weight.grad.abs().sum()),0)
  self.assertTrue(torch.equal(p['uv'][:,:,16:],self.uv[:,:,16:]))
  self.assertTrue(torch.equal(p['uv'].flatten(2)[:,3,m.ids[~m.outer]],self.uv.flatten(2)[:,3,m.ids[~m.outer]]))
 def test_structured_corruption_crosses_seams_without_touching_inner_or_body(self):
  m=self.model;changed_total=0
  for seed in range(12):
   torch.manual_seed(seed);source=self.uv.expand(8,-1,-1,-1).clone();result=augment_connected_uv(source.clone(),m)
   self.assertTrue(torch.equal(result[:,:,16:],source[:,:,16:]))
   self.assertTrue(torch.equal(result.flatten(2)[:,:,m.ids[~m.outer]],source.flatten(2)[:,:,m.ids[~m.outer]]))
   changed=(result[:,3]!=source[:,3]);changed_total+=int(changed.sum())
   self.assertLessEqual(int(changed.flatten(1).sum(1).max()),32)
  self.assertGreater(changed_total,20)
 def test_checkpoint_reload_preserves_learned_adapter(self):
  m=self.model
  with torch.no_grad():m.topology_adapter.message[-1].weight.normal_(0,.01)
  stream=io.BytesIO();torch.save(m.state_dict(),stream);stream.seek(0)
  restored=FinalHeadUVDecoder(**self.config,topology_context=True,boundary_loss_weight=.5).eval();restored.load_state_dict(torch.load(stream,weights_only=True))
  with torch.no_grad():self.assertTrue(torch.equal(m(self.uv,self.evidence)['uv'],restored(self.uv,self.evidence)['uv']))
if __name__=='__main__':unittest.main()
