import unittest
import torch
if not __package__:
    from run_local import bind_checkout
    bind_checkout()
from SkingToolkit.dense_uv_parser.final_head_uv import FinalHeadUVDecoder,final_uv_loss
from SkingToolkit.dense_uv_parser.final_head_uv_revision import decode_revision
from SkingToolkit.dense_uv_parser.prepare_generalization_revision import paired

class SemanticGeometryTests(unittest.TestCase):
 def setUp(self):
  torch.set_num_threads(4);torch.manual_seed(4)
  self.config=dict(width=32,layers=1,revision=2,mappings_dir='/home/ds/llms/github/differentiable_minecraft_renderer/mappings_256x512')
  self.model=FinalHeadUVDecoder(**self.config,semantic_geometry=True).eval()
 def test_occupancy_is_joint_semantic_probability_and_gradients_reach_both_heads(self):
  m=self.model;uv=torch.zeros(1,4,64,64);uv[:,3,:16,:32]=1
  p=m(uv,torch.rand(2,25,56,56));outer=m.outer
  prob=1-p['semantic_logits'].softmax(2)[:,:,0]
  self.assertTrue(torch.allclose(p['alpha_probability'][:,outer],prob[:,outer],atol=2e-6))
  labels=torch.full((1,64,64),-100,dtype=torch.long)
  loss,_=final_uv_loss(m,p,uv,uv,labels,torch.tensor([False]));loss.backward()
  self.assertGreater(float(m.semantic.bias.grad.abs().sum()),0)
  self.assertGreater(float(m.edit.bias.grad.abs().sum()),0)
  self.assertTrue(torch.equal(p['uv'][:,:,16:],uv[:,:,16:]))
 def test_semantic_absence_and_hair_evidence_can_correct_an_edit_prior(self):
  m=self.model;features=torch.zeros(1,len(m.ids),32);uv=torch.zeros(1,len(m.ids),4)
  with torch.no_grad():
   m.edit.bias.fill_(3);m.semantic.bias.fill_(-20);m.semantic.bias[0]=20
   p=decode_revision(m,features,uv)
   self.assertEqual(int(p['alpha'][:,m.outer].sum()),0)
   uv[:,:,3]=1;m.semantic.bias.fill_(-20);m.semantic.bias[4]=20
   p=decode_revision(m,features,uv)
   self.assertTrue(bool((p['alpha'][:,m.outer]==1).all()))
 def test_legacy_default_keeps_original_edit_probability(self):
  m=FinalHeadUVDecoder(**self.config).eval();features=torch.randn(1,len(m.ids),32);uv=torch.rand(1,len(m.ids),4);uv[:,:,3]=(uv[:,:,3]>.5).float()
  p=decode_revision(m,features,uv);expected=torch.where(uv[:,:,3]>.5,-m.edit(features).squeeze(2),m.edit(features).squeeze(2)).sigmoid()
  self.assertTrue(torch.equal(expected,p['alpha_probability']))
 def test_paired_loss_rejects_changes_outside_an_accessory(self):
  from SkingToolkit.dense_uv_parser.final_head_uv_revision import paired_occupancy_loss
  m=self.model;target=torch.zeros(3,4,64,64);target[1,3,12,42]=1;target[2,3,11,35]=1
  truth=target.flatten(2)[:,3,m.ids]
  self.assertEqual(float(paired_occupancy_loss(m,{'alpha_probability':truth},target)),0)
  p=truth.clone();i=m.ids.tolist().index(12*64+45);p[1,i]=.8;p.requires_grad_()
  loss=paired_occupancy_loss(m,{'alpha_probability':p},target);self.assertGreater(float(loss.detach()),0);loss.backward();self.assertGreater(float(p.grad[1,i]),0)
 def test_counterfactuals_change_only_the_added_accessory(self):
  original=torch.ones(4,64,64);original[:3]=.2
  for seed in [17,88,111]:
   base,labels=paired(original,seed,'bare')
   self.assertFalse(bool(((labels==6)|(labels==7)).any()))
   for variant,cls in [('glasses',6),('phones',7)]:
    uv,target=paired(original,seed,variant);mask=target==cls
    self.assertTrue(bool(mask.any()));self.assertTrue(torch.equal(uv[:,~mask],base[:,~mask]));self.assertTrue(torch.equal(target[~mask],labels[~mask]))
if __name__=='__main__':unittest.main()
