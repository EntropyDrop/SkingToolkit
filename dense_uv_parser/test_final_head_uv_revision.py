import io
import unittest
import torch
if not __package__:
    from run_local import bind_checkout
    bind_checkout()
from SkingToolkit.dense_uv_parser.final_head_uv import FinalHeadUVDecoder,evaluation_numerics,final_uv_loss
from SkingToolkit.dense_uv_parser.final_head_uv_revision import tie_material,relation_targets

MAPPINGS='/home/ds/llms/github/differentiable_minecraft_renderer/mappings_256x512'

class RevisionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4);torch.manual_seed(13);evaluation_numerics()
        self.config={'width':32,'layers':1,'revision':2,'mappings_dir':MAPPINGS}
        self.model=FinalHeadUVDecoder(**self.config).eval()
        self.uv=torch.rand(1,4,64,64);self.uv[:,3]=(self.uv[:,3]>.5).float();self.uv[:,:3]*=self.uv[:,3:4]
        self.evidence=torch.rand(2,25,56,56)

    def test_initial_copy_and_body_isolation(self):
        with torch.no_grad():prediction=self.model(self.uv,self.evidence)
        self.assertTrue(torch.equal(prediction['uv'],self.uv))
        with torch.no_grad():self.model.edit.bias.fill_(2.3456);self.model.color_gate.bias.fill_(1.234);self.model.color_delta.bias.fill_(.2)
        result=self.model(self.uv,self.evidence)['uv']
        self.assertTrue(torch.equal(result[:,:,16:],self.uv[:,:,16:]))
        self.assertTrue(bool(((result[:,3]==0)|(result[:,3]==1)).all()))
        invalid=torch.ones(4096,dtype=torch.bool);invalid[self.model.ids]=False
        self.assertTrue(torch.equal(result.flatten(2)[:,:,invalid],self.uv.flatten(2)[:,:,invalid]))

    def test_candidate_projection_keeps_front_and_back_distinct(self):
        projection=self.model.projection;sums=projection.sum(2)
        self.assertTrue(bool(((sums-1).abs()<1e-5).logical_or(sums==0).all()))
        ids=self.model.ids.tolist();front=ids.index(8*64+8);back=ids.index(8*64+24)
        self.assertGreater(float(sums[0,front]),.99);self.assertEqual(float(sums[1,front]),0.)
        self.assertGreater(float(sums[1,back]),.99);self.assertEqual(float(sums[0,back]),0.)
        # Absent outer UV still has candidate image evidence, not only base-visible cells.
        outer=ids.index(8*64+40);self.assertGreater(float(sums[0,outer]),.99)

    def test_learned_material_graph_exactly_ties_all_four_members(self):
        n=len(self.model.ids);rgb=torch.rand(1,n,3,requires_grad=True);alpha=torch.ones(1,n)
        mirror=torch.full((1,n),-20.);layer=mirror.clone();orbit=self.model.material_orbits[17]
        mirror[:,orbit]=20;layer[:,orbit]=20
        result=tie_material(self.model,rgb,alpha,mirror,layer)
        for i in orbit[1:]:self.assertTrue(torch.equal(result[:,orbit[0]],result[:,i]))
        mask=torch.ones(n,dtype=torch.bool);mask[orbit]=False
        self.assertTrue(torch.equal(result[:,mask],rgb[:,mask]))
        result[:,orbit[0]].sum().backward();self.assertTrue(bool((rgb.grad[:,orbit]>.0).all()))
        layer[:]=-20.;mirror[:]=-20.
        self.assertTrue(torch.equal(tie_material(self.model,rgb,alpha,mirror,layer),rgb))

    def test_links_are_conditional_and_preserve_asymmetric_examples(self):
        n=len(self.model.ids);truth=torch.zeros(1,n,4);classes=torch.full((1,n),-100,dtype=torch.long)
        orbit=self.model.material_orbits[20];truth[:,orbit]=torch.tensor([.1,.2,.3,1.])
        classes[:,orbit[:2]]=3;classes[:,orbit[2:]]=5
        mirror,layer=relation_targets(self.model,truth,classes,torch.tensor([True]))
        self.assertTrue(bool(mirror[:,orbit].all()));self.assertTrue(bool(layer[:,orbit].all()))
        mirror,_=relation_targets(self.model,truth,classes,torch.tensor([False]));self.assertFalse(bool(mirror.any()))
        truth[:,orbit[1],0]=.8
        mirror,layer=relation_targets(self.model,truth,classes,torch.tensor([True]))
        self.assertFalse(bool(mirror[:,orbit[0]]));self.assertFalse(bool(layer[:,orbit[1]]))

    def test_new_heads_receive_gradients_and_serialization_is_exact(self):
        target=self.uv.clone();target[:,3,8,40]=1-self.uv[:,3,8,40]
        labels=torch.full((1,64,64),-100,dtype=torch.long)
        prediction=self.model(self.uv,self.evidence);loss,_=final_uv_loss(self.model,prediction,self.uv,target,labels,torch.tensor([False]));loss.backward()
        self.assertGreater(float(self.model.edit.bias.grad.abs().sum()),0)
        self.assertGreater(float(self.model.mirror_link.bias.grad.abs().sum()),0)
        stream=io.BytesIO();torch.save(self.model.state_dict(),stream);stream.seek(0)
        restored=FinalHeadUVDecoder(**self.config).eval();restored.load_state_dict(torch.load(stream,weights_only=True))
        with torch.no_grad():self.assertTrue(torch.equal(self.model(self.uv,self.evidence)['uv'],restored(self.uv,self.evidence)['uv']))

if __name__=='__main__':unittest.main()
