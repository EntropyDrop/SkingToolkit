import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from SkingToolkit.dense_uv_parser.accessories import head_bounds
from SkingToolkit.dense_uv_parser.final_head_material import refine_final_head_uv


class LinearHeadRenderer:
    def forward_view(self,uv,view):
        out=uv.new_zeros(len(uv),4,32,32)
        y0,y1,x0,x1=head_bounds(32,32)
        for i,cell in enumerate((520,527,552,559)):
            a=x0+(x1-x0)*i//4;b=x0+(x1-x0)*(i+1)//4
            rgb=uv.flatten(2)[:,:3,cell]*uv.flatten(2)[:,3:4,cell]
            out[:,:3,y0:y1,a:b]=rgb[:,:,None,None]
        return out


class FinalHeadMaterialTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        ids=torch.tensor([520,527,552,559])
        self.decoder=SimpleNamespace(revision=2,ids=ids,material_orbits=torch.tensor([[0,1,2,3]]),material_orbit_id=torch.zeros(4,dtype=torch.long),material_slot=torch.arange(4))
        self.model=SimpleNamespace(final_head_uv_decoder=self.decoder);self.renderer=LinearHeadRenderer()
        self.uv=torch.zeros(1,4,64,64);self.uv[:,3]=1;self.uv[:,:3]=.7
        self.target=self.uv.clone();self.target.flatten(2)[:,:3,ids]=.2
        self.result={'uv':self.uv.clone(),'details':{'rendered':self.renderer.forward_view(self.target,'front'),'color_source_support':torch.ones(1,32,32,dtype=torch.bool)},'final_head_uv':{'mirror_link_logits':torch.full((1,4),20.),'layer_link_logits':torch.full((1,4),20.)}}
    def test_actual_fit_preserves_geometry_body_and_shared_colours(self):
        refine_final_head_uv(self.model,self.result,self.renderer,['front'],32)
        after=self.result['uv'];info=self.result['final_head_material_refit']
        self.assertTrue(info['accepted']);self.assertLess(info['source_mse_final'],info['source_mse_before']*.02)
        self.assertTrue(torch.equal(after[:,3],self.uv[:,3]));self.assertTrue(torch.equal(after[:,:,16:],self.uv[:,:,16:]))
        rgb=after.flatten(2)[:,:3,self.decoder.ids];self.assertTrue(torch.equal(rgb,rgb[:,:,:1].expand_as(rgb)))
    def test_no_source_is_identity(self):
        self.result['details']['color_source_support'].zero_()
        refine_final_head_uv(self.model,self.result,self.renderer,['front'],32)
        self.assertTrue(torch.equal(self.result['uv'],self.uv))
    def test_accessory_sources_do_not_repaint_inner_material(self):
        ownership=torch.zeros(1,5,32,32);ownership[:,3]=20
        self.result['details']['outputs']={'head_color_ownership_logits':ownership}
        self.result['final_head_uv']['layer_link_logits'].fill_(-20)
        refine_final_head_uv(self.model,self.result,self.renderer,['front'],32)
        after=self.result['uv'].flatten(2)
        self.assertTrue(self.result['final_head_material_refit']['accepted'])
        self.assertTrue(torch.equal(after[:,:3,self.decoder.ids[:2]],self.uv.flatten(2)[:,:3,self.decoder.ids[:2]]))
        self.assertLess(float(after[:,:3,self.decoder.ids[2:]].mean()),.3)
    def test_worse_fit_is_rejected(self):
        candidate=self.uv.clone();candidate[:,:3]=.95
        with patch('SkingToolkit.dense_uv_parser.final_head_material.refine_head_material',return_value=candidate):
            # Only head colour modifications are permitted.
            candidate[:,:,16:]=self.uv[:,:,16:]
            refine_final_head_uv(self.model,self.result,self.renderer,['front'],32)
        self.assertFalse(self.result['final_head_material_refit']['accepted']);self.assertTrue(torch.equal(self.result['uv'],self.uv))
    def test_partial_accessory_footprint_preserves_only_affected_inner_cell(self):
        y0,y1,x0,x1=head_bounds(32,32)
        ownership=torch.full((1,5,32,32),-20.);ownership[:,1]=20
        ownership[:,:,y0,x0]=-20;ownership[:,3,y0,x0]=20
        self.result['details']['outputs']={'head_color_ownership_logits':ownership}
        for links in self.result['final_head_uv'].values():links.fill_(-20)
        refine_final_head_uv(self.model,self.result,self.renderer,['front'],32,protect_inner_footprints=True)
        rgb=self.result['uv'].flatten(2)[:,:3,self.decoder.ids]
        self.assertTrue(torch.equal(rgb[:,:,0],self.uv.flatten(2)[:,:3,self.decoder.ids[0]]))
        # One classified pixel protects its entire UV footprint; unrelated
        # newly exposed inner hair and outer material still follow the source.
        self.assertLess(float(rgb[:,:,1:].mean()),.3)
        self.assertTrue(torch.equal(self.result['uv'][:,3],self.uv[:,3]))
        self.assertTrue(torch.equal(self.result['uv'][:,:,16:],self.uv[:,:,16:]))
    def test_partial_footprint_without_protection_reproduces_contamination(self):
        y0,y1,x0,x1=head_bounds(32,32)
        excluded=torch.zeros(1,32,32,dtype=torch.bool);excluded[:,y0,x0]=True
        self.result['details']['routing']={'head_color_boundary_excluded':excluded}
        for links in self.result['final_head_uv'].values():links.fill_(-20)
        refine_final_head_uv(self.model,self.result,self.renderer,['front'],32)
        self.assertLess(float(self.result['uv'].flatten(2)[:,:3,self.decoder.ids[0]].mean()),.3)
    def test_boundary_footprint_also_protects_inner_cell(self):
        y0,y1,x0,x1=head_bounds(32,32)
        excluded=torch.zeros(1,32,32,dtype=torch.bool);excluded[:,y0,x0]=True
        self.result['details']['routing']={'head_color_boundary_excluded':excluded}
        for links in self.result['final_head_uv'].values():links.fill_(-20)
        refine_final_head_uv(self.model,self.result,self.renderer,['front'],32,protect_inner_footprints=True)
        rgb=self.result['uv'].flatten(2)[:,:3,self.decoder.ids]
        self.assertTrue(torch.equal(rgb[:,:,0],self.uv.flatten(2)[:,:3,self.decoder.ids[0]]))
        self.assertLess(float(rgb[:,:,1:].mean()),.3)
    def test_geometry_and_body_changes_fail(self):
        for channel,y in [(3,8),(0,20)]:
            candidate=self.uv.clone();candidate[:,channel,y,8]=0
            with patch('SkingToolkit.dense_uv_parser.final_head_material.refine_head_material',return_value=candidate):
                with self.assertRaises(RuntimeError):refine_final_head_uv(self.model,self.result,self.renderer,['front'],32)


if __name__=='__main__':unittest.main()
