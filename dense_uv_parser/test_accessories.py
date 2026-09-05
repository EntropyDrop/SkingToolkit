import unittest
import torch
from SkingToolkit.dense_uv_parser.accessory_data import make_accessory_skin, render_accessories
from SkingToolkit.dense_uv_parser.accessories import accessory_loss, restore_logits, connected_object_support
from SkingToolkit.dense_uv_parser.accessory_pipeline import load_pipeline
from SkingToolkit.dense_uv_parser.utils import splat_parser_predictions_to_uv_conditioning
from SkingToolkit.dense_uv_parser.simple_inpainting import simple_symmetry_nearest_inpaint
from SkingToolkit.dense_uv_parser.uv_topology import build_simple_uv_topology
from SkingToolkit.renderer import DifferentiableRenderer


class AccessoryTests(unittest.TestCase):
    def test_object_growth_crosses_uncertain_bridge_but_not_classes_or_background(self):
        p=torch.zeros(1,8,16);identity=torch.zeros_like(p,dtype=torch.long)
        identity[:,3,2:12]=1;p[:,3,2:12]=.6;p[:,3,2]=.99
        identity[:,3,12:15]=2;p[:,3,12:15]=.7
        identity[:,6,3:8]=1;p[:,6,3:8]=.8
        valid=torch.ones_like(identity,dtype=torch.bool);valid[:,3,10]=False
        support,seeds=connected_object_support(p,identity,valid,.9)
        self.assertTrue(bool(support[:,3,2:10].all()))
        self.assertFalse(bool(support[:,3,10:].any()))
        self.assertFalse(bool(support[:,6].any()))
        self.assertEqual(int(seeds.sum()),1)

    def test_authored_ids_are_opaque_outer_objects(self):
        original=torch.ones(4,64,64)
        topology=build_simple_uv_topology()
        kinds=set()
        for seed in range(60):
            skin,identity=make_accessory_skin(original,seed)
            mask=identity>0;kinds.update(identity.unique().tolist())
            self.assertTrue(torch.all(skin[3][mask]==1))
            self.assertTrue(torch.all(topology.layer[mask]==1))
            self.assertTrue(torch.all(topology.part[mask]==0))
        self.assertEqual(kinds,{0,1,2,3})

    def test_outside_head_is_inactive(self):
        logits=restore_logits(torch.randn(2,4,224,224),(512,256))
        self.assertTrue(torch.all(logits[:,:,176:].argmax(1)==0))

    def test_missing_lens_is_penalized(self):
        target=torch.zeros(1,32,32,dtype=torch.long);target[:,10:20,5:12]=1;target[:,10:20,20:27]=1;target[:,13:15,12:20]=1
        correct=torch.nn.functional.one_hot(target,4).permute(0,3,1,2).float()*10
        broken=correct.clone();broken[:,:,10:20,5:12]=0;broken[:,0,10:20,5:12]=10
        self.assertGreater(float(accessory_loss(broken,target)),float(accessory_loss(correct,target))+.2)

    def test_inner_fill_never_copies_outer_only_source(self):
        uv=torch.zeros(4,64,64);uv[:,8,40]=torch.tensor([1.,0.,0.,1.])
        fixed,_=simple_symmetry_nearest_inpaint(uv)
        topology=build_simple_uv_topology()
        self.assertEqual(int((fixed[3][topology.layer==0]>.5).sum()),0)
        self.assertTrue(torch.equal(fixed[:,8,40],uv[:,8,40]))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA renderer integration')
    def test_semantic_evidence_rescues_inner_without_background_leak(self):
        renderer=DifferentiableRenderer('/home/ds/llms/github/differentiable_minecraft_renderer/mappings_256x512').cuda()
        config=load_pipeline()['routing'];views=config['views']
        uv,objects=make_accessory_skin(torch.ones(4,64,64),2)
        images,labels,fg=render_accessories(uv[None].cuda(),objects[None].cuda(),renderer,views)
        n,h,w=fg.shape
        logits=torch.full((n,3,h,w),-12.,device='cuda');logits[:,0]=12.
        output={'layer':logits,'foreground':torch.where(fg[:,None],12.,-12.),'affine':torch.zeros(n,3,device='cuda')}
        cfg={**config,'outer_uv_min_source_pixels':100000}
        old,details=splat_parser_predictions_to_uv_conditioning(images,output,renderer=renderer,observed_foreground=fg,return_details=True,**cfg)
        self.assertEqual(int((details['routing']['layer']==1).sum()),0)
        output['accessory_logits']=torch.nn.functional.one_hot(labels,4).permute(0,3,1,2).float()*24-12
        cond,details=splat_parser_predictions_to_uv_conditioning(images,output,renderer=renderer,observed_foreground=fg,return_details=True,**cfg)
        support=details['routing']['accessory_supported']
        self.assertGreater(int(support.sum()),0)
        self.assertTrue(torch.all(details['routing']['layer'][support]==1))
        self.assertFalse(bool((support&~fg).any()))
        self.assertGreater(int((cond[:,9]>.5).sum()),0)
        empty=torch.zeros_like(fg)
        _,details=splat_parser_predictions_to_uv_conditioning(images,output,renderer=renderer,observed_foreground=empty,return_details=True,**cfg)
        self.assertFalse(bool(details['routing']['foreground'].any()))


if __name__=='__main__':
    torch.set_num_threads(4)
    unittest.main()
