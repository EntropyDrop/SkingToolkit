import unittest
import torch
from SkingToolkit.dense_uv_parser.ownership_data import make_ownership_skin,render_ownership
from SkingToolkit.dense_uv_parser.uv_topology import build_simple_uv_topology


class OwnershipTests(unittest.TestCase):
    def test_authored_ownership_has_correct_layers_and_preserves_body(self):
        original=torch.rand(4,64,64);topology=build_simple_uv_topology();seen=set()
        for seed in range(120):
            uv,labels=make_ownership_skin(original,seed);seen.update(labels.unique().tolist())
            self.assertTrue(torch.equal(uv[:,16:],original[:,16:]))
            self.assertTrue(bool((topology.layer[(labels==1)|(labels==2)]==0).all()))
            self.assertTrue(bool((topology.layer[(labels==3)|(labels==4)]==1).all()))
            self.assertTrue(bool((uv[3][labels>0]==1).all()))
        self.assertEqual(seen,set(range(5)))

    def test_hidden_hair_never_uses_green_face_or_recursive_fill_as_source(self):
        from SkingToolkit.dense_uv_parser.head_inpainting import repair_hidden_head_material
        uv=torch.ones(1,4,64,64);uv[:,:3]=torch.tensor([.1,.9,.2])[None,:,None,None]
        known=[8*64,8*64+7,12*64+9]
        uv[0,:3,8,0]=torch.tensor([.08,.06,.04]);uv[0,:3,8,7]=torch.tensor([.12,.1,.08])
        conditioning=torch.zeros(1,12,64,64)
        for index in known:conditioning[0,4,index//64,index%64]=1
        flat=torch.tensor(known).repeat_interleave(12).reshape(1,6,6).repeat(2,1,1)
        label=torch.full((2,6,6),2);label.flatten(1)[:,24:]=1
        logits=torch.nn.functional.one_hot(label,5).permute(0,3,1,2).float()*24-12
        routing={'headphone_colour_rejected':torch.ones(2,6,6,dtype=torch.bool),'color_foreground':torch.ones(2,6,6,dtype=torch.bool),'layer':torch.zeros(2,6,6,dtype=torch.long),'part':torch.zeros(2,6,6,dtype=torch.long),'flat_uv':flat}
        fixed=repair_hidden_head_material(uv,conditioning,{'routing':routing,'outputs':{'head_ownership_logits':logits}},['front_left','back_left'])
        self.assertLess(float(fixed[0,1,11,3]),.2)
        self.assertTrue(torch.equal(fixed[0,:,12,9],uv[0,:,12,9]))
        self.assertTrue(torch.equal(fixed[:,:,16:],uv[:,:,16:]))
        self.assertTrue(torch.equal(fixed[:,:,:16,32:],uv[:,:,:16,32:]))
        self.assertTrue(torch.equal(fixed[:,3],uv[:,3]))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA renderer integration')
    def test_authored_visibility_never_teaches_occluded_inner_as_headphones(self):
        from SkingToolkit.renderer import DifferentiableRenderer
        renderer=DifferentiableRenderer('/home/ds/llms/github/differentiable_minecraft_renderer/mappings_256x512').cuda()
        for seed in range(120):
            uv,labels=make_ownership_skin(torch.ones(4,64,64),seed)
            if (labels==4).any():break
        images,target=render_ownership(uv[None].cuda(),labels[None].cuda(),renderer,['front_left','back_left'])
        self.assertGreater(int((target==4).sum()),100)
        self.assertGreater(int((target==2).sum()),100)
        self.assertFalse(bool(((target>0)&(images[:,3]<.5)).any()))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA renderer integration')
    def test_face_corrects_outer_and_headphones_correct_inner_with_same_rgb(self):
        from SkingToolkit.renderer import DifferentiableRenderer
        from SkingToolkit.dense_uv_parser.utils import splat_parser_predictions_to_uv_conditioning
        from SkingToolkit.dense_uv_parser.accessory_pipeline import load_pipeline
        renderer=DifferentiableRenderer('/home/ds/llms/github/differentiable_minecraft_renderer/mappings_256x512').cuda()
        for seed in range(120):
            uv,labels=make_ownership_skin(torch.ones(4,64,64),seed)
            if (labels==4).any():break
        images,target=render_ownership(uv[None].cuda(),labels[None].cuda(),renderer,['front_left','back_left'])
        n,h,w=target.shape;images[:,:3]=.4;fg=images[:,3]>.5
        wrong=torch.where(target==4,0,1)
        outputs={'layer':torch.nn.functional.one_hot(wrong,3).permute(0,3,1,2).float()*24-12,'foreground':torch.where(fg[:,None],12.,-12.),'affine':torch.zeros(n,3,device='cuda'),'accessory_logits':torch.zeros(n,4,h,w,device='cuda')}
        outputs['accessory_logits'][:,0]=24
        outputs['head_ownership_logits']=torch.nn.functional.one_hot(target,5).permute(0,3,1,2).float()*24-12
        _,details=splat_parser_predictions_to_uv_conditioning(images,outputs,renderer=renderer,observed_foreground=fg,return_details=True,**load_pipeline()['routing'])
        r=details['routing'];inner=r['ownership_inner_supported'];outer=r['ownership_outer_supported']
        self.assertGreater(int(inner.sum()),0);self.assertGreater(int(outer.sum()),0)
        self.assertTrue(bool((r['layer'][inner]==0).all()));self.assertTrue(bool((r['layer'][outer]==1).all()))
        self.assertFalse(bool((r['accessory_supported']&inner).any()))
        config={**load_pipeline()['routing'],'head_color_separation_inset':2}
        _,separated=splat_parser_predictions_to_uv_conditioning(images,outputs,renderer=renderer,observed_foreground=fg,return_details=True,**config)
        sr=separated['routing'];guard=sr['head_color_boundary_excluded']
        self.assertTrue(torch.equal(sr['foreground'],r['foreground']))
        self.assertGreater(int(guard.sum()),0)
        self.assertFalse(bool((sr['color_foreground']&guard).any()))
        self.assertTrue(torch.equal(sr['color_foreground'][sr['part']!=0],r['color_foreground'][r['part']!=0]))
        # One isolated confident view cannot add an object rejected by its pair.
        outputs['headphone_presence_logit']=torch.tensor([20.,-20.],device='cuda')
        outputs['headphone_presence_consensus']=True
        outputs['headphone_presence_threshold']=.99
        _,blocked=splat_parser_predictions_to_uv_conditioning(images,outputs,renderer=renderer,observed_foreground=fg,return_details=True,**config)
        self.assertFalse(bool(blocked['routing']['ownership_outer_supported'].any()))
        outputs['headphone_presence_logit'][:]=20.
        _,accepted=splat_parser_predictions_to_uv_conditioning(images,outputs,renderer=renderer,observed_foreground=fg,return_details=True,**config)
        self.assertTrue(bool(accepted['routing']['ownership_outer_supported'].any()))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA renderer integration')
    def test_material_does_not_fit_inner_hair_to_semantic_accessory_pixels(self):
        from SkingToolkit.renderer import DifferentiableRenderer
        from SkingToolkit.dense_uv_parser.material_refine import refine_head_material
        renderer=DifferentiableRenderer('/home/ds/llms/github/differentiable_minecraft_renderer/mappings_256x512').cuda();t=build_simple_uv_topology()
        uv=torch.zeros(1,4,64,64,device='cuda');valid=t.valid.cuda();inner=(t.layer==0).cuda()&valid;uv[:,3,inner]=1;uv[:,:3]=.15
        views=['front_left','back_left'];images=torch.stack([renderer.forward_view(uv,v) for v in views],1).flatten(0,1)
        images[:,:3]=images.new_tensor([.1,.9,.2])[None,:,None,None]
        sources=images[:,3]>.5;ownership=torch.zeros(2,5,*sources.shape[-2:],device='cuda');ownership[:,4]=24
        fitted=refine_head_material(uv,images,sources,renderer,views,steps=3,ownership=ownership)
        self.assertTrue(torch.equal(fitted,uv))
        unsupported=refine_head_material(uv,images,sources,renderer,views,steps=3,inner_texel_support=torch.zeros(1,64,64,device='cuda',dtype=torch.bool))
        self.assertTrue(torch.equal(unsupported,uv))


if __name__=='__main__':
    torch.set_num_threads(4);unittest.main()
