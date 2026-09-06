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

    def test_authored_hat_band_and_brim_match_across_four_faces(self):
        original=torch.ones(4,64,64)
        checked=0
        for seed in range(60):
            skin,identity,components=make_accessory_skin(original,seed,return_components=True)
            if not (components==5).any():continue
            faces=[(40,8),(56,8),(32,8),(48,8)]
            masks=[components[y:y+8,x:x+8]==5 for x,y in faces]
            if not all(int(m.sum())==8 for m in masks):continue
            self.assertTrue(all(torch.equal(masks[0],m) for m in masks[1:]))
            bottom=int(masks[0].any(1).nonzero()[-1])
            rows=[skin[:3,y+bottom,x:x+8] for x,y in faces]
            self.assertTrue(all(torch.equal(rows[0],r) for r in rows[1:]))
            checked+=1
        self.assertGreater(checked,5)

    def test_hat_components_distinguish_inner_band_from_outer_brim(self):
        topology=build_simple_uv_topology();seen=set()
        for seed in range(80):
            uv,objects,parts=make_accessory_skin(torch.ones(4,64,64),seed,return_components=True)
            seen.update(parts.unique().tolist())
            inner=(parts==1)|(parts==2);outer=parts>=3
            self.assertTrue(bool((topology.layer[inner]==0).all()))
            self.assertTrue(bool((objects[inner]==0).all()))
            self.assertTrue(bool((topology.layer[outer]==1).all()))
            self.assertTrue(bool((objects[outer]==2).all()))
            self.assertTrue(bool((uv[3][parts>0]==1).all()))
        self.assertEqual(seen,set(range(6)))

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
    def test_material_fit_reduces_visible_error_without_changing_alpha_or_body(self):
        from SkingToolkit.dense_uv_parser.material_refine import refine_head_material
        renderer=DifferentiableRenderer('/home/ds/llms/github/differentiable_minecraft_renderer/mappings_256x512').cuda()
        topology=build_simple_uv_topology();head=(topology.valid&(topology.part==0)).cuda()
        target=torch.ones(1,4,64,64,device='cuda');target[:,:3]=.35
        target[0,:3,head]=torch.tensor([.9,.1,.2],device='cuda')[:,None]
        initial=target.clone();initial[0,:3,head]=.3
        views=['front_left','back_left']
        render=lambda skin:torch.stack([renderer.forward_view(skin,v) for v in views],1).flatten(0,1)
        images=render(target);source=images[:,3]>.99
        fitted=refine_head_material(initial,images,source,renderer,views,steps=10)
        self.assertTrue(torch.equal(initial[:,3],fitted[:,3]))
        self.assertTrue(torch.equal(initial[:,:,~head],fitted[:,:,~head]))
        before=(render(initial)[:,:3]-images[:,:3]).square().mean()
        after=(render(fitted)[:,:3]-images[:,:3]).square().mean()
        self.assertLess(float(after),float(before)*.15)
        empty=refine_head_material(initial,images,torch.zeros_like(source),renderer,views,steps=2)
        self.assertTrue(torch.equal(empty,initial))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA renderer integration')
    def test_learned_inner_band_corrects_outer_route_without_colour_rules(self):
        from SkingToolkit.dense_uv_parser.utils import build_static_surface_routing
        renderer=DifferentiableRenderer('/home/ds/llms/github/differentiable_minecraft_renderer/mappings_256x512').cuda()
        cfg=load_pipeline()['routing'];views=cfg['views']
        for seed in range(100):
            uv,objects,parts=make_accessory_skin(torch.ones(4,64,64),seed,return_components=True)
            if (parts==2).any():break
        images,labels,fg,parts=render_accessories(uv[None].cuda(),objects[None].cuda(),renderer,views,parts[None].cuda())
        n,h,w=fg.shape
        layer=torch.full((n,3,h,w),-12.,device='cuda');layer[:,1]=12.
        accessory=torch.full((n,4,h,w),-12.,device='cuda');accessory[:,0]=12.
        component=torch.nn.functional.one_hot(parts,6).permute(0,3,1,2).float()*24-12
        outputs={'layer':layer,'foreground':torch.where(fg[:,None],12.,-12.),'affine':torch.zeros(n,3,device='cuda'),
                 'accessory_logits':accessory,'hat_component_logits':component}
        _,detail=splat_parser_predictions_to_uv_conditioning(images,outputs,renderer=renderer,observed_foreground=fg,return_details=True,**cfg)
        route=detail['routing'];support=route['hat_inner_supported']
        self.assertGreater(int(support.sum()),0)
        self.assertTrue(bool((route['layer'][support]==0).all()))
        self.assertFalse(bool((support&~fg).any()))
        for vi,view in enumerate(views):
            static=build_static_surface_routing(renderer,view,images.device)
            self.assertTrue(torch.equal(route['flat_uv'][vi][support[vi]],static['flat_uv'][0][support[vi]]))

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
        _,no_color=splat_parser_predictions_to_uv_conditioning(images,output,renderer=renderer,observed_foreground=fg,observed_color_support=torch.zeros_like(fg),return_details=True,**cfg)
        self.assertTrue(torch.equal(no_color['routing']['foreground'],details['routing']['foreground']))
        self.assertFalse(bool(no_color['routing']['color_foreground'].any()))
        empty=torch.zeros_like(fg)
        _,details=splat_parser_predictions_to_uv_conditioning(images,output,renderer=renderer,observed_foreground=empty,return_details=True,**cfg)
        self.assertFalse(bool(details['routing']['foreground'].any()))


if __name__=='__main__':
    torch.set_num_threads(4)
    unittest.main()
