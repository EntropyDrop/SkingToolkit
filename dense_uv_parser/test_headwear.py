"""Regression tests for component ownership rather than RGB heuristics."""
import unittest
import torch
from SkingToolkit.dense_uv_parser.headwear import component_decisions
from SkingToolkit.dense_uv_parser.headwear_data import make_headwear_skin


class HeadwearTests(unittest.TestCase):
    def test_unknown_base_never_copies_decorative_band(self):
        from SkingToolkit.dense_uv_parser.headwear import isolate_headwear_material
        uv=torch.ones(1,4,64,64);uv[:,:3]=torch.tensor([.9,.02,.02])[None,:,None,None]
        uv[0,:3,8,8]=.1
        conditioning=torch.zeros(1,12,64,64);conditioning[0,4,8,8]=1;conditioning[0,4,10,8]=1
        index=torch.tensor([8*64+8]*12+[10*64+8]*12).reshape(1,4,6).repeat(2,1,1)
        family=torch.tensor([1]*12+[2]*12).reshape(1,4,6).repeat(2,1,1)
        route={'headwear_supported':torch.ones_like(index,dtype=torch.bool),'headwear_family':family,
               'headwear_probability':torch.ones_like(index,dtype=torch.float32),
               'color_foreground':torch.ones_like(index,dtype=torch.bool),'part':torch.zeros_like(index),'flat_uv':index}
        fixed,_=isolate_headwear_material(uv,conditioning,{'routing':route},['front_left','back_left'])
        self.assertTrue(torch.equal(fixed[0,:3,11,8],torch.full((3,),.1)))
        self.assertTrue(torch.equal(fixed[0,:,10,8],uv[0,:,10,8]))
        self.assertTrue(torch.equal(fixed[:,:,16:],uv[:,:,16:]))
        self.assertTrue(torch.equal(fixed[:,:,:16,32:],uv[:,:,:16,32:]))
        self.assertTrue(torch.equal(fixed[:,3],uv[:,3]))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA renderer integration')
    def test_crown_visibility_and_routing_are_independent_of_colour(self):
        from SkingToolkit.renderer import DifferentiableRenderer
        from SkingToolkit.dense_uv_parser.headwear_data import render_headwear
        from SkingToolkit.dense_uv_parser.utils import splat_parser_predictions_to_uv_conditioning
        from SkingToolkit.dense_uv_parser.accessory_pipeline import load_pipeline
        renderer=DifferentiableRenderer('/home/ds/llms/github/differentiable_minecraft_renderer/mappings_256x512').cuda()
        for seed in range(100):
            uv,components=make_headwear_skin(torch.ones(4,64,64),seed)
            if (components==6).any():break
        images,truth=render_headwear(uv[None].cuda(),components[None].cuda(),renderer,['front_left','back_left'])
        self.assertFalse(((truth>0)&(images[:,3]<.5)).any())
        self.assertGreater(int((truth==6).sum()),100)
        images[:,:3]=.4;fg=images[:,3]>.5;n,h,w=truth.shape
        outputs={'layer':torch.zeros(n,3,h,w,device='cuda'),'foreground':torch.where(fg[:,None],12.,-12.),'affine':torch.zeros(n,3,device='cuda'),'accessory_logits':torch.zeros(n,4,h,w,device='cuda')}
        outputs['layer'][:,0]=24;outputs['accessory_logits'][:,0]=24
        outputs['headwear_logits']=torch.nn.functional.one_hot(truth,7).permute(0,3,1,2).float()*24-12
        _,details=splat_parser_predictions_to_uv_conditioning(images,outputs,renderer=renderer,observed_foreground=fg,return_details=True,**load_pipeline()['routing'])
        route=details['routing'];mask=route['headwear_supported']&(route['headwear_family']==4)
        self.assertGreater(int(mask.sum()),100)
        self.assertTrue((route['layer'][mask]==1).all())
        rejected=route['headwear_color_rejected']
        self.assertFalse((route['color_foreground']&rejected).any())
        outputs['headwear_presence_logits']=torch.tensor([[20.,20.],[20.,-20.]],device='cuda')
        _,blocked=splat_parser_predictions_to_uv_conditioning(images,outputs,renderer=renderer,observed_foreground=fg,return_details=True,**load_pipeline()['routing'])
        self.assertFalse(blocked['routing']['headwear_supported'].any())
        outputs['headwear_presence_logits'][:]=20.
        _,accepted=splat_parser_predictions_to_uv_conditioning(images,outputs,renderer=renderer,observed_foreground=fg,return_details=True,**load_pipeline()['routing'])
        self.assertTrue(accepted['routing']['headwear_supported'].any())
        # Optional branch ablation may retain scalar presence metadata.
        ablated={k:v for k,v in outputs.items() if k not in ('accessory_logits','headwear_logits')}
        splat_parser_predictions_to_uv_conditioning(images,ablated,renderer=renderer,observed_foreground=fg,return_details=True,**load_pipeline()['routing'])
        from SkingToolkit.dense_uv_parser.headwear import reconcile_crown_top
        accepted['outputs']['head_ownership_logits']=torch.zeros(n,5,h,w,device='cuda')
        accepted['outputs']['head_ownership_logits'][:,2]=torch.where(truth==0,24.,-24.)
        result=reconcile_crown_top(_,accepted,renderer,['front_left','back_left'])
        removed=(_[:,9]>.5)&~(result[:,9]>.5)
        self.assertFalse((removed & (components[None].cuda()==6)).any(),str(removed.nonzero().tolist()))
        self.assertTrue(torch.equal(result[:,:5],_[:,:5]))
        self.assertTrue(torch.equal(result[:,:,16:],_[:,:,16:]))

    def test_band_uses_one_layer_across_views_and_players_stay_separate(self):
        logits=torch.full((4,7,8,8),-10.)
        logits[:,0]=0
        logits[0,2,2:6]=8;logits[1,4,2:6]=7
        logits[2,4,2:6]=8;logits[3,4,2:6]=8
        support,layer,family,_=component_decisions(logits,torch.ones(4,8,8,dtype=torch.bool),2)
        self.assertTrue(support[:,2:6].all())
        self.assertEqual(layer[:2][support[:2]].unique().tolist(),[0])
        self.assertEqual(layer[2:][support[2:]].unique().tolist(),[1])
        self.assertTrue((family[support]==2).all())

    def test_crown_holes_and_absent_foreground_are_preserved(self):
        logits=torch.full((2,7,12,12),-10.);logits[:,0]=0
        logits[:,6,2:10,2:10]=8;logits[:,6,4:8,4:8]=-10
        fg=torch.ones(2,12,12,dtype=torch.bool);fg[:,2,2:10]=False
        support,layer,_,_=component_decisions(logits,fg,2)
        self.assertFalse(support[:,4:8,4:8].any());self.assertFalse(support[:,2].any())
        self.assertTrue((layer[support]==1).all())

    def test_low_confidence_speckles_do_not_create_component(self):
        logits=torch.full((2,7,8,8),-10.);logits[:,0]=0;logits[:,6,2:4,2:4]=8
        support,_,_,_=component_decisions(logits,torch.ones(2,8,8,dtype=torch.bool),2)
        self.assertFalse(support.any())

    def test_authored_layers_body_and_crown_open_top(self):
        source=torch.rand(4,64,64);crowns=bands=0
        for seed in range(80):
            uv,labels=make_headwear_skin(source,seed)
            self.assertTrue(torch.equal(uv[:,16:],source[:,16:]))
            for ids in ((1,3),(2,4)):
                self.assertFalse((labels==ids[0]).any() and (labels==ids[1]).any())
            if (labels==6).any():
                crowns+=1
                self.assertFalse((labels[:,:32]==6).any())
                self.assertFalse((labels[1:7,41:47]==6).any())
                self.assertTrue((uv[3][labels==6]==1).all())
            bands+=int(((labels==2)|(labels==4)).any())
        self.assertGreater(crowns,10);self.assertGreater(bands,3)


if __name__=='__main__':unittest.main()
