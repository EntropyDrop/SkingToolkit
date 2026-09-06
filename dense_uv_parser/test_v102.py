"""Joint identity, paired context, authored geometry and routing regressions."""
import unittest
import torch
from SkingToolkit.dense_uv_parser.head_semantics import HeadSemanticsHead,project_semantics,apply_joint_head_routing
from SkingToolkit.dense_uv_parser.head_semantics_data import make_joint_skin,render_joint_batch
from SkingToolkit.dense_uv_parser.uv_topology import build_simple_uv_topology


class SemanticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(4)

    def test_semantics_are_mutually_exclusive(self):
        logits=torch.randn(4,14,16,16)
        own=project_semantics(logits,'ownership').softmax(1)
        hw=project_semantics(logits,'headwear').softmax(1)
        self.assertTrue(torch.all(own[:,1]+hw[:,6]<=1+1e-6))
        self.assertTrue(torch.allclose(own[:,1],logits.softmax(1)[:,[1,3]].sum(1),atol=1e-6))
        self.assertTrue(torch.allclose(hw[:,6],logits.softmax(1)[:,13],atol=1e-6))

    def test_affine_interpolation_keeps_one_categorical_distribution(self):
        from SkingToolkit.dense_uv_parser.utils import canonicalize_parser_outputs
        joint=torch.randn(2,14,16,16)*5
        affine=torch.tensor([[.02,-.015,.01]]).repeat(2,1)
        result=canonicalize_parser_outputs({'head_semantics_logits':joint,'headwear_logits':project_semantics(joint,'headwear'),'head_ownership_logits':project_semantics(joint,'ownership'),'affine':affine})
        self.assertTrue(torch.equal(result['headwear_logits'],project_semantics(result['head_semantics_logits'],'headwear')))
        self.assertTrue(torch.all(result['headwear_logits'].softmax(1)[:,6]+result['head_ownership_logits'].softmax(1)[:,1]<=1+1e-6))

    def test_phone_expert_survives_affine_projection(self):
        from SkingToolkit.dense_uv_parser.utils import canonicalize_parser_outputs
        joint=torch.randn(2,14,16,16);old=torch.randn(2,5,16,16)
        result=canonicalize_parser_outputs({'head_semantics_logits':joint,'head_color_ownership_logits':old,'joint_phone_expert_accepted':torch.tensor([True,False]),'affine':torch.zeros(2,3)})
        self.assertTrue(torch.equal(result['head_ownership_logits'][0],result['head_color_ownership_logits'][0]))
        self.assertTrue(torch.equal(result['head_ownership_logits'][1],project_semantics(result['head_semantics_logits'],'ownership')[1]))

    def test_authored_features_and_layers(self):
        top=build_simple_uv_topology();source=torch.rand(4,64,64);source[3]=1
        feature_counts=[0,0,0];deep_caps=0
        for seed in range(32):
            d=make_joint_skin(source,seed,kind='crown',beard_layer=seed%2)
            self.assertTrue(torch.equal(d['uv'][:,16:],source[:,16:]))
            self.assertTrue(torch.all(d['labels'][d['nose']]==1))
            self.assertTrue(torch.all(d['labels'][d['beard']]==(5 if seed%2 else 3)))
            self.assertTrue(torch.all(top.layer[d['beard']]==seed%2))
            self.assertTrue(torch.all(d['labels'][d['caps']]==13))
            self.assertFalse(d['caps'][3:5,43:45].any())
            deep_caps+=int(d['caps'][1:7,41:47].sum())
            for i,key in enumerate(('nose','beard','caps')):feature_counts[i]+=int(d[key].sum())
        self.assertTrue(all(n>0 for n in feature_counts));self.assertGreater(deep_caps,0)

    def test_authored_mixed_beard_shares_coordinates_and_texture(self):
        for seed in range(12):
            d=make_joint_skin(torch.ones(4,64,64),seed,kind='bare',beard_layer=2,partial_hair=True)
            outer=d['labels'][:16,32:]==5;inner=d['labels'][:16,:32]==3
            self.assertTrue(outer.any());self.assertTrue((inner&~outer).any())
            self.assertTrue(torch.all(inner[outer]))
            self.assertTrue(torch.equal(d['uv'][:3,:16,:32][:,outer],d['uv'][:3,:16,32:][:,outer]))
            self.assertTrue((d['labels']==4).any())
            self.assertLessEqual(int((d['labels'][:8,40:48]==4).sum()),8)
            self.assertTrue(torch.equal(d['labels'][8:16,7]==3,d['labels'][8:16,8]==3))
            self.assertTrue(torch.equal(d['labels'][8:16,16]==3,d['labels'][8:16,15]==3))

    def test_mixed_beard_alignment_rejects_visible_mismatch_and_fills_only_hidden(self):
        from SkingToolkit.dense_uv_parser.head_semantics import reconcile_beard_alignment,complete_aligned_beard_material
        # Three overlapping cells: visible skin (wrong), visible beard (right),
        # unobserved base (unknown). A separate inner beard establishes overlap.
        idx=torch.tensor([14*64+12]*16+[15*64+12]*16+[15*64+44]*16+[14*64+13]*16+[14*64+45]*16+[15*64+46]*16)
        classes=torch.tensor([3]*16+[1]*16+[5]*16+[3]*16+[5]*16+[5]*16)
        logits=torch.nn.functional.one_hot(classes,14).T[None,:,None].float()*40-20
        r={'color_foreground':torch.ones(1,1,len(idx),dtype=torch.bool),'part':torch.zeros(1,1,len(idx),dtype=torch.long),'flat_uv':idx[None,None]}
        c=torch.zeros(1,10,64,64);c[:,8,14:16,44:47]=1;c[:,4,15,12]=1;c[:,4,14,13]=1
        d={'routing':r,'outputs':{'head_semantics_logits':logits}}
        fixed=reconcile_beard_alignment(c,d,['front'])
        self.assertEqual(float(fixed[0,8,15,44]),0)
        self.assertEqual(float(fixed[0,8,14,45]),1)
        self.assertEqual(float(fixed[0,8,15,46]),1)
        uv=torch.rand(1,4,64,64);uv[:,3]=1;before=uv.clone()
        filled=complete_aligned_beard_material(uv,d)
        self.assertTrue(torch.equal(filled[0,:3,15,14],before[0,:3,15,46]))
        self.assertTrue(torch.equal(filled[0,:3,14,13],before[0,:3,14,13]))
        self.assertTrue(torch.equal(filled[:,:,16:],before[:,:,16:]))

    def test_head_context_never_mixes_players_and_reads_back_view(self):
        torch.manual_seed(8);head=HeadSemanticsHead(semantic_dim=16).eval()
        crop=torch.rand(4,4,32,32);semantic=torch.rand(4,16,2,2)
        with torch.no_grad():
            original=head(crop,semantic)[0]
            altered_crop=crop.clone();altered_crop[1]=1-crop[1]
            altered=head(altered_crop,semantic)[0]
        self.assertFalse(torch.equal(original[0],altered[0]))
        self.assertTrue(torch.equal(original[2:],altered[2:]))
        with self.assertRaises(ValueError):head(crop[:3],semantic[:3])


@unittest.skipUnless(torch.cuda.is_available(),'CUDA renderer integration')
class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from SkingToolkit.renderer import DifferentiableRenderer
        from SkingToolkit.dense_uv_parser.utils import build_static_surface_routing
        torch.set_num_threads(4)
        cls.renderer=DifferentiableRenderer('/home/ds/llms/github/differentiable_minecraft_renderer/mappings_256x512').cuda()
        cls.views=['front_left','back_left']
        cls.static=[build_static_surface_routing(cls.renderer,v,torch.device('cuda')) for v in cls.views]

    def test_two_views_required_to_remove_hair_corner(self):
        h,w=self.static[0]['masks'].shape[-2:];fg=torch.ones(2,h,w,device='cuda',dtype=torch.bool)
        def route():
            keys=('flat_uv','part','face','texel_center_score')
            r={k:torch.stack([s[k][1] for s in self.static]) for k in keys}
            r.update(layer=torch.ones_like(fg,dtype=torch.long),foreground=fg.clone(),accessory_supported=fg.clone(),ownership_inner_supported=~fg,headwear_supported=~fg,headwear_layer=torch.full_like(fg,-1,dtype=torch.long),headwear_family=torch.zeros_like(fg,dtype=torch.long))
            return r
        logits=torch.full((2,14,h,w),-20.,device='cuda');logits[:,4]=20
        logits[0,4]=-20;logits[0,2]=20
        r=route();apply_joint_head_routing(r,{'head_semantics_logits':logits},fg,self.renderer,self.views)
        self.assertFalse(r['joint_inner_hair_veto'].any())
        logits[1,4]=-20;logits[1,2]=20
        r=route();apply_joint_head_routing(r,{'head_semantics_logits':logits},fg,self.renderer,self.views)
        self.assertTrue(r['joint_inner_hair_veto'].any())
        for vi,s in enumerate(self.static):
            changed=r['joint_inner_hair_veto'][vi]
            self.assertTrue(torch.all(s['face'][1][changed]==4))
            self.assertTrue(torch.all(r['layer'][vi][changed]==0))

    def test_mixed_beard_routing_preserves_both_layers_and_mouth(self):
        from SkingToolkit.dense_uv_parser.head_semantics import apply_beard_component_routing
        h,w=self.static[0]['masks'].shape[-2:];fg=torch.ones(2,h,w,device='cuda',dtype=torch.bool)
        logits=torch.full((2,14,h,w),-20.,device='cuda');logits[:,0]=20
        for vi,s in enumerate(self.static):
            for layer,cls,row in ((0,3,14),(1,5,15)):
                mask=s['masks'][layer]&(s['part'][layer]==0)&(s['face'][layer]==0)&(s['flat_uv'][layer]//64==row)
                logits[vi,:,mask]=-20;logits[vi,cls][mask]=20
        hole=self.static[0]['masks'][1]&(self.static[0]['flat_uv'][1]==15*64+44)
        logits[0,:,hole]=-20;logits[0,1][hole]=20
        routing={k:torch.stack([s[k][0] for s in self.static]) for k in ['flat_uv','part','face','texel_center_score']}
        routing.update(layer=torch.zeros_like(fg,dtype=torch.long),foreground=fg.clone(),accessory_supported=~fg,ownership_inner_supported=~fg,headwear_supported=~fg,headwear_layer=torch.full_like(fg,-1,dtype=torch.long),headwear_family=torch.zeros_like(fg,dtype=torch.long))
        apply_beard_component_routing(routing,{'head_semantics_logits':logits},fg,self.renderer,self.views)
        self.assertTrue(routing['beard_inner_supported'].any());self.assertTrue(routing['beard_outer_supported'].any())
        self.assertFalse(routing['beard_component_supported'][0][hole].any())
        self.assertTrue(torch.all(routing['layer'][0][hole]==0))
        routing['part'].fill_(1);original_layer=routing['layer'].clone()
        apply_beard_component_routing(routing,{'head_semantics_logits':logits},fg,self.renderer,self.views)
        self.assertFalse(routing['beard_component_supported'].any())
        self.assertTrue(torch.equal(original_layer,routing['layer']))

    def test_all_faces_remove_observed_strays_keep_mixed_beard_and_partial_hair(self):
        from SkingToolkit.dense_uv_parser.crown_geometry import prune_crown_top
        top=build_simple_uv_topology();outer=(top.valid&(top.part==0)&(top.layer==1)).cuda()
        d=make_joint_skin(torch.ones(4,64,64),6,kind='bare',beard_layer=2,partial_hair=True)
        uv=d['uv'][None].cuda();labels=d['labels'][None].cuda()
        semantic=torch.zeros_like(uv);semantic[:,0]=torch.isin(labels,labels.new_tensor([4,5,6,7,10,11,12,13])).float();semantic[:,3]=uv[:,3]
        rendered=torch.cat([self.renderer.forward_view(semantic,v) for v in self.views])
        # Actual rendered truth includes aligned inner/outer beard and outer fringe.
        same,_=prune_crown_top(uv,rendered[:,0]-rendered[:,1],rendered[:,3],self.renderer,self.views,removable_uv=outer)
        self.assertTrue(torch.equal(same,uv))
        for x,y in ((49,9),(50,14),(49,15)):
            corrupt=uv.clone();corrupt[:,3,y,x]=1
            fixed,_=prune_crown_top(corrupt,rendered[:,0]-rendered[:,1],rendered[:,3],self.renderer,self.views,removable_uv=outer)
            self.assertTrue(torch.equal(fixed[:,3],uv[:,3]),(x,y))
            self.assertTrue(torch.equal(fixed[:,:3],corrupt[:,:3]))

    def test_head_surface_routes_secondary_front_beard_without_touching_body(self):
        from SkingToolkit.dense_uv_parser.head_semantics import apply_head_surface_routing
        from SkingToolkit.dense_uv_parser.utils import build_dense_parser_batch
        source=make_joint_skin(torch.ones(4,64,64),6,kind='bare',beard_layer=2,partial_hair=True)
        uv=source['uv'][None].cuda();truth=source['labels'].cuda()
        uv[:,:,:16,32:40]=0;uv[:,:,:16,48:56]=0
        images=[];joint=[];surfaces=[];expected=[]
        for view in self.views:
            image,t=build_dense_parser_batch(uv,self.renderer,view)
            head=(t['foreground'][:,0]>.5)&(t['part']==0)
            index=(t['uv']*63).round().long().clamp(0,63);flat=index[:,1]*64+index[:,0]
            labels=truth.flatten()[flat].masked_fill(~head,0)
            joint.append(torch.nn.functional.one_hot(labels,14).permute(0,3,1,2).float()*40-20)
            surfaces.append(torch.nn.functional.one_hot((t['face']+1).masked_fill(~head,0),7).permute(0,3,1,2).float()*40-20)
            images.append(image);expected.append(t['face'])
        fg=torch.cat(images)[:,3]>.5
        routing={k:torch.stack([s[k][1] for s in self.static]) for k in ['flat_uv','part','face','texel_center_score']}
        routing.update(layer=torch.zeros_like(fg,dtype=torch.long),surface=torch.zeros_like(fg,dtype=torch.long),confidence=torch.zeros_like(fg,dtype=torch.float),accessory_supported=~fg,ownership_inner_supported=~fg)
        before=routing['part'].clone()
        original_uv=routing['flat_uv'].clone()
        outputs={'head_semantics_logits':torch.cat(joint),'head_surface_logits':torch.cat(surfaces)}
        apply_head_surface_routing(routing,{**outputs,'headwear_presence_logits':torch.full((2,2),20.,device='cuda')},fg,self.renderer,self.views)
        self.assertFalse(routing['head_surface_supported'].any())
        self.assertTrue(torch.equal(original_uv,routing['flat_uv']))
        apply_head_surface_routing(routing,outputs,fg,self.renderer,self.views)
        accepted=routing['head_surface_supported'];self.assertTrue(accepted.any())
        self.assertTrue(torch.all(routing['face'][accepted]==torch.cat(expected)[accepted]))
        self.assertTrue(torch.equal(routing['part'][before!=0],before[before!=0]))
        self.assertTrue((routing['surface'][accepted]>=2).any())

    def test_rendered_targets_use_visible_surface(self):
        d=make_joint_skin(torch.ones(4,64,64),6,kind='crown',beard_layer=0)
        b={k:v[None].cuda() for k,v in d.items()};b['mode']=torch.tensor([0],device='cuda')
        images,labels,modes,features=render_joint_batch(b,self.renderer,self.views)
        self.assertEqual(images.shape,(2,4,512,256));self.assertEqual(modes.tolist(),[0,0])
        self.assertTrue(torch.all(labels[features[:,0]]==1));self.assertTrue(torch.all(labels[features[:,2]]==13))
        self.assertFalse(features[:,2][labels==2].any())

    def test_rendered_hair_solver_removes_false_plate_and_preserves_outer_hair(self):
        from SkingToolkit.dense_uv_parser.crown_geometry import prune_crown_top
        found=set()
        for seed in range(30):
            d=make_joint_skin(torch.ones(4,64,64),seed,kind='bare')
            outer=bool((d['labels'][:8,40:48]==4).any())
            if outer in found:continue
            found.add(outer);uv=d['uv'][None].cuda();labels=d['labels'][None].cuda()
            semantic=torch.zeros_like(uv);semantic[:,0]=torch.isin(labels,labels.new_tensor([4,5,6,7,10,11,12,13])).float();semantic[:,3]=uv[:,3]
            rendered=torch.cat([self.renderer.forward_view(semantic,v) for v in self.views])
            corrupt=uv.clone();corrupt[:,3,2:6,42:46]=1
            result,_=prune_crown_top(corrupt,rendered[:,0]-rendered[:,1],rendered[:,3],self.renderer,self.views)
            self.assertTrue(torch.equal(result[:,3,:8,40:48],uv[:,3,:8,40:48]))
            if len(found)==2:break
        self.assertEqual(found,{False,True})

    def test_crown_solver_preserves_authored_inward_caps(self):
        from SkingToolkit.dense_uv_parser.crown_geometry import prune_crown_top
        for seed in (1,4,8):
            d=make_joint_skin(torch.ones(4,64,64),seed,kind='crown');uv=d['uv'][None].cuda();labels=d['labels'][None].cuda()
            semantic=torch.zeros_like(uv);semantic[:,0]=(labels==13).float();semantic[:,3]=uv[:,3]
            rendered=torch.cat([self.renderer.forward_view(semantic,v) for v in self.views])
            target=rendered[:,0]-rendered[:,1];fg=rendered[:,3]
            result,_=prune_crown_top(uv,target,fg,self.renderer,self.views,crown_uv=labels==13,protected_top=labels==4)
            self.assertTrue(torch.equal(result[:,3][labels==13],uv[:,3][labels==13]))
            self.assertTrue(torch.equal(result[:,3][labels==4],uv[:,3][labels==4]))


if __name__=='__main__':unittest.main()
