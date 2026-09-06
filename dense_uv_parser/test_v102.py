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

    def test_rendered_targets_use_visible_surface(self):
        d=make_joint_skin(torch.ones(4,64,64),6,kind='crown',beard_layer=0)
        b={k:v[None].cuda() for k,v in d.items()};b['mode']=torch.tensor([0],device='cuda')
        images,labels,modes,features=render_joint_batch(b,self.renderer,self.views)
        self.assertEqual(images.shape,(2,4,512,256));self.assertEqual(modes.tolist(),[0,0])
        self.assertTrue(torch.all(labels[features[:,0]]==1));self.assertTrue(torch.all(labels[features[:,2]]==13))
        self.assertFalse(features[:,2][labels==2].any())

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
