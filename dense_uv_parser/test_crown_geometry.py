"""Occlusion regression: preserve authored caps and remove unsupported top plates."""
import unittest
import torch

from SkingToolkit.dense_uv_parser.crown_geometry import crop_renderer, prune_crown_top
from SkingToolkit.dense_uv_parser.headwear_data import make_headwear_skin
from SkingToolkit.dense_uv_parser.uv_topology import build_simple_uv_topology


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA renderer integration')
class CrownGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from SkingToolkit.renderer import DifferentiableRenderer
        torch.set_num_threads(4)
        cls.renderer = DifferentiableRenderer('/home/ds/llms/github/differentiable_minecraft_renderer/mappings_256x512').cuda()
        cls.views = ['front_left', 'back_left']
        topology = build_simple_uv_topology()
        cls.outer = (topology.valid & (topology.part == 0) & (topology.layer == 1)).cuda()
        cls.top = cls.outer & (topology.face.cuda() == 4)

    def authored(self, start=0, caps=None):
        for seed in range(start, start + 1000):
            uv, labels = make_headwear_skin(torch.ones(4,64,64), seed)
            if not (labels == 6).any():
                continue
            has_caps = bool((uv[3].cuda()[self.top] > .5).any())
            if caps is None or has_caps == caps:
                return uv[None].cuda(), labels[None].cuda()
        self.fail('No authored crown found')

    def target(self, uv, labels):
        semantic = torch.zeros_like(uv)
        semantic[:,0] = (labels == 6).float()
        semantic[:,3] = uv[:,3]
        with torch.no_grad():
            render = torch.stack([self.renderer.forward_view(semantic, v) for v in self.views],1).flatten(0,1)
        return render[:,0] - render[:,1], render[:,3]

    def test_cropped_renderer_is_exact_and_does_not_mutate_original(self):
        uv, _ = self.authored()
        cropped, (y,x) = crop_renderer(self.renderer, self.views, 512,256)
        for view in self.views:
            full = self.renderer.forward_view(uv,view)
            self.assertTrue(torch.equal(full[:,:,y,x],cropped.forward_view(uv,view)))
            self.assertEqual(self.renderer.forward_view(uv,view).shape[-2:],(512,256))

    def test_real_caps_are_preserved_and_false_interior_plate_removed(self):
        for start in (0, 1000, 2000, 3000):
            truth, labels = self.authored(start, caps=True)
            p, fg = self.target(truth, labels)
            candidate = truth.clone()
            # A central horizontal sheet has no authored support. Preserve all
            # true rim caps, including those visible only through another face.
            candidate[:,3,2:6,42:46] = 1
            result, records = prune_crown_top(candidate,p,fg,self.renderer,self.views,crown_uv=labels==6)
            self.assertTrue(torch.equal(result[:,3][labels==6],truth[:,3][labels==6]))
            self.assertFalse(result[:,3,2:6,42:46].any())
            self.assertTrue(torch.equal(candidate[:,:3],result[:,:3]))
            outside = ~self.top
            self.assertTrue(torch.equal(candidate[:,:,outside],result[:,:,outside]))
            self.assertLess(records[0]['after_loss'],records[0]['before_loss'])

    def test_empty_top_is_recovered_without_colour_or_fixed_rim_prior(self):
        truth, labels = self.authored(4000,caps=False)
        p, fg = self.target(truth, labels)
        candidate = truth.clone()
        candidate[:,3,self.top] = 1
        candidate[:,:3] = .37
        result, _ = prune_crown_top(candidate,p,fg,self.renderer,self.views,crown_uv=labels==6)
        self.assertFalse(result[:,3,self.top].any())
        self.assertTrue(torch.equal(candidate[:,:3],result[:,:3]))

    def test_players_are_independent_and_correct_geometry_stays_unchanged(self):
        first, a = self.authored(5000,caps=True)
        second, b = self.authored(6000,caps=False)
        truth = torch.cat([first,second]);labels=torch.cat([a,b])
        p, fg = self.target(truth,labels)
        result, records = prune_crown_top(truth,p,fg,self.renderer,self.views,crown_uv=labels==6)
        self.assertTrue(torch.equal(truth,result))
        self.assertEqual([len(r['removed']) for r in records],[0,0])

    def test_incomplete_views_are_rejected(self):
        uv, labels = self.authored()
        p, fg = self.target(uv,labels)
        with self.assertRaises(ValueError):
            prune_crown_top(uv,p[:1],fg[:1],self.renderer,self.views)


if __name__ == '__main__':
    unittest.main()
