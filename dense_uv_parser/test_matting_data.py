import unittest
import torch
from SkingToolkit.dense_uv_parser.matting_data import warp_foreground,compose,make_composite

class MattingDataTests(unittest.TestCase):
    def test_transform_never_introduces_hidden_background_rgb_into_foreground(self):
        rgba=torch.zeros(4,12,8);rgba[:3]=torch.tensor([1.,0.,1.])[:,None,None]
        rgba[:3,2:10,2:6]=torch.tensor([.2,.6,.9])[:,None,None];rgba[3,2:10,2:6]=1
        theta=torch.tensor([[.93,0,.047],[0,1.05,-.035]])
        premult,a=warp_foreground(rgba,theta,32)
        expected=torch.tensor([.2,.6,.9])[:,None,None]*a
        self.assertTrue(torch.allclose(premult,expected,atol=1e-6))
        black=compose(premult,a,torch.zeros_like(premult));white=compose(premult,a,torch.ones_like(premult))
        self.assertTrue(torch.allclose(white-black,(1-a).expand_as(black),atol=1e-6))
    def test_backgrounds_are_varied_with_reproducible_soft_alpha(self):
        rgba=torch.ones(4,20,12);rgba[3,:2]=0;rgba[3,-2:]=0;rgba[:3]=.4
        kinds=set()
        for seed in range(50):
            x,a,k=make_composite(rgba,40,seed);kinds.add(k)
            self.assertTrue(bool(torch.isfinite(x).all()));self.assertTrue(bool((a>=0).all()&(a<=1).all()))
        self.assertEqual(kinds,set(range(7)))
        x,a,_=make_composite(rgba,40,3);y,b,_=make_composite(rgba,40,3)
        self.assertTrue(torch.equal(x,y)&torch.equal(a,b));self.assertTrue(bool(((a>0)&(a<1)).any()))
if __name__=='__main__':torch.set_num_threads(2);unittest.main()
