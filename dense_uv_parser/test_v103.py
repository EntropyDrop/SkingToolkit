import unittest
import numpy as np
import torch
from SkingToolkit.dense_uv_parser.v103_data import make_structured_skin
from SkingToolkit.dense_uv_parser.v103_losses import pool_uv, structured_uv_loss
from SkingToolkit.dense_uv_parser.uv_topology import build_simple_uv_topology


class StructuredUVTests(unittest.TestCase):
    def original(self):
        uv=torch.rand(4,64,64,generator=torch.Generator().manual_seed(7))
        uv[3]=1
        return uv

    def test_symmetric_beard_all_faces_and_mixed_layers(self):
        mirror=build_simple_uv_topology().mirrored_texel.reshape(-1)
        mixed_count=0
        for seed in range(20):
            item=make_structured_skin(self.original(),seed,symmetric=True,profile="flat")
            labels=item["labels"].flatten();uv=item["uv"].flatten(1)
            for cls in (3,5):
                mask=labels==cls
                self.assertTrue(torch.equal(mask,mask[mirror]))
                self.assertTrue(torch.equal(uv[:,mask],uv[:,mirror[mask]]))
            outer=torch.nonzero(labels==5).flatten()
            mixed=outer[labels[outer-32]==3]
            mixed_count+=len(mixed)
            self.assertTrue(torch.equal(uv[:,mixed],uv[:,mixed-32]))
        self.assertGreater(mixed_count,0)

    def test_fringe_has_no_forced_top_or_rear_shell(self):
        for seed in range(12):
            item=make_structured_skin(self.original(),seed,profile="fringe")
            self.assertFalse((item["labels"][:8,40:48]==4).any())
            self.assertFalse((item["labels"][8:16,56:64]==4).any())
            self.assertTrue((item["labels"][8:12,40:48]==4).any())

    def test_asymmetric_cases_remain_asymmetric(self):
        mirror=build_simple_uv_topology().mirrored_texel.reshape(-1)
        found=False
        for seed in range(20):
            item=make_structured_skin(self.original(),seed,symmetric=False)
            beard=(item["labels"]==5).flatten()
            found|=not torch.equal(beard,beard[mirror])
        self.assertTrue(found)

    def test_body_unmodified_and_hair_variants_differ(self):
        original=self.original()
        masks=[]
        for profile in ("flat","fringe","capped","shell"):
            item=make_structured_skin(original,41,profile=profile)
            self.assertTrue(torch.equal(original[:,16:],item["uv"][:,16:]))
            masks.append((item["labels"]==4).numpy())
        self.assertTrue(all(not np.array_equal(masks[i],masks[i+1]) for i in range(3)))

    def test_uv_pooling_balances_cells_and_preserves_gradient(self):
        p=torch.tensor([[[[.1,.3,.9]]]]*4,requires_grad=True)
        index=torch.tensor([[[1,1,2]]]*4);valid=torch.ones_like(index,dtype=torch.bool)
        pooled,counts=pool_uv(p,index,valid)
        self.assertAlmostEqual(float(pooled[0,0,1]),.2,places=5)
        self.assertAlmostEqual(float(pooled[0,0,2]),.9,places=5)
        self.assertEqual(int(counts[0,1]),8)
        pooled.sum().backward();self.assertTrue(torch.isfinite(p.grad).all())

    def test_wrong_uv_layer_is_penalized_and_symmetry_is_conditional(self):
        ids=torch.tensor([14*64+40,14*64+47,14*64+8,14*64+15])
        index=ids[None,None].expand(4,1,-1)
        valid=torch.ones_like(index,dtype=torch.bool)
        labels=torch.zeros(1,64,64,dtype=torch.long)
        labels.flatten()[ids]=torch.tensor([5,5,3,3])
        good=torch.full((4,14,1,4),-7.)
        good[:,5,0,:2]=7;good[:,3,0,2:]=7
        bad=good.clone();bad[:,5,0,1]=-7;bad[:,3,0,1]=7;bad.requires_grad_(True)
        args=(index,valid,labels,torch.tensor([0]))
        a,_=structured_uv_loss(good,*args,torch.tensor([True]))
        b,metrics=structured_uv_loss(bad,*args,torch.tensor([True]))
        self.assertGreater(float(b),float(a)+1)
        self.assertGreater(float(metrics["uv_mirror"]),0)
        self.assertGreater(float(metrics["mixed_cells"]),0)
        _,off=structured_uv_loss(bad,*args,torch.tensor([False]))
        self.assertEqual(float(off["uv_mirror"]),0)
        b.backward();self.assertGreater(float(bad.grad.abs().sum()),0)


if __name__=="__main__":
    torch.set_num_threads(4)
    unittest.main()
