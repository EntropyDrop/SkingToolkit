import io
import unittest
import torch
from SkingToolkit.dense_uv_parser.final_head_uv import FinalHeadUVDecoder,final_uv_loss,evaluation_numerics


class FinalHeadTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7);torch.set_num_threads(4);evaluation_numerics()
        self.model=FinalHeadUVDecoder(width=32,layers=1)
        self.uv=torch.rand(1,4,64,64);self.uv[:,3]=1
        self.uv[:,:,0:16,32:]=0
        self.evidence=torch.rand(2,25,56,56)

    def test_initial_identity_and_body_isolation(self):
        result=self.model(self.uv,self.evidence)['uv']
        self.assertTrue(torch.equal(result,self.uv))
        with torch.no_grad():self.model.alpha.bias.fill_(20);self.model.color_gate.bias.fill_(20);self.model.color_delta.bias.fill_(.2)
        result=self.model(self.uv,self.evidence)['uv']
        self.assertTrue(torch.equal(result[:,:,16:],self.uv[:,:,16:]))
        invalid=torch.ones(4096,dtype=torch.bool);invalid[self.model.ids]=False
        self.assertTrue(torch.equal(result.flatten(2)[:,:,invalid],self.uv.flatten(2)[:,:,invalid]))

    def test_empty_outer_cells_receive_gradient(self):
        labels=torch.full((1,64,64),-100,dtype=torch.long)
        result=self.model(self.uv,self.evidence)
        loss,metrics=final_uv_loss(self.model,result,self.uv,self.uv,labels,torch.tensor([False]))
        self.assertEqual(int(metrics['empty_outer_cells']),384)
        loss.backward()
        self.assertGreater(float(self.model.alpha.bias.grad.abs().sum()),0)

    def test_false_added_and_missing_cells_penalized(self):
        labels=torch.full((1,64,64),-100,dtype=torch.long)
        _,good=final_uv_loss(self.model,self.model(self.uv,self.evidence),self.uv,self.uv,labels,torch.tensor([False]))
        wrong=self.uv.clone();wrong[:,:,8:16,40:48]=1
        _,bad=final_uv_loss(self.model,self.model(wrong,self.evidence),wrong,self.uv,labels,torch.tensor([False]))
        self.assertGreater(float(bad['occupancy']),float(good['occupancy']))
        _,missing=final_uv_loss(self.model,self.model(self.uv,self.evidence),self.uv,wrong,labels,torch.tensor([False]))
        self.assertGreater(float(missing['occupancy']),float(good['occupancy']))

    def test_serialized_and_train_eval_results_match(self):
        with torch.no_grad():
            self.model.alpha.weight.normal_(0,.03);self.model.color_delta.weight.normal_(0,.03)
        self.model.train();a=self.model(self.uv,self.evidence)['uv']
        self.model.eval();b=self.model(self.uv,self.evidence)['uv']
        self.assertTrue(torch.equal(a,b))
        buffer=io.BytesIO();torch.save(self.model.state_dict(),buffer);buffer.seek(0)
        restored=FinalHeadUVDecoder(width=32,layers=1).eval();restored.load_state_dict(torch.load(buffer,weights_only=True))
        self.assertTrue(torch.equal(b,restored(self.uv,self.evidence)['uv']))

    def test_mirror_topology_and_conditional_symmetry(self):
        ids=self.model.ids.tolist();a=ids.index(15*64+39);b=ids.index(15*64+48)
        self.assertEqual(int(self.model.mirror[a]),b)
        labels=torch.full((1,64,64),-100,dtype=torch.long);labels[0,15,[39,48]]=5
        wrong=self.uv.clone();wrong[0,3,15,39]=1
        result=self.model(wrong,self.evidence)
        _,on=final_uv_loss(self.model,result,wrong,self.uv,labels,torch.tensor([True]))
        _,off=final_uv_loss(self.model,result,wrong,self.uv,labels,torch.tensor([False]))
        self.assertGreater(float(on['symmetry']),0);self.assertEqual(float(off['symmetry']),0)

    def test_stable_background_median_preserves_legacy_values(self):
        from SkingToolkit.dense_uv_parser.utils import estimate_solid_background_color
        for device in (['cpu','cuda'] if torch.cuda.is_available() else ['cpu']):
            images=torch.rand(3,4,32,32,device=device)
            images[1,:3,:4,:4]=.5
            images[2,0,0,0]=float('nan')
            torch.use_deterministic_algorithms(False)
            old=estimate_solid_background_color(images)
            evaluation_numerics()
            new=estimate_solid_background_color(images)
            self.assertTrue(torch.allclose(old[0],new[0],rtol=0,atol=0,equal_nan=True))
            self.assertTrue(torch.equal(old[1],new[1]))


if __name__=='__main__':unittest.main()
