import io
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
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


class TrainingRecoveryTests(unittest.TestCase):
    def test_optimizer_and_random_stream_resume_exactly(self):
        from SkingToolkit.dense_uv_parser.final_uv_training_state import save_recovery,restore_recovery
        torch.manual_seed(5);random.seed(5);np.random.seed(5)
        device='cuda' if torch.cuda.is_available() else 'cpu'
        model=torch.nn.Linear(3,2).to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001)
        def step():
            opt.zero_grad();x=torch.rand(5,3,device=device)+random.random()+np.random.rand()
            model(x).square().sum().backward();opt.step()
        step()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'state.pt';save_recovery(path,model,opt,1,{'steps':10})
            step();expected={k:v.clone() for k,v in model.state_dict().items()}
            expected_moments=[v['exp_avg'].clone() for v in opt.state.values()]
            torch.rand(20,device=device);random.random();np.random.rand()
            self.assertEqual(restore_recovery(path,model,opt,{'steps':10}),1)
            step()
            for k,v in model.state_dict().items():self.assertTrue(torch.equal(v,expected[k]))
            for a,b in zip(expected_moments,opt.state.values()):self.assertTrue(torch.equal(a,b['exp_avg']))
            with self.assertRaises(ValueError):restore_recovery(path,model,opt,{'steps':11})

    def test_failed_write_retains_previous_recovery(self):
        from SkingToolkit.dense_uv_parser.final_uv_training_state import atomic_save
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'state.pt';atomic_save({'step':1},path);before=path.read_bytes()
            def interrupted(value,destination):
                Path(destination).write_bytes(b'incomplete');raise OSError('disk write interrupted')
            with patch('torch.save',side_effect=interrupted):
                with self.assertRaises(OSError):atomic_save({'step':2},path)
            self.assertEqual(path.read_bytes(),before)

    def test_roundtrip_reports_bounded_rgb_but_rejects_geometry_and_body(self):
        from SkingToolkit.dense_uv_parser.final_uv_training_state import compare_roundtrip
        uv=torch.zeros(1,4,64,64);uv[:,3]=1;uv[:,:3]=.5;evidence=torch.zeros(2,25,56,56)
        fresh=uv.clone();fresh[0,1,12,30]+=.003
        def compare(other,**kwargs):
            return compare_roundtrip(uv,other,uv,kwargs.get('base',other),evidence,kwargs.get('evidence',evidence))
        report=compare(fresh);self.assertTrue(report['passed']);self.assertFalse(report['rgba_exact']);self.assertEqual(report['png_rgb_max_delta'],1)
        alpha=fresh.clone();alpha[0,3,12,40]=0;self.assertFalse(compare(alpha)['passed'])
        body=fresh.clone();body[0,1,20,10]+=.00001;self.assertFalse(compare(body)['passed'])
        large=fresh.clone();large[0,1,12,30]+=.002;self.assertFalse(compare(large)['passed'])
        self.assertFalse(compare(fresh,evidence=evidence+.001)['passed'])

    def test_upstream_warning_keeps_failed_status_and_never_hides_geometry_errors(self):
        from SkingToolkit.dense_uv_parser.final_uv_training_state import compare_roundtrip,upstream_rgb_warning
        uv=torch.ones(1,4,64,64)*.5;uv[:,3]=1;fresh=uv.clone();fresh[0,1,12,30]+=.004
        evidence=torch.zeros(2,25,56,56)
        report=compare_roundtrip(uv,fresh,uv,fresh,evidence,evidence)
        report.update(serialized_decoder_exact=True,fresh_inputs_decoder_exact=True)
        self.assertFalse(report['passed']);self.assertTrue(upstream_rgb_warning(report));self.assertFalse(report['passed'])
        for key in ('alpha_exact','body_exact','evidence_exact','serialized_decoder_exact','fresh_inputs_decoder_exact'):
            self.assertFalse(upstream_rgb_warning({**report,key:False}))
        self.assertFalse(upstream_rgb_warning({**report,'base_rgb_max_delta':0}))
        self.assertFalse(upstream_rgb_warning({'passed':False,'reason':'non-finite roundtrip tensors'}))


if __name__=='__main__':unittest.main()
