"""UV-level mirror checks, independent from semantic pixel accuracy."""
import unittest
import numpy as np
from SkingToolkit.dense_uv_parser.uv_reference_repair import repair_uv,symmetry_report,metadata

class RepairTests(unittest.TestCase):
    def fixture(self):
        uv=np.zeros((64,64,4),np.uint8);uv[:,:,3]=255;uv[:,:,:3]=120
        reference=uv.copy();reference[8,40]=[10,20,30,255]
        hair=np.zeros((64,64),bool);hair[8,40]=True
        scope=np.zeros_like(hair);scope[15,[6,7,16,17,38,39,48,49]]=True
        support=scope.copy();sources=scope.copy()
        uv[15,38]=[0,0,0,0];uv[15,48]=[0,0,0,0]
        uv[15,49]=[22,30,40,255];uv[15,39]=[30,40,50,255]
        return uv,reference,hair,scope,support,sources

    def test_physical_side_face_mirror_is_not_whole_png_flip(self):
        _,_,m=metadata()
        self.assertEqual(int(m[15*64+39]),15*64+48)
        self.assertEqual(int(m[15*64+38]),15*64+49)
        uv,_,_,scope,_,_=self.fixture()
        self.assertEqual(symmetry_report(uv,scope)['outer']['alpha_mismatched_texels'],4)

    def test_mixed_beard_symmetry_alignment_and_reference_are_exact(self):
        args=self.fixture();uv,ref,hair,scope,_,_=args;original=uv.copy()
        result,report=repair_uv(*args)
        self.assertTrue(report['body_exact']);self.assertTrue(report['outside_regions_exact']);self.assertTrue(report['hair_reference_exact'])
        self.assertEqual(report['after']['outer']['alpha_mismatched_texels'],0)
        self.assertEqual(report['after']['inner']['rgb_max_difference'],0)
        self.assertEqual(report['after']['outer']['rgb_max_difference'],0)
        self.assertTrue(np.array_equal(result[15,6],result[15,38]))
        self.assertTrue(np.array_equal(result[15,7],result[15,39]))
        self.assertTrue(np.array_equal(uv,original))
        self.assertTrue(np.array_equal(result[hair],ref[hair]))

    def test_reject_unreviewed_partner_body_and_overlap(self):
        args=list(self.fixture());args[3]=args[3].copy();args[3][15,49]=False;args[4][15,49]=False
        with self.assertRaisesRegex(ValueError,'mirror partners'):repair_uv(*args)
        args=list(self.fixture());args[2][20,20]=True
        with self.assertRaisesRegex(ValueError,'head UV'):repair_uv(*args)
        args=list(self.fixture());args[2][15,39]=True
        with self.assertRaisesRegex(ValueError,'Overlapping'):repair_uv(*args)

    def test_do_not_invent_inner_beard_for_outer_without_support(self):
        args=list(self.fixture());args[4][15,[6,17]]=False
        with self.assertRaisesRegex(ValueError,'corresponding inner beard'):repair_uv(*args)

if __name__=='__main__':unittest.main()
