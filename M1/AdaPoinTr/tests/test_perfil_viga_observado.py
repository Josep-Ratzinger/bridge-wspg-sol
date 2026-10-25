import unittest
import numpy as np
from M1.AdaPoinTr.src.perfiles_viga import fit_cross_profile,profiled_beam,inside,sample_union


class ObservedBeamProfileTest(unittest.TestCase):
    def test_head_is_separated_from_higher_wide_girder(self):
        rng=np.random.default_rng(6);cloud=[];columns=[]
        for s in (-4.,4.):
            z=rng.uniform(0,2,10000);d=rng.uniform(-1,1,10000)*(.5+.25*z);cloud.append(np.column_stack([rng.uniform(s-.2,s+.2,10000),d,z]))
            z=rng.uniform(2.,2.5,12000);d=rng.uniform(-2,2,12000);cloud.append(np.column_stack([rng.uniform(s-.2,s+.2,12000),d,z]));columns.append({'s':s,'d':0.,'radius':.4})
        profile=fit_cross_profile(np.vstack(cloud),columns,0.,3.)
        self.assertAlmostEqual(profile['new_top_z'],2.,delta=.11)
        self.assertGreater(profile['depths'][-1],profile['depths'][0])
        part=profiled_beam([[-4,0],[4,0],[5,3],[-5,3]],profile)
        self.assertFalse(inside(part,np.array([[0,0,2.8]]))[0])
        points,tags=sample_union([part],4000)
        self.assertTrue((points[:,2]<=profile['new_top_z']+1e-7).all());self.assertTrue((tags==1).all());self.assertTrue(inside(part,points).all())


if __name__=='__main__':unittest.main()
