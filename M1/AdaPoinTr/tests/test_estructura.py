import unittest
import numpy as np
from M1.AdaPoinTr.src.geometrias_sinteticas import prism,box,sample_surface
from M1.AdaPoinTr.src.ajuste_estructura import fit_structure


class GuidedFitTest(unittest.TestCase):
    def test_four_support_scene_is_not_forced_to_ten_and_gaps_stay_open(self):
        peaks=np.array([-3.5,-1.2,1.2,3.5]);parts=[prism([[-4.,4.],[4.,4.],[5.,5.],[-5.,5.]],1.2,1,[0,0,0],1)]
        parts.extend(box([.6,.8,4.],[s,0,2.],2+i) for i,s in enumerate(peaks))
        full,_,tags=sample_surface(parts,14000,np.random.default_rng(5))
        keep=(tags==1)&(abs(full[:,0])>3.8)
        for i,f in enumerate([1.,.25,.1,1.]):keep|=(tags==2+i)&(full[:,2]<=4*f)
        observed=full[keep];fitted,record=fit_structure(observed,peaks,5.2,7)
        self.assertEqual(record['support_count'],4)
        surface,_,_=sample_surface(fitted,10000,np.random.default_rng(8))
        self.assertTrue(np.isfinite(surface).all())
        below=surface[:,2]<record['beam_bottom_z']-.02;in_column=np.zeros(len(surface),bool)
        for c in record['columns']:in_column|=abs(surface[:,0]-c['s'])<=c['width_shared']/2+1e-6
        self.assertFalse((below&~in_column).any())


if __name__=='__main__':unittest.main()
