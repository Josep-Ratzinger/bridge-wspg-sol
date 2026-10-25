import tempfile
import unittest
from pathlib import Path
import numpy as np
from M1.AdaPoinTr.src.completado_automatico import choose_shared_section,merge_surface,verify_output,merge_density_peaks
from M1.AdaPoinTr.src.exportacion import write_bridge
from M1.AdaPoinTr.src.geometrias_contextuales import pair,recipe,parts_for
from M1.AdaPoinTr.src.perfiles_viga import inside,sample_union


class AutomaticCompletionTest(unittest.TestCase):
    def test_two_ridges_of_one_column_are_not_two_supports(self):
        c={'density_hist_s':np.arange(9.),'density_hist':np.array([0,10,6,11,0,0,10,1,0.]),'density_peaks_s':np.array([1.,3.,6.])}
        merged=merge_density_peaks(c);self.assertEqual(len(merged),2);self.assertAlmostEqual(merged[-1],6.)

    def test_rectangular_evidence_overrides_round_hypothesis(self):
        def section(status,c,r):return {'status':status,'models':{'cilindrica':{'validation_score':c},'rectangular':{'validation_score':r}}}
        self.assertEqual(choose_shared_section([section('rectangular',.1,.01),section('incierta',.02,.02)])[0],'rectangular')
        self.assertEqual(choose_shared_section([section('incierta',.021,.02)])[0],'cilindrica')
        self.assertEqual(choose_shared_section([section('incierta',.1,.02)])[0],'rectangular')

    def test_reference_is_not_used_to_normalize_and_masks_keep_hidden_points_out(self):
        c=pair('train',4,2);self.assertTrue(c['visible_mask'][c['reference_source_indices']].all())
        recovered=c['partial']*c['scale']+c['center']
        self.assertTrue(np.allclose((recovered.min(0)+recovered.max(0))/2,c['center'],atol=2e-6))
        self.assertEqual(c['partial'].shape,(4096,3));self.assertEqual(c['complete'].shape,(16384,3))
        self.assertNotEqual(recipe('train',4)['base_id'],recipe('test',4)['base_id'])

    def test_compound_profile_preserves_empty_gaps_and_contacts(self):
        for index in (0,1,2):
            p=recipe('test',index);parts=parts_for(p);points,tags=sample_union(parts,3000,10)
            self.assertTrue(np.isfinite(points).all());below=points[:,2]<p['profile']['levels_z'][0]-1e-6
            self.assertFalse(((tags==1)&below).any())
            for i,col in enumerate(p['columns']):
                contact=np.array([[col['s'],col['d'],col['top']]])
                # Borde de la columna entra en el cabezal; centro puede quedar bajo talud.
                self.assertLessEqual(col['top'],p['profile']['new_top_z'])
                self.assertTrue(inside(parts[i+1],contact).all())

    def test_neural_outlier_is_rejected_and_sources_survive_merge(self):
        obs=np.array([[0.,0,0]]);surface=np.array([[1.,0,0],[2.,0,0],[3.,0,0]])
        points,tags,sources,accepted=merge_surface(obs,surface,np.array([2,2,1]),np.array([[1.01,0,0],[200.,0,0]]))
        self.assertEqual(accepted,1);self.assertEqual(len(points),3);self.assertEqual(int((sources==1).sum()),1)
        self.assertLess(points.max(),4.)

    def test_originals_ids_and_colors_are_exactly_preserved(self):
        x=np.array([[1,2,3],[-1,-2,-3]],dtype='f4');rgb=np.array([[1,2,3],[250,10,30]],dtype='u1');ids=np.array([17,91],dtype='u4')
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'out.ply';write_bridge(out,x,rgb,ids,np.array([[.2,.3,.4]],dtype='f4'),np.array([2]),np.array([2]),np.array([0]))
            verify_output(out,x,rgb,ids,1)

    def test_incomplete_header_fails_without_waiting_forever(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'incomplete.ply';out.write_bytes(b'ply\nformat binary_little_endian 1.0\n')
            with self.assertRaises(ValueError):verify_output(out,np.empty((0,3)),np.empty((0,3)),np.array([]),0)


if __name__=='__main__':unittest.main()
