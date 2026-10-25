import unittest
import numpy as np
from M1.AdaPoinTr.src.seleccion_formas import select_curve,section_polygon,augment_section,loft_beam_parts,beam_underside,observed_beam_forms
from M1.AdaPoinTr.src.secciones_columnas import fit_section
from M1.AdaPoinTr.src.perfiles_viga import inside,sample_union


class ShapeSelectionTest(unittest.TestCase):
    def test_beam_does_not_force_a_trapezoid(self):
        z=np.linspace(0,1,11)
        self.assertEqual(select_curve(z,np.ones(11)*2,.02)[1]['selected'],'rectangular')
        self.assertEqual(select_curve(z,1+z,.02)[1]['selected'],'trapezoidal')
        self.assertEqual(select_curve(z,1+z*z,.005)[1]['selected'],'curvo')
        selected=select_curve(z,1+z+2*np.maximum(z-.5,0),.005)[1]['selected']
        self.assertEqual(selected,'hombro_0.5')

    def test_columns_include_ellipse_and_octagon_with_full_coverage(self):
        rng=np.random.default_rng(18)
        for kind,params in [('eliptica',[0,0,.7,.35,.2]),('octagonal',[0,0,.55,.55,.16])]:
            poly=section_polygon(kind,params,192);n=6000;ids=rng.integers(len(poly),size=n);t=rng.random(n)
            xy=poly[ids]*(1-t[:,None])+poly[(ids+1)%len(poly)]*t[:,None]+rng.normal(0,.0008,(n,2))
            points=np.c_[xy,rng.uniform(0,3,n)];base=fit_section(points,[0,0],1.1,1.1,3)
            fit=augment_section(points,base,1.1,1.1,3)
            self.assertEqual(fit['status'],kind)

    def test_partial_face_does_not_confirm_complex_shape(self):
        rng=np.random.default_rng(7);t=rng.uniform(-.3,.3,1000);points=np.c_[.5*np.cos(t),.5*np.sin(t),rng.uniform(0,2,1000)]
        base=fit_section(points,[0,0],1.,1.,4);fit=augment_section(points,base,1.,1.,4)
        self.assertNotIn(fit['status'],('eliptica','octagonal'))

    def test_nonconvex_beam_keeps_waist_and_has_no_triangle_tip(self):
        profile={'levels_z':[0,.2,.3,.7,.8,1.],'depths':[2.,2.,.5,.5,2.,2.],
                 'left_s':[-3.]*6,'right_s':[3.]*6,'relative_centers_d':[0.]*6,
                 'row_center_slope':0.,'row_center_intercept':0.}
        parts=loft_beam_parts(profile);void=np.array([[0,.8,.5]])
        self.assertFalse(any(inside(p,void)[0] for p in parts))
        p,tags=sample_union(parts,3000,8);middle=p[(p[:,2]>.31)&(p[:,2]<.69)]
        self.assertTrue((abs(middle[:,1])<=.250001).all());self.assertTrue((tags==1).all())
        self.assertEqual(beam_underside(-2.,profile),0.)

    def test_endpoint_uses_observed_profile_instead_of_old_triangle(self):
        rng=np.random.default_rng(12);n=50000;z=rng.uniform(0,1,n);s=rng.uniform(-4,4,n)
        d=rng.uniform(-1,1,n)*(.6+.3*z);observed=np.c_[s,d,z]
        columns=[{'s':-3.5,'d':0,'radius':.35},{'s':3.5,'d':0,'radius':.35}]
        prior={'levels_z':np.linspace(0,1,7).tolist(),'depths':np.linspace(1.2,1.8,7).tolist(),'relative_centers_d':[0.]*7,
               'row_center_slope':0.,'row_center_intercept':0.,'new_top_z':1.,'bottom_z':0.}
        result=observed_beam_forms(observed,columns,prior,[[-3,0],[3,0],[8,1],[-8,1]])
        self.assertEqual(result['shape_selection']['left_end']['selected'],'rectangular')
        self.assertTrue(np.all(np.array(result['left_s'])>-4.1));self.assertTrue(np.all(np.array(result['right_s'])<4.1))


if __name__=='__main__':unittest.main()
