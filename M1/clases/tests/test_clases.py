import json,tempfile,unittest
from pathlib import Path
import numpy as np
from plyfile import PlyData,PlyElement
from M1.clases.anotacion import MODULE,asignar,escribir_ply
from M1.SPG.pureza import evaluar_pureza
from M1.SPG.particion import sha256


class AnnotationTests(unittest.TestCase):
    def test_point_regions_do_not_require_spg_and_exclude_estimates_from_reference(self):
        cfg=json.loads((MODULE/'configs/lurin.json').read_text(encoding='utf8'))
        x=np.array([[8.,-5.,-4.],[10.,-10.7,-.7],[0.,-5.,0.],[0.,-10.7,1.],[-45.,-5.,0.],[8.,-5.,-4.]])
        rgb=np.tile([150,130,30],(len(x),1));origin=np.array([0,0,0,0,0,1],dtype='u1')
        proposed,reference,confidence=asignar(x,rgb,origin,cfg)
        np.testing.assert_array_equal(proposed,[0,1,2,3,4,0]);np.testing.assert_array_equal(reference,[0,1,2,3,4,-1])
        self.assertTrue((confidence[proposed<0]==0).all())

    def test_user_corrections_separate_median_deck_strip_and_lower_surface(self):
        cfg=json.loads((MODULE/'configs/lurin.json').read_text(encoding='utf8'))
        x=np.array([[0.,.7,.14],[0.,8.6,.32],[8.,5.,-6.35],[8.,5.,-6.35],[-45.,-5.,3.]])
        rgb=np.tile([150,130,30],(len(x),1));origin=np.array([0,0,0,1,0],dtype='u1')
        proposed,reference,_=asignar(x,rgb,origin,cfg)
        np.testing.assert_array_equal(proposed[:3],[5,2,6]);np.testing.assert_array_equal(reference[:3],[5,2,6])
        self.assertNotEqual(proposed[3],6);self.assertEqual(reference[3],-1);self.assertEqual(proposed[4],-1)

    def test_purity_excludes_estimates_and_marks_assisted_reference_preliminary(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);x=np.array([[0,0,0],[1,0,0],[0,1,0]],dtype='f4')
            red=root/'red.npz';np.savez(red,xyz=x,origen=np.array([0,0,1],dtype='u1'))
            part=root/'part.npz';np.savez(part,celda_a_superpunto=np.array([0,0,1]),metadatos=np.array(json.dumps({'sha256_reduccion':sha256(red)})))
            dtype=[(k,'f4') for k in 'xyz']+[('punto_id','u4'),('clase','i4')];v=np.zeros(3,dtype=dtype)
            for i,k in enumerate('xyz'):v[k]=x[:,i]
            v['punto_id']=np.arange(3);v['clase']=[0,1,0]
            file=root/'ref.ply';PlyData([PlyElement.describe(v,'vertex')],text=False,comments=['tipo_etiquetas=anotacion_asistida_preliminar']).write(str(file))
            r=evaluar_pureza(red,part,file,root/'score')
            self.assertEqual(r['puntos_etiquetados'],2);self.assertEqual(r['pureza_ponderada_puntos'],.5);self.assertEqual(r['cobertura_puntos'],1.)
            self.assertEqual(r['puntos_estimados_excluidos'],1);self.assertTrue(r['evaluacion_preliminar']);self.assertFalse(r['validacion_humana_experta'])
            PlyData([PlyElement.describe(v,'vertex')],text=False,comments=['tipo_etiquetas=propuesta_automatica']).write(str(file))
            with self.assertRaises(ValueError):evaluar_pureza(red,part,file,root/'bad')


if __name__=='__main__':unittest.main()
