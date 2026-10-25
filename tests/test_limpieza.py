import json,tempfile,unittest
from pathlib import Path
import numpy as np
from plyfile import PlyData,PlyElement
from M1.limpieza_y_preparacion.limpieza import filtrar_ruido,normalizar,limpiar_preparar
from M1.limpieza_y_preparacion.reduccion import reducir_nube

class LimpiezaTest(unittest.TestCase):
    def test_aislados_retirados_y_estructura_delgada_preservada(self):
        # Superficie densa y baranda menos densa: no confundir densidad con ruido.
        xx,yy=np.meshgrid(np.arange(25)*.02,np.arange(25)*.02)
        plano=np.column_stack((xx.ravel(),yy.ravel(),np.zeros(xx.size)))
        linea=np.column_stack((np.arange(25)*.08,np.full(25,.8),np.full(25,.3)))
        aislados=np.array([[20,20,20],[-20,-20,-20]])
        x=np.vstack((plano,linea,aislados));antes=x.copy()
        keep,_,meta=filtrar_ruido(x,desviaciones=1)
        self.assertTrue(keep[:len(plano)+len(linea)].all());self.assertFalse(keep[-2:].any())
        np.testing.assert_array_equal(x,antes)
    def test_proteccion_lineal_activa(self):
        xx,yy=np.meshgrid(np.arange(40)*.015,np.arange(40)*.015)
        plano=np.column_stack((xx.ravel(),yy.ravel(),np.zeros(xx.size)))
        linea=np.column_stack((np.arange(30)*.08,np.full(30,2.),np.zeros(30)))
        keep,_,m=filtrar_ruido(np.vstack((plano,linea)),desviaciones=1)
        self.assertGreater(m['puntos_lineales_protegidos'],0)
        self.assertTrue(keep[-30:].all())
    def test_normalizacion_conserva_proporciones_y_es_reversible(self):
        x=np.array([[10.,-7.,2.],[15.,-3.,4.],[12.,-5.,3.]])
        n,c,s=normalizar(x)
        np.testing.assert_allclose(n*s+c,x,atol=1e-6)
        self.assertAlmostEqual(float(np.ptp(n,axis=0).max()),1.)
        self.assertAlmostEqual(np.linalg.norm(n[1]-n[0])*s,np.linalg.norm(x[1]-x[0]),places=5)
    def test_correspondencias_colores_y_fuentes_intactos(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);rng=np.random.default_rng(11);x=np.r_[rng.uniform(0,.5,(120,3)),[[10,10,10]]].astype('f4')
            v=np.zeros(len(x),dtype=[(n,'f4') for n in 'xyz']+[(n,'u1') for n in ('red','green','blue')])
            for j,n in enumerate('xyz'):v[n]=x[:,j]
            for j,n in enumerate(('red','green','blue')):v[n]=rng.integers(0,256,len(x))
            manual=p/'manual.ply';PlyData([PlyElement.describe(v,'vertex')],text=False).write(manual)
            reducir_nube(manual,p/'reduccion',.01);antes=manual.read_bytes()
            m=limpiar_preparar(manual,p/'reduccion/reduccion.npz',p/'final',vecinos=8,radio=.2)
            with np.load(p/'final/preparacion.npz') as d,np.load(p/'reduccion/reduccion.npz') as base:
                ids=d['indices_reduccion'];np.testing.assert_array_equal(d['xyz'],base['xyz'][ids]);np.testing.assert_array_equal(d['rgb'],base['rgb'][ids])
                np.testing.assert_array_equal(np.bincount(d['punto_a_celda'],minlength=len(ids)),d['conteos'])
                np.testing.assert_allclose(d['xyz_normalizado']*d['escala']+d['centro'],d['xyz'],atol=1e-5)
                total=m['puntos_reducidos']+m['retirados'];self.assertEqual(total,len(base['xyz']))
            self.assertEqual(manual.read_bytes(),antes)
            with self.assertRaises(ValueError):limpiar_preparar(manual,p/'reduccion/reduccion.npz',p/'final')
    def test_entrada_invalida(self):
        with self.assertRaises(ValueError):filtrar_ruido(np.full((30,3),np.nan))
        with self.assertRaises(ValueError):normalizar(np.zeros((3,3)))
if __name__=='__main__':unittest.main()
