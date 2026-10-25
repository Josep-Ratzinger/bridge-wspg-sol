import unittest,tempfile,zipfile
from pathlib import Path
import numpy as np
import torch
from M2.corpus import read_semantic
from M2.utilidades import partial_loss,allowed_matrix,centros_relativos,resumen

class CorpusFinal(unittest.TestCase):
    def test_lectura_zip_conserva_xyz_y_labels_sin_usar_rgb(self):
        n=5; xyz=np.arange(n*3,dtype='<f4').reshape(n,3); labels=np.array([8,4,5,6,1],dtype='u1')
        header=('ply\nformat binary_little_endian 1.0\nelement points 5\nproperty float x\nproperty float y\nproperty float z\n'
            'element color 5\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nelement label 5\nproperty uchar label\nend_header\n').encode('ascii')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'test.zip'
            with zipfile.ZipFile(path,'w') as z: z.writestr('bridge.ply',header+xyz.tobytes()+bytes(n*3)+labels.tobytes())
            with zipfile.ZipFile(path) as z: x,y,count=read_semantic(z,'bridge.ply',5,42)
            np.testing.assert_array_equal(x,xyz); np.testing.assert_array_equal(y,labels); self.assertEqual(count,n)

    def test_perdida_union_acepta_cualquiera_pero_no_otra(self):
        hist=torch.zeros(1,12); hist[0,11]=10
        allowed=allowed_matrix('cpu'); weight=torch.ones(10)
        losses=[]
        for predicted in [1,2,0]:
            x=torch.full((1,10),-8.); x[0,predicted]=8.; x.requires_grad_()
            loss=partial_loss(x,hist,weight,allowed); loss.backward()
            self.assertTrue(torch.isfinite(x.grad).all()); losses.append(float(loss.detach()))
        self.assertLess(losses[0],.001); self.assertLess(losses[1],.001)
        self.assertGreater(losses[2],10)

    def test_centros_no_dependen_de_coordenadas_geograficas(self):
        c=np.random.default_rng(1).normal(size=(100,3))
        np.testing.assert_allclose(centros_relativos(c),centros_relativos(c+1000),atol=1e-5)
