import unittest
import numpy as np
import torch
from M2.taxonomia import semanticbridge,uav
from M2.estructura import calcular,costes
from M2.SOL.estructural import preparar,perdida
from M2.WSPG.relacional import WSPGRelacional
from M2.evaluacion import no_regresion
from M2.datos import CLASSES
from M2.sinteticos import generar


class Relaciones(unittest.TestCase):
    def test_superestructura_no_se_convierte_en_viga_exacta(self):
        self.assertEqual(semanticbridge([4]).tolist(),[6])
        self.assertEqual(semanticbridge([5,8]).tolist(),[4,1])
        self.assertEqual(semanticbridge([1]).tolist(),[80])
        self.assertEqual(uav([1,2,3,4]).tolist(),[6,1,2,8])

    def test_estructura_no_necesita_etiquetas_y_conserva_escala(self):
        xyz,lab,_=generar(82000);idx=np.arange(0,len(xyz),8);xyz=xyz[idx]
        group=np.arange(len(xyz))//30;a,m=calcular(xyz,group);b,_=calcular(xyz*2+[50,20,7],group)
        self.assertEqual(a.shape[1],20);self.assertTrue(np.isfinite(a).all());self.assertFalse(m['clases_leidas'])
        np.testing.assert_allclose(a,b,atol=.025)

    def test_viga_y_estribo_penalizados_en_apoyo_vertical_interior(self):
        f=np.zeros((1,20),dtype='f4');f[0,[0,1,2,3,7,9,12,13,14,18,19]]=[-.5,-.95,-.1,.85,.45,1,.8,.9,1,1,1]
        c=costes(f)[0];self.assertGreater(c[1],c[0]);self.assertGreater(c[7],c[0])

    def test_cara_vertical_no_obliga_a_ser_columna(self):
        f=np.zeros((1,20),dtype='f4');f[0,[0,2,3,7,9,14,18,19]]=[-.12,-.05,.15,.4,1,1,1,1]
        # A girder face has no tall vertical footprint evidence.
        self.assertLess(costes(f)[0,1],.2)

    def test_SOL_diferenciable_y_continuidad_lateral(self):
        f=np.zeros((4,20),dtype='f4');f[:,0]=.12;f[:,2]=.2;f[:,6]=1;f[:,8]=1;f[:,10]=1;f[:,18:]=1
        edges=np.array([[0,1,2],[1,2,3]]);prepared=preparar(f,edges,'cpu')
        logits=torch.randn(4,10,requires_grad=True);loss=perdida(logits,prepared);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(prepared[3].sum()),0)

    def test_residual_inicial_preserva_backbone_y_no_lo_entrena(self):
        torch.manual_seed(4);model=WSPGRelacional().eval();p=torch.randn(4,32,8);cent=torch.randn(4,3)
        enabled=torch.ones(4);edges=torch.tensor([[0,1,2],[1,2,3]]);edge=torch.randn(3,13);nodes=torch.randn(4,20)
        with torch.no_grad():
            a=model.base(p,cent,enabled,edges,edge);b=model(p,cent,enabled,edges,edge,nodes)
        torch.testing.assert_close(a,b);self.assertFalse(any(x.requires_grad for x in model.base.parameters()))

    def test_rechaza_regresion_aunque_aumente_la_media(self):
        ref=dict(IoU={k:.3 for k in CLASSES},barandas_tramos_cobertura50=.6)
        candidate=dict(IoU={k:.8 for k in CLASSES},barandas_tramos_cobertura50=.9)
        candidate['IoU'][CLASSES[1]]=.29
        self.assertFalse(no_regresion(candidate,ref)[0]);candidate['IoU'][CLASSES[1]]=.31
        self.assertTrue(no_regresion(candidate,ref)[0]);candidate['barandas_tramos_cobertura50']=.5
        self.assertFalse(no_regresion(candidate,ref)[0])
