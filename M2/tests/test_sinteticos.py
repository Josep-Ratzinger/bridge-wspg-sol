import unittest
import numpy as np
from M2.sinteticos import generar

class SinteticosFinales(unittest.TestCase):
    def test_combinaciones_sin_dependencia_modular_y_barandas_abiertas(self):
        cases=[generar(82000+i) for i in range(16)]
        self.assertEqual(set(p[2]['trazado'] for p in cases),{'recto','arco','S'})
        self.assertGreater(len(set(p[2]['baranda'] for p in cases)),2)
        for xyz,lab,p in cases:
            self.assertTrue(np.isfinite(xyz).all());self.assertEqual(len(xyz),len(lab))
            self.assertTrue(set(np.unique(lab)).issubset(set(range(10))))
            self.assertFalse(p['validacion_estructural'])
        a=generar(82000);b=generar(82000);np.testing.assert_array_equal(a[0],b[0])
