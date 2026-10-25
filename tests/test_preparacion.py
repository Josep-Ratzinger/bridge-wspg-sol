import unittest
import numpy as np
from M1.SPG.preparacion import muestrear_grupos


class PreparacionTest(unittest.TestCase):
    def test_correspondencia_normalizacion_y_grupos_pequenos(self):
        xyz = np.random.default_rng(3).normal(size=(171, 3))
        geo = np.random.default_rng(4).random((171, 5))
        labels = np.r_[np.zeros(150, dtype=int), np.ones(20, dtype=int), [2]]
        d = muestrear_grupos(xyz, geo, labels)
        self.assertEqual(d["puntos"].shape, (3, 128, 8))
        np.testing.assert_array_equal(d["embedding_habilitado"], [True, False, False])
        for i in range(3):
            idx = d["indices_reducidos"][i]
            self.assertTrue((labels[idx] == i).all())
            np.testing.assert_allclose(d["puntos"][i, :, 3:], geo[idx], rtol=1e-6)
            np.testing.assert_allclose(d["puntos"][i, :, :3]*d["escalas"][i]+d["centros"][i], xyz[idx], atol=1e-6)
        self.assertEqual(len(np.unique(d["indices_reducidos"][0])), 128)
        self.assertTrue(np.isfinite(d["puntos"]).all())
        np.testing.assert_array_equal(d["puntos"], muestrear_grupos(xyz, geo, labels)["puntos"])

    def test_rechaza_grupos_ausentes_y_atributos_incorrectos(self):
        xyz = np.zeros((3, 3)); geo = np.zeros((3, 5))
        with self.assertRaises(ValueError):
            muestrear_grupos(xyz, geo, np.array([0, 0, 2]))
        with self.assertRaises(ValueError):
            muestrear_grupos(xyz, geo[:, :4], np.zeros(3, dtype=int))
