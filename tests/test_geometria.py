"""Comprobaciones geométricas con formas conocidas y archivos temporales."""

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
from plyfile import PlyData

from M1.SPG.geometria import calcular_geometria, descriptores, procesar_geometria, elevacion_z_minmax


class GeometriaTest(unittest.TestCase):
    def test_lineas_y_plano(self):
        horizontal = np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]], float)
        vertical = horizontal[:, [1, 2, 0]]
        g, valido = descriptores(np.stack([horizontal, vertical]))
        np.testing.assert_allclose(g, [[1, 0, 0, 0], [1, 0, 0, 1]], atol=1e-7)
        self.assertTrue(valido.all())
        plano = np.array([[[-1, -1, 0], [1, -1, 0], [-1, 1, 0], [1, 1, 0]]], float)
        g, _ = descriptores(plano)
        np.testing.assert_allclose(g, [[0, 1, 0, 0]], atol=1e-7)

    def test_escala_traslacion_y_bloques(self):
        xyz = np.random.default_rng(42).normal(size=(120, 3))
        g, radio, validos = calcular_geometria(xyz, vecinos=12, bloque=13)
        otro, radio2, _ = calcular_geometria(xyz * 9.797362 + [10, -3, 5], vecinos=12, bloque=37)
        np.testing.assert_allclose(g, otro, atol=1e-6)
        np.testing.assert_allclose(radio2, radio * 9.797362, rtol=1e-6)
        self.assertTrue(validos.all())
        self.assertTrue(((g >= 0) & (g <= 1)).all())
        np.testing.assert_allclose(g[:, :3].sum(axis=1), 1, atol=1e-6)

    def test_propio_excluido_y_degenerados(self):
        xyz = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]], float)
        _, radios, _ = calcular_geometria(xyz, vecinos=3, bloque=2)
        np.testing.assert_allclose(radios, [4, 3, 2, 4])
        g, _, validos = calcular_geometria(np.zeros((8, 3)), vecinos=3)
        np.testing.assert_array_equal(g, 0)
        self.assertFalse(validos.any())
        with self.assertRaises(ValueError):
            calcular_geometria(xyz, vecinos=4)

    def test_elevacion_explicita_global_e_invariante(self):
        xyz = np.array([[0, 0, -2], [1, 0, 3], [2, 0, 8]], dtype=float)
        e, meta = elevacion_z_minmax(xyz)
        np.testing.assert_allclose(e, [0, .5, 1])
        otra, _ = elevacion_z_minmax(xyz * 7 + [2, 3, 100])
        np.testing.assert_array_equal(e, otra)
        plana, _ = elevacion_z_minmax(np.array([[0, 0, 3], [1, 0, 3]]))
        np.testing.assert_array_equal(plana, 0)
        self.assertIn("adaptación propia", meta["procedencia"])

    def test_archivos_y_entrada_intacta(self):
        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp)
            entrada = raiz / "reduccion.npz"
            xyz = np.random.default_rng(12).normal(size=(50, 3)).astype(np.float32)
            np.savez_compressed(entrada, xyz=xyz, metadatos=np.array("{}"))
            firma = hashlib.sha256(entrada.read_bytes()).hexdigest()
            procesar_geometria(entrada, raiz / "salida", vecinos=8, bloque=11)
            self.assertEqual(firma, hashlib.sha256(entrada.read_bytes()).hexdigest())
            with np.load(raiz / "salida" / "geometria.npz", allow_pickle=False) as datos:
                g = datos["geof"]
                self.assertEqual(g.shape, (50, 5))
                self.assertEqual(datos["nombres"].tolist(), ["linealidad", "planaridad", "dispersion", "verticalidad", "elevacion"])
            ply = PlyData.read(str(raiz / "salida" / "geometria.ply"), mmap=False)
            np.testing.assert_array_equal(ply["vertex"]["x"], xyz[:, 0])
            np.testing.assert_array_equal(ply["vertex"]["linealidad"], g[:, 0])
            with self.assertRaises(ValueError):
                procesar_geometria(entrada, raiz / "salida", vecinos=8)


if __name__ == "__main__":
    unittest.main()
