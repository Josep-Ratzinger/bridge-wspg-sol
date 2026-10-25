"""Pruebas del parámetro de corte, conectividad y correspondencia original."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from plyfile import PlyData

from M1.SPG.particion import grafo_vecinos, resolver_particion, generar_superpuntos


class ParticionTest(unittest.TestCase):
    def test_regularizacion_realmente_controla_cortes(self):
        geof = np.array([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0]], dtype=np.float32)
        u = np.array([0, 1, 2], dtype=np.uint32)
        v = np.array([1, 2, 3], dtype=np.uint32)
        dist = np.ones(3, dtype=np.float32)
        bajo, _, _ = resolver_particion(geof, u, v, dist, 0.001, verbose=False)
        alto, _, _ = resolver_particion(geof, u, v, dist, 100, verbose=False)
        self.assertEqual(len(np.unique(bajo)), 2)
        self.assertEqual(bajo[0], bajo[1])
        self.assertNotEqual(bajo[1], bajo[2])
        self.assertEqual(len(np.unique(alto)), 1)
        np.testing.assert_array_equal(geof[:, 3], 0)  # no modifica entrada

    def test_elevacion_participa_en_los_cortes(self):
        geof = np.zeros((4, 5), dtype=np.float32)
        geof[2:, 4] = 1
        u, v = np.array([0, 1, 2]), np.array([1, 2, 3])
        grupos, _, _ = resolver_particion(geof, u, v, np.ones(3), 0.001, verbose=False)
        self.assertNotEqual(grupos[1], grupos[2])
        self.assertEqual(grupos[0], grupos[1])
        with self.assertRaisesRegex(ValueError, "cinco"):
            resolver_particion(geof[:, :4], u, v, np.ones(3), verbose=False)

    def test_grafo_ordenado_sin_bucles_y_grupos_conexos(self):
        xyz = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], float)
        u, v, dist = grafo_vecinos(xyz, vecinos=2, bloque=2)
        self.assertEqual(len(u), 8)
        self.assertTrue(np.all(u[1:] >= u[:-1]))
        self.assertTrue(np.all(u != v))
        np.testing.assert_allclose(dist, np.linalg.norm(xyz[u] - xyz[v], axis=1))
        geof = np.zeros((4, 5), dtype=np.float32)
        grupos, _, _ = resolver_particion(geof, np.array([0, 2]), np.array([1, 3]), np.ones(2), verbose=False)
        self.assertEqual(len(np.unique(grupos)), 2)
        self.assertNotEqual(grupos[0], grupos[2])
        with self.assertRaises(ValueError):
            resolver_particion(geof, u[::-1], v[::-1], dist[::-1], verbose=False)

    def test_archivos_conservan_todos_los_puntos(self):
        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp)
            red, geo = raiz / "reduccion.npz", raiz / "geometria.npz"
            xyz = np.column_stack([np.arange(8), np.zeros((8, 2))]).astype(np.float32)
            mapa = np.repeat(np.arange(8, dtype=np.uint32), np.arange(1, 9))
            np.savez_compressed(red, xyz=xyz, punto_a_celda=mapa, conteos=np.arange(1, 9), metadatos=np.array("{}"))
            firma = hashlib.sha256(red.read_bytes()).hexdigest()
            meta = {"sha256_entrada": firma}
            np.savez_compressed(geo, geof=np.tile([1, 0, 0, 0, 0], (8, 1)).astype(np.float32),
                valido=np.ones(8, dtype=bool), nombres=np.array(["linealidad", "planaridad", "dispersion", "verticalidad", "elevacion"]),
                metadatos=np.array(json.dumps(meta)))
            informe = generar_superpuntos(red, geo, raiz / "salida", vecinos=2)
            self.assertEqual(informe["puntos_originales"], 36)
            with np.load(raiz / "salida" / "superpuntos.npz", allow_pickle=False) as datos:
                np.testing.assert_array_equal(datos["punto_a_superpunto"], datos["celda_a_superpunto"][mapa])
                self.assertEqual(int(datos["tamanos_originales"].sum()), 36)
            ply = PlyData.read(str(raiz / "salida" / "superpuntos.ply"), mmap=False)
            np.testing.assert_array_equal(ply["vertex"]["x"], xyz[:, 0])
            self.assertEqual(firma, hashlib.sha256(red.read_bytes()).hexdigest())
            with self.assertRaises(ValueError):
                generar_superpuntos(red, geo, raiz / "salida", vecinos=2)
            # Una geometría de otro archivo debe rechazarse antes del cálculo.
            np.savez_compressed(red, xyz=xyz + 1)
            with self.assertRaisesRegex(ValueError, "no corresponde"):
                generar_superpuntos(red, geo, raiz / "otra_salida", vecinos=2)


if __name__ == "__main__":
    unittest.main()
