import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from plyfile import PlyData
from M1.SPG.grafo import aristas_voronoi, agregar_conexiones, generar_grafo
from M1.SPG.pureza import calcular_pureza, plantilla, evaluar_pureza
from M1.SPG.particion import sha256


class GrafoPurezaTest(unittest.TestCase):
    def test_tetraedro_y_reciprocidad(self):
        xyz = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float)
        grupos = np.array([0,0,1,1])
        u,v = aristas_voronoi(xyz,grupos)
        self.assertEqual(set(zip(u,v)), {(0,2),(0,3),(1,2),(1,3)})
        g = agregar_conexiones(xyz,grupos,u,v)
        np.testing.assert_array_equal(g["origen"],[0,1])
        self.assertEqual(g["atributos"].shape,(2,13))
        np.testing.assert_allclose(g["atributos"][0,:3],[.5,-.5,-.5])
        np.testing.assert_allclose(g["atributos"][0,3:6],[.5,.5,.5])
        np.testing.assert_allclose(g["atributos"][1,:3],-g["atributos"][0,:3])
        np.testing.assert_allclose(g["atributos"][1,3:6],g["atributos"][0,3:6])
        np.testing.assert_allclose(g["atributos"][1,6:],-g["atributos"][0,6:])
        np.testing.assert_array_equal(g["contactos"],[4,4])
        otro = agregar_conexiones(xyz+10000,grupos,np.r_[u,v,u],np.r_[v,u,v])
        np.testing.assert_allclose(otro["atributos"],g["atributos"],atol=1e-6)

    def test_grafo_sin_conexiones_y_degeneracion(self):
        xyz = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float)
        u,v = aristas_voronoi(xyz,np.zeros(4,dtype=int))
        g = agregar_conexiones(xyz,np.zeros(4,dtype=int),u,v)
        self.assertEqual(g["atributos"].shape,(0,13))
        with self.assertRaises(ValueError):
            aristas_voronoi(xyz*np.array([1,1,0]),np.array([0,0,1,1]))

    def test_pureza_conocida_y_etiquetado_incompleto(self):
        grupos = np.array([0,0,0,0,1,1])
        clases = np.array([0,0,0,1,1,1])
        resumen,d = calcular_pureza(grupos,clases,np.array([4,2]))
        self.assertAlmostEqual(resumen["pureza_ponderada_puntos"],5/6)
        self.assertAlmostEqual(resumen["pureza_media_grupos_evaluados"],.875)
        np.testing.assert_allclose(d["pureza"],[.75,1])
        parcial,_ = calcular_pureza(grupos,np.array([0,-1,-1,-1,1,-1]),np.array([4,2]))
        self.assertAlmostEqual(parcial["cobertura_puntos"],2/6)
        self.assertIn("subconjunto",parcial["alcance"])
        with self.assertRaises(ValueError):
            calcular_pureza(grupos,np.full(6,-1),np.array([4,2]))

    def test_archivos_ids_reordenados_firmas_y_no_sobrescritura(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            xyz=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype=np.float32)
            red=p/"red.npz"; part=p/"part.npz"
            np.savez(red,xyz=xyz,rgb=np.full((4,3),100,dtype=np.uint8))
            np.savez(part,celda_a_superpunto=np.array([0,0,1,1]),metadatos=np.array(json.dumps({"sha256_reduccion":sha256(red)})))
            informe=generar_grafo(red,part,p/"grafo")
            self.assertEqual(informe["conexiones_dirigidas"],2)
            with self.assertRaises(ValueError):
                generar_grafo(red,part,p/"grafo")
            plantilla(red,p/"etiquetas.ply")
            ply=PlyData.read(str(p/"etiquetas.ply"),mmap=False)
            ply["vertex"].data["clase"]=[0,1,1,1]
            ply["vertex"].data=ply["vertex"].data[[3,1,0,2]].copy()
            ply.write(str(p/"etiquetas.ply"))
            resumen=evaluar_pureza(red,part,p/"etiquetas.ply",p/"pureza")
            self.assertAlmostEqual(resumen["pureza_ponderada_puntos"],.75)
            self.assertEqual(resumen["cobertura_puntos"],1)
            ply["vertex"].data["punto_id"][0]=ply["vertex"].data["punto_id"][1]
            ply.write(str(p/"mal.ply"))
            with self.assertRaises(ValueError):
                evaluar_pureza(red,part,p/"mal.ply",p/"error")
