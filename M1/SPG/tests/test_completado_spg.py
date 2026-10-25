import json,tempfile,unittest
from pathlib import Path
import numpy as np
from plyfile import PlyData,PlyElement
from M1.SPG.entrada import preparar_completado
from M1.SPG.procedencia import resumir_origen
from M1.SPG.particion import sha256
from M1.SPG.__main__ import ejecutar


class CompletedSPGTest(unittest.TestCase):
    def cloud(self,folder):
        rng=np.random.default_rng(33);n=240
        v=np.zeros(n,dtype=[(k,'f4') for k in ('x','y','z')]+[(k,'u1') for k in ('red','green','blue','origen','forma_propuesta')]+[('punto_id','i4'),('reduccion_id','u4')])
        xyz=rng.normal(size=(n,3)).astype('f4')
        for j,k in enumerate(('x','y','z')):v[k]=xyz[:,j]
        v['red']=30;v['green']=120;v['blue']=210;v['origen'][200:]=1;v['forma_propuesta']=2
        v['punto_id'][:200]=np.arange(200);v['punto_id'][200:]=-1;v['reduccion_id'][:200]=np.arange(200)+17;v['reduccion_id'][200:]=np.iinfo('u4').max
        file=folder/'puente.ply';PlyData([PlyElement.describe(v,'vertex')],text=False).write(str(file))
        (folder/'ejecucion.json').write_text(json.dumps({'output_sha256':sha256(file),'original_points_preserved':200,'estimated_points_added':40}))
        return file,xyz

    def test_group_counts_preserve_provenance_without_semantic_classes(self):
        r=resumir_origen(np.array([0,1,1,0,1],dtype='u1'),np.array([0,0,1,1,1]))
        np.testing.assert_array_equal(r['observados_por_superpunto'],[1,1]);np.testing.assert_array_equal(r['estimados_por_superpunto'],[1,2])
        np.testing.assert_allclose(r['fraccion_estimada'],[.5,2/3])
        with self.assertRaises(ValueError):resumir_origen(np.array([2]),np.array([0]))

    def test_input_keeps_points_and_ignores_shape_hypotheses(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);file,xyz=self.cloud(folder);out=folder/'entrada.npz';meta=preparar_completado(file,out)
            with np.load(out) as d:
                np.testing.assert_array_equal(d['xyz'],xyz);np.testing.assert_array_equal(d['punto_a_celda'],np.arange(len(xyz)))
                self.assertNotIn('forma_propuesta',d.files);self.assertNotIn('clase',d.files)
                self.assertEqual(int(d['origen'].sum()),40);self.assertEqual(meta['reduccion_adicional'],False)
            (folder/'ejecucion.json').write_text(json.dumps({'output_sha256':'incorrecto'}))
            with self.assertRaises(ValueError):preparar_completado(file,folder/'otra.npz')

    def test_entire_pipeline_and_resume_keep_all_points_and_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);file,xyz=self.cloud(folder);out=folder/'spg'
            r=ejecutar(file,out);self.assertEqual(len(r['etapas']),5)
            with np.load(out/'superpuntos/superpuntos.npz') as groups, np.load(out/'grafo/grafo.npz') as graph, np.load(out/'entrada_red/entrada_superpuntos.npz') as net:
                self.assertEqual(len(groups['punto_a_superpunto']),len(xyz));self.assertEqual(int(groups['estimados_por_superpunto'].sum()),40)
                np.testing.assert_array_equal(graph['estimados_por_superpunto'],groups['estimados_por_superpunto'])
                np.testing.assert_array_equal(net['origen_muestreado'],(net['indices_reducidos']>=200).astype('u1'))
                self.assertEqual(net['puntos'].shape[1:],(128,8));self.assertEqual(graph['atributos'].shape[1],13)
            before=sha256(out/'grafo/grafo.npz');ejecutar(file,out,True);self.assertEqual(before,sha256(out/'grafo/grafo.npz'))


if __name__=='__main__':unittest.main()
