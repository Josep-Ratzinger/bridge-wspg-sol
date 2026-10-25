"""Inferencia final sobre el SPG completo, sin leer etiquetas de referencia."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from plyfile import PlyData,PlyElement
from .datos import ROOT,MODELOS,RESULTADOS,CLASSES,MASKS,PALETA,firma,guardar
from .utilidades import centros_relativos,tensors
from .estructura import calcular
from .WSPG.relacional import WSPGRelacional


def cargar_modelo(nombre='WSPG_SOL',modelos=MODELOS):
    path=Path(modelos)/nombre/'mejor.pt'
    ck=torch.load(path,map_location='cpu',weights_only=True);cfg=ck['config']
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=WSPGRelacional(strength=cfg.get('strength',1.),base_scale=cfg.get('base_scale',1.)).to(device)
    model.load_state_dict(ck['state_dict']);model.eval()
    scaler={k:v.numpy() for k,v in ck['scaler'].items()}
    np.testing.assert_array_equal(model.nm.cpu().numpy(),scaler['nm'])
    np.testing.assert_array_equal(model.ns.cpu().numpy(),scaler['ns'])
    return model,scaler,device,cfg


def leer_spg(spg):
    spg=Path(spg);files={'ply':spg/'superpuntos/superpuntos.ply',
        'particion':spg/'superpuntos/superpuntos.npz','grafo':spg/'grafo/grafo.npz',
        'entrada_red':spg/'entrada_red/entrada_superpuntos.npz'}
    before={k:firma(v) for k,v in files.items()}
    with np.load(files['entrada_red']) as f:d={k:f[k] for k in ['puntos','centros','embedding_habilitado']}
    with np.load(files['grafo']) as f:
        d['aristas']=np.stack([f['origen'],f['destino']]).astype('i8');d['atributos_aristas']=f['atributos']
    with np.load(files['particion']) as f:group=f['celda_a_superpunto']
    source=PlyData.read(str(files['ply']));vertex=source['vertex'].data
    if len(vertex)!=len(group):raise ValueError('PLY y partición de distinto tamaño')
    if group.min()<0 or group.max()>=len(d['puntos']):raise ValueError('Grupos inválidos')
    xyz=np.column_stack([vertex[k] for k in ['x','y','z']])
    d['atributos_nodo'],fit=calcular(xyz,group)
    d['centros']=centros_relativos(d['centros']);d['histogramas']=np.zeros((len(d['centros']),len(MASKS)))
    return d,group,source,files,before,fit


@torch.no_grad()
def aplicar(nombre='WSPG_SOL',spg=ROOT/'M1/SPG/resultados',salida=None,modelos=MODELOS):
    model,scaler,device,cfg=cargar_modelo(nombre,modelos)
    d,group,source,files,before,fit=leer_spg(spg);x,_=tensors(d,scaler,device)
    probs=model(*x).softmax(1);pred=probs.argmax(1).cpu().numpy().astype('i2');conf=probs.max(1).values.cpu().numpy()
    original=source['vertex'].data
    dtype=list(original.dtype.descr)+[('clase_predicha','<i2'),('confianza','<f4'),('red_spg','u1'),('green_spg','u1'),('blue_spg','u1')]
    v=np.empty(len(original),dtype=dtype)
    for k in original.dtype.names:v[k]=original[k]
    for a,k in enumerate(['red','green','blue']):v[k+'_spg']=original[k];v[k]=PALETA[pred[group],a]
    v['clase_predicha']=pred[group];v['confianza']=conf[group]
    dest=Path(salida) if salida else RESULTADOS/nombre;dest.mkdir(parents=True,exist_ok=True)
    PlyData([PlyElement.describe(v,'vertex')],text=False,comments=list(source.comments)+[
        'Prediccion aprendida en otros puentes; sin etiquetas Lurin en inferencia',
        'RGB de grupos originales en red_spg green_spg blue_spg']).write(str(dest/'puente_etiquetado.ply'))
    np.savez_compressed(dest/'predicciones.npz',clase_por_superpunto=pred,confianza_por_superpunto=conf)
    assert all(firma(v)==before[k] for k,v in files.items())
    assert all(np.array_equal(v[k],original[k]) for k in original.dtype.names if k not in ['red','green','blue'])
    guardar(dest/'verificacion.json',dict(puntos=len(v),superpuntos=len(pred),M1_sin_modificar=True,
        todos_los_puntos_conservados=True,atributos_originales_conservados=True,etiquetas_lurin_leidas=False,
        entrada_sha256=before,clases=CLASSES,salida_sha256=firma(dest/'puente_etiquetado.ply'),
        modelo=nombre,pesos_sha256=firma(Path(modelos)/nombre/'mejor.pt'),ajuste_automatico=fit))
    print('Puente etiquetado:',dest/'puente_etiquetado.ply',flush=True)
    return pred,conf


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--modelo',choices=['WSPG','WSPG_SOL'],default='WSPG_SOL')
    p.add_argument('--spg',type=Path,default=ROOT/'M1/SPG/resultados');p.add_argument('--salida',type=Path)
    p.add_argument('--modelos',type=Path,default=MODELOS);a=p.parse_args();aplicar(a.modelo,a.spg,a.salida,a.modelos)


if __name__=='__main__':main()
