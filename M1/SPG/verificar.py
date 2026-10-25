"""Verifica correspondencias, procedencia y dimensiones del resultado SPG."""
import argparse,json
from pathlib import Path
import numpy as np
from plyfile import PlyData
from .particion import sha256


def verificar(salida):
    salida=Path(salida).resolve();proceso=json.loads((salida/'proceso.json').read_text(encoding='utf8'))
    if len(proceso['etapas'])!=5 or 'etapa_actual' in proceso:raise ValueError('El recorrido SPG no terminó')
    for stage in proceso['etapas'].values():
        for name,expected in stage['archivos'].items():
            if sha256(salida/name)!=expected:raise ValueError('Archivo modificado: '+name)
    original=Path(proceso['parametros']['entrada'])
    if sha256(original)!=proceso['parametros']['sha256_entrada']:raise ValueError('El completado de entrada cambió')
    vertices=PlyData.read(str(original))['vertex'].data
    with np.load(salida/'entrada/nube_completada.npz') as data:
        xyz=data['xyz'];rgb=data['rgb'];origen=data['origen'];n=len(xyz)
        if not np.array_equal(data['punto_a_celda'],np.arange(n)) or not (data['conteos']==1).all():raise ValueError('Se aplicó una reducción inesperada')
        for j,k in enumerate(('x','y','z')):
            if not np.array_equal(xyz[:,j],vertices[k]):raise ValueError('XYZ alterado')
        for j,k in enumerate(('red','green','blue')):
            if not np.array_equal(rgb[:,j],vertices[k]):raise ValueError('RGB alterado')
        for k in ('origen','punto_id','reduccion_id'):
            if not np.array_equal(data[k],vertices[k]):raise ValueError('Identidad o procedencia alterada')
    with np.load(salida/'superpuntos/superpuntos.npz') as groups:
        labels=groups['celda_a_superpunto'];n_groups=len(groups['tamanos_reducidos'])
        if len(labels)!=n or not np.array_equal(labels,groups['punto_a_superpunto']):raise ValueError('Correspondencia de partición inválida')
        observed=groups['observados_por_superpunto'];estimated=groups['estimados_por_superpunto']
        if not np.array_equal(observed+estimated,groups['tamanos_reducidos']):raise ValueError('Conteos de origen incorrectos')
    with np.load(salida/'geometria/geometria.npz') as geo:
        geof=geo['geof']
        if geof.shape!=(n,5) or not geo['valido'].all() or not np.isfinite(geof).all():raise ValueError('Geometría inválida')
    with np.load(salida/'grafo/grafo.npz') as graph, np.load(salida/'entrada_red/entrada_superpuntos.npz') as net:
        edges=len(graph['origen'])
        if graph['atributos'].shape!=(edges,13) or not np.isfinite(graph['atributos']).all():raise ValueError('Atributos de aristas inválidos')
        if graph['centroides'].shape!=(n_groups,3):raise ValueError('Cantidad de nodos incoherente')
        if net['puntos'].shape!=(n_groups,128,8) or not np.isfinite(net['puntos']).all():raise ValueError('Entrada de red inválida')
        sample=net['indices_reducidos']
        if sample.max()>=n or sample.shape!=(n_groups,128):raise ValueError('Índices muestreados inválidos')
        if not np.array_equal(net['origen_muestreado'],origen[sample]):raise ValueError('Procedencia del muestreo perdida')
        if not np.array_equal(net['puntos'][:,:,3:],geof[sample]):raise ValueError('Atributos muestreados alterados')
        for data in (graph,net):
            if not np.array_equal(data['estimados_por_superpunto'],estimated) or not np.array_equal(data['observados_por_superpunto'],observed):raise ValueError('Procedencia no coincide entre etapas')
        pequeños=int((~net['embedding_habilitado']).sum())
    report={'verificado':True,'puntos':n,'superpuntos':n_groups,'conexiones_dirigidas':edges,
            'observados':int((origen==0).sum()),'estimados':int((origen==1).sum()),
            'grupos_solo_observados':int((estimated==0).sum()),'grupos_solo_estimados':int((observed==0).sum()),
            'grupos_mixtos':int(((observed>0)&(estimated>0)).sum()),'grupos_menores_de_25_con_mascara':pequeños,
            'coordenadas_rgb_ids_origen_preservados':True,'reduccion_adicional':False,
            'pureza_semantica_calculada':False,'motivo_pureza_pendiente':'faltan etiquetas de referencia',
            'sha256_completado':sha256(original)}
    (salida/'verificacion.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8')
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('salida',type=Path,nargs='?',default=Path(__file__).resolve().parent/'resultados');a=p.parse_args()
    print(json.dumps(verificar(a.salida),indent=2,ensure_ascii=False))


if __name__=='__main__':main()
