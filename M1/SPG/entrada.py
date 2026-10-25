"""Adapta el completado a SPG sin otra reducción, conservando IDs y origen."""
import json
from pathlib import Path
import numpy as np
from plyfile import PlyData
from .particion import sha256


def preparar_completado(archivo,salida,registro=None):
    archivo,salida=Path(archivo).resolve(),Path(salida).resolve()
    if salida.exists():raise ValueError('La entrada SPG ya existe; no se sobrescribe')
    firma=sha256(archivo)
    registro=Path(registro) if registro else archivo.parent/'ejecucion.json'
    aprobado=json.loads(registro.read_text(encoding='utf8')) if registro.is_file() else None
    if aprobado and aprobado['output_sha256']!=firma:raise ValueError('El PLY no coincide con la ejecución de completado')
    ply=PlyData.read(str(archivo));v=ply['vertex'].data
    requeridos={'x','y','z','red','green','blue','origen','punto_id','reduccion_id'}
    if not requeridos.issubset(v.dtype.names or ()):raise ValueError('Faltan coordenadas, RGB, origen o identificadores del completado')
    xyz=np.column_stack([v[k] for k in ('x','y','z')]);rgb=np.column_stack([v[k] for k in ('red','green','blue')])
    origen=np.asarray(v['origen']).copy();punto_id=np.asarray(v['punto_id']).copy();reduccion_id=np.asarray(v['reduccion_id']).copy()
    if not len(xyz) or not np.isfinite(xyz).all() or not np.isin(origen,[0,1]).all():raise ValueError('Coordenadas o procedencia inválidas')
    if len(xyz)>=2**32:raise ValueError('Entrada fuera del rango de índices de SPG')
    observados=int((origen==0).sum());estimados=int((origen==1).sum())
    if not np.array_equal(punto_id[origen==0],np.arange(observados)) or not (punto_id[origen==1]==-1).all():raise ValueError('Identidades del completado inconsistentes')
    if aprobado and (aprobado['original_points_preserved']!=observados or aprobado['estimated_points_added']!=estimados):raise ValueError('Conteos no coinciden con el completado')
    meta={'metodo':'adaptación de formato; correspondencia identidad, sin reducción adicional de densidad',
          'archivo_entrada':str(archivo),'sha256_entrada':firma,'puntos_originales':len(xyz),'puntos_reducidos':len(xyz),
          'observados':observados,'estimados':estimados,'unidades':'las del completado; sin certificar metros',
          'origen':'0 observado / 1 estimado; no es clase semántica','hipotesis_geometricas_usadas_como_etiquetas':False,
          'completado_verificado_por_hash':aprobado is not None,'reduccion_adicional':False}
    salida.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(salida,xyz=xyz,rgb=rgb,origen=origen,punto_id=punto_id,reduccion_id=reduccion_id,
                        punto_a_celda=np.arange(len(xyz),dtype='u4'),conteos=np.ones(len(xyz),dtype='u4'),
                        metadatos=np.array(json.dumps(meta,ensure_ascii=False)))
    return meta
