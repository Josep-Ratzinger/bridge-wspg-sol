"""Limpieza residual conservadora y normalización reversible, antes del completado.

Combina criterios estadísticos y de radio; protege cadenas localmente lineales.
Es una elección operativa de M1, no un parámetro prescrito por la tesis ni WSPG.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from plyfile import PlyData, PlyElement


def sha256(archivo):
    h=hashlib.sha256()
    with Path(archivo).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def filtrar_ruido(xyz, vecinos=20, desviaciones=3., radio=.12, minimo_vecinos=3, bloque=50000):
    x=np.asarray(xyz,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=3 or len(x)<=vecinos or not np.isfinite(x).all():
        raise ValueError('Se requieren más puntos que vecinos, con XYZ finitos.')
    if vecinos<4 or not np.isfinite([desviaciones,radio]).all() or desviaciones<=0 or radio<=0 or not 1<=minimo_vecinos<=vecinos or bloque<1:
        raise ValueError('Parámetros de filtrado inválidos.')
    tree=cKDTree(x);medias=np.empty(len(x),dtype=np.float32);cercanos=np.empty(len(x),dtype=np.uint16)
    for inicio in range(0,len(x),bloque):
        fin=min(inicio+bloque,len(x));d=tree.query(x[inicio:fin],k=vecinos+1,workers=4)[0][:,1:]
        medias[inicio:fin]=d.mean(axis=1);cercanos[inicio:fin]=(d<=radio).sum(axis=1)
    media=float(medias.mean(dtype=np.float64));std=float(medias.std(dtype=np.float64));umbral=media+desviaciones*std
    sospechosos=(medias>umbral)&(cercanos<minimo_vecinos)
    protegidos=np.zeros(len(x),dtype=bool);ids=np.flatnonzero(sospechosos&(cercanos>=1))
    # Cuatro puntos localmente alineados pueden ser una baranda o un borde fino.
    if len(ids):
        d,ind=tree.query(x[ids],k=4,workers=4);q=x[ind];q=q-q.mean(axis=1,keepdims=True)
        cov=np.einsum('nki,nkj->nij',q,q)/3.;lam=np.linalg.eigvalsh(cov)
        lineal=(lam[:,1]<.03*np.maximum(lam[:,2],1e-20))&(d[:,-1]<=3*radio)
        protegidos[ids[lineal]]=True
    quitar=sospechosos&~protegidos
    meta={'vecinos':vecinos,'desviaciones':desviaciones,'radio':radio,'minimo_vecinos_excluyendo_propio':minimo_vecinos,
          'media_distancias':media,'desviacion_distancias':std,'umbral_estadistico':umbral,
          'regla':f'media a {vecinos} vecinos > media global + {desviaciones} desviaciones Y menos de {minimo_vecinos} vecinos a radio {radio}; excepción lineal local',
          'sospechosos_antes_proteccion':int(sospechosos.sum()),'puntos_lineales_protegidos':int(protegidos.sum()),
          'retirados':int(quitar.sum()),'conservados':int((~quitar).sum()),'iteraciones':1,
          'limite':'no distingue semánticamente errores densos de superficies reales; requiere revisión visual'}
    return ~quitar,medias,meta


def normalizar(xyz):
    x=np.asarray(xyz,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=3 or not len(x) or not np.isfinite(x).all():raise ValueError('XYZ inválidos')
    centro=(x.min(axis=0)+x.max(axis=0))/2;escala=float(np.ptp(x,axis=0).max())
    if escala<=0:raise ValueError('La nube no tiene extensión')
    normalizada=((x-centro)/escala).astype(np.float32)
    return normalizada,centro,escala


def diagnosticar_orientacion(xyz):
    # Coherencia de Z respecto a la superficie horizontal dominante: no certifica gravedad.
    x=np.asarray(xyz);z=x[:,2];lo,hi=np.quantile(z,[.05,.95])
    if hi-lo<.02:return {'estado':'no estimada; rango insuficiente'}
    h,e=np.histogram(z,bins=np.arange(lo,hi+.02,.02));modo=float((e[h.argmax()]+e[h.argmax()+1])/2)
    ids=np.flatnonzero(np.abs(z-modo)<.07)[::10]
    if len(ids)<100:return {'estado':'no estimada; muestra insuficiente'}
    a=np.column_stack((x[ids,:2],np.ones(len(ids))));coef=np.linalg.lstsq(a,x[ids,2],rcond=None)[0]
    ang=float(np.degrees(np.arctan(np.linalg.norm(coef[:2]))))
    return {'plano_z_modo':modo,'inclinacion_superficie_dominante_grados':ang,'muestra':len(ids),
            'rotacion_aplicada':False,'limite':'coherencia geométrica de la orientación existente; no vertical física medida'}


def escribir_ply(archivo,xyz,rgb,ids,distancias=None):
    campos=[(n,'<f4') for n in 'xyz']+[(n,'u1') for n in ('red','green','blue')]+[('reduccion_id','<u4')]
    if distancias is not None:campos.append(('distancia_media','<f4'))
    v=np.empty(len(xyz),dtype=campos)
    for j,n in enumerate('xyz'):v[n]=xyz[:,j]
    for j,n in enumerate(('red','green','blue')):v[n]=rgb[:,j]
    v['reduccion_id']=ids
    if distancias is not None:v['distancia_media']=distancias
    PlyData([PlyElement.describe(v,'vertex')],text=False,byte_order='<').write(str(archivo))


def limpiar_preparar(manual,reduccion,salida,vecinos=20,desviaciones=3.,radio=None,minimo_vecinos=3):
    manual,reduccion,salida=map(Path,(manual,reduccion,salida))
    destinos=[salida/n for n in ('puente_preparado.ply','preparacion.npz','puntos_retirados.ply')]
    if any(p.exists() for p in destinos):raise ValueError('Ya existe una salida de preparación; no se sobrescribe')
    if any(p.resolve() in (manual.resolve(),reduccion.resolve()) for p in destinos):raise ValueError('La salida coincide con una entrada')
    with np.load(reduccion,allow_pickle=False) as d:
        x=d['xyz'];rgb=d['rgb'];mapa=d['punto_a_celda'];conteos=d['conteos'];origen=json.loads(str(d['metadatos']))
    if origen['sha256_entrada']!=sha256(manual):raise ValueError('La reducción no corresponde al PLY manual actual; recalcular reducción')
    if len(mapa)!=origen['puntos_originales'] or not np.array_equal(np.bincount(mapa,minlength=len(x)),conteos):raise ValueError('Correspondencias inválidas')
    radio=float(4*origen['tamano_celda']) if radio is None else radio
    print(f'Filtrando {len(x):,} puntos reducidos; k={vecinos}, radio={radio:g}...',flush=True)
    conservar,medias,filtro=filtrar_ruido(x,vecinos,desviaciones,radio,minimo_vecinos)
    if conservar.sum()<=vecinos:raise ValueError('Filtrado sin suficientes puntos; revisar parámetros')
    ids=np.flatnonzero(conservar).astype(np.uint32);final=x[conservar];colores=rgb[conservar]
    norm,centro,escala=normalizar(final)
    inversa=norm.astype(np.float64)*escala+centro
    error=float(np.abs(inversa-final).max())
    indice_nuevo=np.full(len(x),-1,dtype=np.int32);indice_nuevo[conservar]=np.arange(conservar.sum(),dtype=np.int32)
    mapa_total=indice_nuevo[mapa];indices_originales=np.flatnonzero(mapa_total>=0).astype(np.uint32);mapa_limpio=mapa_total[indices_originales].astype(np.uint32)
    conteos_final=conteos[conservar]
    if not np.array_equal(np.bincount(mapa_limpio,minlength=len(final)),conteos_final):raise AssertionError('Error en correspondencias filtradas')
    meta={'etapa':'limpieza residual y preparación previa al completado','archivo_manual':manual.name,'sha256_manual':origen['sha256_entrada'],
        'sha256_reduccion_entrada':sha256(reduccion),'puntos_originales_entrada_manual':len(mapa),'puntos_originales':len(indices_originales),
        'puntos_reducidos_antes':len(x),'puntos_reducidos':len(final),'retirados':int((~conservar).sum()),
        'tamano_celda':origen['tamano_celda'],'unidades':origen['unidades'],'filtro':filtro,
        'normalizacion':{'formula':'xyz_normalizado=(xyz-centro)/escala; xyz=xyz_normalizado*escala+centro',
            'centro':centro.tolist(),'escala':escala,'escala_isotropica':True,'rotacion':np.eye(3).tolist(),
            'error_maximo_reversion_unidades_nube':error,'aplicada_solo_a':'xyz_normalizado del NPZ; PLY y xyz conservan marco local de entrada'},
        'orientacion':diagnosticar_orientacion(final),'campos_ply':['x','y','z','red','green','blue','reduccion_id'],
        'correspondencia':'punto_a_celda corresponde a indices_originales en la nube manual; puntos filtrados no se reasignan',
        'estado':'limpieza conservadora ejecutada; pendiente revisión visual del usuario antes de completado',
        'completado':False,'version_numpy':np.__version__,
        'fuente_filtros':'https://www.open3d.org/docs/latest/tutorial/geometry/pointcloud_outlier_removal.html',
        'adaptacion':'combinación conservadora SOR Y radio, con protección lineal propia; implementación NumPy/SciPy'}
    salida.mkdir(parents=True,exist_ok=True)
    escribir_ply(destinos[0],final,colores,ids)
    # Rojo para identificar lo retirado al superponerlo a la nube preparada.
    rojos=np.tile(np.array([255,50,50],dtype=np.uint8),((~conservar).sum(),1))
    escribir_ply(destinos[2],x[~conservar],rojos,np.flatnonzero(~conservar).astype(np.uint32),medias[~conservar])
    np.savez_compressed(destinos[1],xyz=final,rgb=colores,xyz_normalizado=norm,centro=centro,escala=np.array(escala),
        indices_reduccion=ids,indices_originales=indices_originales,punto_a_celda=mapa_limpio,conteos=conteos_final,
        metadatos=np.array(json.dumps(meta,ensure_ascii=False)))
    print(f'Conservados: {len(final):,}; retirados: {meta["retirados"]:,}; error de normalización: {error:.3g}',flush=True)
    return meta
