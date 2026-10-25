"""Propuestas y anotaciones preliminares por punto; no utiliza la partición SPG."""
import json,hashlib
from pathlib import Path
import numpy as np
from plyfile import PlyData,PlyElement

MODULE=Path(__file__).resolve().parent
PROJECT=MODULE.parents[1]


def sha256(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def between(v,limits):return (v>=limits[0])&(v<=limits[1])


def asignar(xyz,rgb,origen,cfg):
    """Referencia parcial con regiones corregidas por el usuario, sin consultar SPG."""
    xyz,rgb,origen=map(np.asarray,(xyz,rgb,origen));n=len(xyz)
    if xyz.shape!=(n,3) or rgb.shape!=xyz.shape or origen.shape!=(n,) or not np.isfinite(xyz).all():raise ValueError('Datos inválidos')
    if not np.isin(origen,[0,1]).all():raise ValueError('Procedencia inválida')
    x,y,z=xyz.T;span=between(x,cfg['limites_span_x']);width=between(y,cfg['limites_tablero_y']);plane=cfg['plano_tablero_z']
    propuesta=np.full(n,-1,dtype='i2');referencia=np.full(n,-1,dtype='i2');confidence=np.zeros(n,dtype='f4')
    r,g,b=rgb.astype('f4').T;yellow=(r>35)&(g>30)&(r>b*1.12)&(g>b*1.12)
    green=(g>r*1.2)&(g>b*1.15);near_row=between(x,cfg['fila_columnas_x'])&between(y,cfg['fila_columnas_y'])
    base=cfg['base_columnas_z_intercepto']+cfg['base_columnas_z_pendiente_y']*y
    column=near_row&(z>base+.10)&(z<cfg['base_viga_z']-.05)
    head=near_row&(z>=cfg['base_viga_z']-.04)&(z<cfg['techo_cabezal_z']+.06)
    edge=(between(y,cfg['bandas_baranda_y'][0])|between(y,cfg['bandas_baranda_y'][1]))
    beam=span&edge&(z>plane-1.45)&(z<plane-.25)
    strip=span&between(y,cfg['franja_tablero_y'])&between(z-plane,cfg['franja_tablero_z_relativo'])&~green
    deck=(span&width&(z>=plane-.18)&(z<=plane+.25)&~green)|strip
    rails=edge
    railing=span&rails&(z>plane+.26)&(z<plane+2.05)&~green
    access=(~span)&between(z-plane,cfg['accesos_z_relativo'])&~green
    divider=span&between(y,cfg['separador_y'])&between(z-plane,cfg['separador_z_relativo'])&~green
    lower=(origen==0)&between(x,cfg['superficie_inferior_x'])&between(y,cfg['superficie_inferior_y'])&between(z-base,cfg['superficie_inferior_z_relativo_base'])&~green
    for cls,mask,score in [(4,access,.80),(6,lower,.75),(2,deck,.88),(3,railing,.82),(5,divider,.85),(1,beam|head,.80),(0,column,.90)]:
        propuesta[mask]=cls;confidence[mask]=score
    # Referencia visual preliminar: solo observaciones y superficies reconocibles.
    # No se incluyen estimados ni zonas de transición elegidas por la partición.
    obs=origen==0;interior=span
    road_lanes=between(y,[-9.7,-.35])|between(y,[2.0,11.5])
    ref_deck=obs&deck&interior&road_lanes&(z>=plane-.10)&(z<=plane+.12)
    ref_deck|=obs&strip&between(z-plane,[-.10,.48])
    ref_columns=obs&column&(z>base+.28)&(z<cfg['base_viga_z']-.20)
    ref_beams=obs&((beam&yellow&(z>plane-1.20)&(z<plane-.42))|(head&yellow&(z>cfg['base_viga_z']+.12)&(z<cfg['techo_cabezal_z']-.08)))
    ref_rails=obs&railing&interior&edge&(z>plane+.48)&(z<plane+1.65)
    ref_access=obs&access&between(z-plane,[-.40,.28])
    ref_divider=obs&divider&between(y,[.15,1.40])&between(z-plane,[-.02,.65])
    ref_lower=obs&lower&between(z-base,[-.45,.05])
    for cls,mask in [(4,ref_access),(6,ref_lower),(2,ref_deck),(3,ref_rails),(5,ref_divider),(1,ref_beams),(0,ref_columns)]:referencia[mask]=cls
    # No dejar en referencia una clase distinta de la propuesta en zonas solapadas.
    referencia[referencia!=propuesta]=-1
    # Ninguna etiqueta obtenida del color/forma propuestos por el completado.
    referencia[origen==1]=-1
    return propuesta,referencia,confidence


def escribir_ply(path,xyz,rgb,origen,ids_fuente,clases,cfg,reference=False):
    mask=clases>=0 if reference else np.ones(len(xyz),bool);ids=np.flatnonzero(mask)
    field='clase' if reference else 'clase_propuesta'
    dtype=[(k,'<f4') for k in ('x','y','z')]+[(k,'u1') for k in ('red','green','blue','origen')]+[('punto_id','<u4'),('punto_id_fuente','<i4'),(field,'<i2')]
    vertices=np.empty(len(ids),dtype=dtype)
    for j,k in enumerate(('x','y','z')):vertices[k]=xyz[ids,j]
    colors=np.full((len(ids),3),145,dtype='u1')
    for c in cfg['clases']:colors[clases[ids]==c['id']]=c['rgb']
    for j,k in enumerate(('red','green','blue')):vertices[k]=colors[:,j]
    vertices['origen']=origen[ids];vertices['punto_id']=ids;vertices['punto_id_fuente']=ids_fuente[ids];vertices[field]=clases[ids]
    comments=['tipo_etiquetas=anotacion_asistida_preliminar' if reference else 'tipo_etiquetas=propuesta_automatica',
              'no validacion humana experta; no ground truth definitivo',
              'punto_id indice estable en nube_completada.npz; origen 0 observado 1 estimado']
    path.parent.mkdir(parents=True,exist_ok=True);PlyData([PlyElement.describe(vertices,'vertex')],text=False,byte_order='<',comments=comments).write(str(path))


def generar(reemplazar=False):
    cfg_path=MODULE/'configs/lurin.json';cfg=json.loads(cfg_path.read_text(encoding='utf8'))
    source=PROJECT/'M1/SPG/resultados/entrada/nube_completada.npz';out=MODULE/'resultados'
    if (out/'clases.npz').exists() and not reemplazar:raise ValueError('La anotación ya existe; usar actualización explícita')
    firma=sha256(source)
    if firma!=cfg['sha256_entrada_autorizada']:raise ValueError('Estas regiones son específicas de Lurín; no aplicarlas automáticamente a otra nube')
    with np.load(source) as d:xyz=d['xyz'];rgb=d['rgb'];origen=d['origen'];ids=d['punto_id'];reduccion=d['reduccion_id']
    proposed,reference,confidence=asignar(xyz,rgb,origen,cfg)
    out.mkdir(parents=True,exist_ok=True)
    previous=None
    if reemplazar and (out/'anotacion.json').exists():
        previous={'anotacion_previa':json.loads((out/'anotacion.json').read_text(encoding='utf8'))}
        score=out/'pureza_preliminar/pureza.json'
        if score.exists():previous['pureza_previa']=json.loads(score.read_text(encoding='utf8'))
        previous['comparabilidad']='Cambian taxonomía y cobertura; la diferencia de pureza no demuestra mejora del SPG, que permanece fijo.'
        (out/'revision').mkdir(exist_ok=True);(out/'revision/cambios_anotacion.json').write_text(json.dumps(previous,indent=2,ensure_ascii=False),encoding='utf8')
    tmp_npz=out/'clases_nuevas.npz';tmp_ply=out/'puente_clases_nuevo.ply';tmp_ref=MODULE/'datos/referencia_nueva.ply'
    np.savez_compressed(tmp_npz,clase_propuesta=proposed,clase_referencia_preliminar=reference,confianza_heuristica=confidence,origen=origen,punto_id_fuente=ids,reduccion_id=reduccion)
    escribir_ply(tmp_ply,xyz,rgb,origen,ids,proposed,cfg)
    escribir_ply(tmp_ref,xyz,rgb,origen,ids,reference,cfg,True)
    for a,b in [(tmp_npz,out/'clases.npz'),(tmp_ply,out/'puente_clases.ply'),(tmp_ref,MODULE/'datos/referencia_preliminar.ply')]:a.replace(b)
    counts={str(c['id']):{'nombre':c['nombre'],'propuestos_observados':int(((proposed==c['id'])&(origen==0)).sum()),
                          'propuestos_estimados':int(((proposed==c['id'])&(origen==1)).sum()),
                          'referencia_preliminar':int((reference==c['id']).sum()) if c['id']>=0 else 0} for c in cfg['clases']}
    record={'metodo':cfg['metodo'],'especifico_de_lurin':True,'clasificador_entrenado':False,'usa_particion_spg_para_etiquetar':False,
            'sha256_entrada':firma,'entrada':str(source),'sha256_config':sha256(cfg_path),'clases':counts,'referencia':cfg['referencia'],
            'puntos':len(xyz),'observados':int((origen==0).sum()),'estimados':int((origen==1).sum()),
            'puntos_referencia':int((reference>=0).sum()),'cobertura_referencia_observados':float((reference>=0).sum()/(origen==0).sum()),
            'confianza':'puntuación heurística de propuesta, no probabilidad calibrada',
            'puntos_sin_asignar':'elementos dudosos, transiciones y objetos fuera de las regiones reconocibles; no se fuerza su clasificación',
            'superficie_inferior':'categoría geométrica de fondo; no identifica el material ni confirma que sea cimentación',
            'coordenadas_originales_no_modificadas':True,'sha256_codigo':sha256(Path(__file__)),
            'archivos_sha256':{str(p.relative_to(MODULE)):sha256(p) for p in (out/'clases.npz',out/'puente_clases.ply',MODULE/'datos/referencia_preliminar.ply')}}
    if sha256(source)!=firma:raise ValueError('Entrada modificada')
    (out/'anotacion.json').write_text(json.dumps(record,indent=2,ensure_ascii=False),encoding='utf8')
    if reemplazar:
        for name in ('pureza.json','pureza.npz'):
            p=(out/'pureza_preliminar'/name).resolve()
            if not p.is_relative_to(MODULE.resolve()):raise ValueError('Salida fuera del módulo')
            if p.exists():p.unlink()
    return record
