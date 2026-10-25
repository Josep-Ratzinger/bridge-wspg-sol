"""Preparación del corpus final, con taxonomía correcta y geometría densa."""
import argparse,json,zipfile,hashlib,urllib.request
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from plyfile import PlyData
from .datos import ROOT,BASE,CORPUS,CLASSES,MASKS,SOURCES,firma,guardar
from .taxonomia import semanticbridge,uav
from .estructura import calcular as estructura
from .sinteticos import generar
from M1.SPG.geometria import calcular_geometria,elevacion_z_minmax,descriptores
from M1.SPG.particion import grafo_vecinos,resolver_particion
from M1.SPG.grafo import aristas_voronoi,agregar_conexiones
from M1.SPG.preparacion import muestrear_grupos
def marco(xyz):
    """PCA XY sin etiquetas, Z conservado. Traslación numérica reversible."""
    center=np.median(xyz,axis=0); q=np.asarray(xyz-center,dtype='f8')
    _,v=np.linalg.eigh(np.cov(q[:,:2].T)); axis=v[:,-1]
    if axis[np.argmax(abs(axis))]<0: axis=-axis
    rot=np.eye(3); rot[:2,:2]=np.array([[axis[0],-axis[1]],[axis[1],axis[0]]])
    return (q@rot).astype('f4'),{'centro':center.tolist(),'rotacion':rot.tolist()}

def read_semantic(z,member,limit,seed):
    # plyfile sobre ZipExtFile recorre decenas de millones de filas en Python.
    # Esta fuente tiene tres bloques escalares fijos: lectura vectorizada.
    with z.open(member) as f:
        header=[]
        for _ in range(50):
            line=f.readline().decode('ascii').strip(); header.append(line)
            if line=='end_header': break
        n=int(next(s for s in header if s.startswith('element points ')).split()[-1])
        expected=['ply','format binary_little_endian 1.0',f'element points {n}',
            'property float x','property float y','property float z',f'element color {n}',
            'property uchar red','property uchar green','property uchar blue',
            f'element label {n}','property uchar label','end_header']
        if header!=expected: raise ValueError('Esquema SemanticBridge distinto al auditado')
        idx=np.sort(np.random.default_rng(seed).choice(n,min(n,limit),replace=False))
        raw=f.read(n*12)
        if len(raw)!=n*12: raise ValueError('XYZ truncado')
        xyz=np.frombuffer(raw,dtype='<f4').reshape(n,3)[idx].copy(); del raw
        if len(f.read(n*3))!=n*3: raise ValueError('RGB truncado')
        raw=f.read(n)
        if len(raw)!=n: raise ValueError('Etiquetas truncadas')
        labels=np.frombuffer(raw,dtype='u1')[idx].copy()
        if f.read(1): raise ValueError('Contenido inesperado tras las etiquetas')
    return xyz,labels,n

def read_uav(path,limit,seed):
    p=PlyData.read(str(path)); d=p.elements[0].data; n=len(d)
    idx=np.sort(np.random.default_rng(seed).choice(n,min(n,limit),replace=False))
    xyz=np.column_stack([d[k][idx] for k in ['x','y','z']]); labels=d['scalar_label'][idx].astype('i4')
    return xyz,labels,n

def preparar_grafo(xyz,masks,out,metadata):
    if out.exists():
        report=json.loads(out.with_suffix('.json').read_text(encoding='utf8'))
        if report['sha256']!=firma(out): raise ValueError('Grafo modificado: '+str(out))
        return report
    xyz,frame=marco(xyz)
    valid=np.isfinite(xyz).all(1)
    xyz=xyz[valid]; masks=masks[valid]
    # Duplicados exactos se eliminan solo en fuentes de entrenamiento, nunca Lurín.
    _,idx=np.unique(xyz,axis=0,return_index=True); idx=np.sort(idx); xyz=xyz[idx]; masks=masks[idx]
    print(out.stem,'geometría...',flush=True)
    local,_,good=calcular_geometria(xyz)
    if not good.all(): raise ValueError('Vecindarios degenerados en '+str(out))
    height,_=elevacion_z_minmax(xyz); geof=np.column_stack([local,height])
    src,dst,dist=grafo_vecinos(xyz)
    print(out.stem,'partición...',flush=True)
    group,_,_=resolver_particion(geof,src,dst,dist,verbose=False)
    packed=muestrear_grupos(xyz,geof,group)
    print(out.stem,'grafo...',flush=True)
    u,v=aristas_voronoi(xyz,group); graph=agregar_conexiones(xyz,group,u,v)
    hist=np.zeros((len(packed['puntos']),len(MASKS)),dtype='i8')
    for k,mask in enumerate(MASKS): hist[:,k]=np.bincount(group[masks==mask],minlength=len(hist))
    np.savez_compressed(out,puntos=packed['puntos'],centros=packed['centros'].astype('f4'),
        embedding_habilitado=packed['embedding_habilitado'],
        aristas=np.stack([graph['origen'],graph['destino']]).astype('i8'),
        atributos_aristas=graph['atributos'],histogramas=hist,
        xyz=xyz,grupo_por_punto=group,mascara_clase=masks)
    report=dict(metadata,sha256=firma(out),puntos=len(xyz),superpuntos=len(hist),
        aristas=len(graph['origen']),histograma_mascaras=hist.sum(0).tolist(),marco=frame,
        perfil_spg='mismas funciones M1: geof45, partición k10 lambda.03, Delaunay13, 128x8',
        original_sin_modificar=True)
    guardar(out.with_suffix('.json'),report)
    print(out.stem,len(xyz),'puntos',len(hist),'superpuntos',flush=True)
    return report


def geometria_densa(d,report,support):
    """Mismos puntos/grupos, vecinos consultados en la fuente original densa."""
    rotation=np.asarray(report['marco']['rotacion']);center=np.asarray(report['marco']['centro'])
    query=d['xyz'].astype('f8')@rotation.T+center;tree=cKDTree(support)
    geof=np.zeros((len(query),5),dtype='f4');height,_=elevacion_z_minmax(d['xyz']);geof[:,4]=height
    for start in range(0,len(query),4096):
        q=query[start:start+4096];_,idx=tree.query(q,k=46,workers=1)
        local=(support[idx[:,1:]].astype('f8')-q[:,None,:])@rotation
        values,valid=descriptores(np.concatenate([np.zeros((len(q),1,3)),local],axis=1))
        if not valid.all():raise ValueError('Vecindarios densos degenerados')
        geof[start:start+len(q),:4]=values
    packed=muestrear_grupos(d['xyz'],geof,d['grupo_por_punto'])
    np.testing.assert_allclose(packed['puntos'][:,:,:3],d['puntos'][:,:,:3],atol=1e-6)
    d['puntos']=packed['puntos']


def cargar(corpus=CORPUS):
    corpus=Path(corpus);manifest=json.loads((corpus/'manifiesto.json').read_text(encoding='utf8'))
    if manifest.get('taxonomia_version')!=2:raise ValueError('Se requiere corpus final con superestructura ambigua')
    data=[]
    from .utilidades import centros_relativos
    for row in manifest['puentes']:
        path=corpus/row['archivo']
        if firma(path)!=row['sha256']:raise ValueError('Datos modificados: '+row['id'])
        with np.load(path) as f:
            d={k:f[k] for k in ['puntos','centros','embedding_habilitado','aristas','atributos_aristas','histogramas']}
            d['atributos_nodo']=f['estructura']
        d['centros']=centros_relativos(d['centros']);d.update(id=row['id'],tipo=row['tipo'],division=row['division'])
        data.append(d)
    return manifest,data


def preparar(fuentes=None,corpus=CORPUS):
    spec=json.loads((BASE/'configs/corpus.json').read_text(encoding='utf8'));fuentes=Path(fuentes or spec['fuentes_por_defecto'])
    corpus=Path(corpus);folder=corpus/'grafos';folder.mkdir(parents=True,exist_ok=True)
    manifestpath=corpus/'manifiesto.json';previous=json.loads(manifestpath.read_text(encoding='utf8')) if manifestpath.exists() else {}
    previous={r['id']:r for r in previous.get('puentes',[])};reports=[]
    def existente(identity):
        if identity not in previous or not (folder/(identity+'.npz')).exists():return None
        row=previous[identity]
        if row.get('taxonomia_version')!=2 or firma(corpus/row['archivo'])!=row['sha256']:raise ValueError('Corpus incompatible o modificado')
        return row
    for source in spec['reales']:
        identity=source['id'];cached=existente(identity)
        if cached is not None:reports.append(cached);continue
        if identity.startswith('semanticbridge_'):
            path=fuentes/'datasets_reales/semanticbridge/mls_data.zip'
            with zipfile.ZipFile(path) as z:
                xyz,labels,n=read_semantic(z,source['miembro'],spec['max_puntos_reales'],source['semilla'])
                with z.open(source['miembro']) as f:
                    while f.readline().decode('ascii').strip()!='end_header':pass
                    raw=f.read(n*12)
                    if len(raw)!=n*12:raise ValueError('Fuente densa truncada')
                support=np.frombuffer(raw,dtype='<f4').reshape(n,3)
            masks=semanticbridge(labels)
        else:
            name=identity.split('_',1)[1].capitalize()+' Bridge_annotation.ply'
            path=BASE/'datos/fuentes'/name if identity=='uav_seokdap' else fuentes/'datasets_reales/uav_lidar'/name
            if not path.exists() and identity=='uav_seokdap':
                path.parent.mkdir(parents=True,exist_ok=True)
                urllib.request.urlretrieve('https://zenodo.org/api/records/18531104/files/Seokdap%20Bridge_annotation.ply/content',path)
                if hashlib.md5(path.read_bytes()).hexdigest()!='67bb56b06e0bbb629994411aa8ecea3c':raise ValueError('Descarga Seokdap incorrecta')
            xyz,labels,n=read_uav(path,spec['max_puntos_reales'],source['semilla'])
            vertex=PlyData.read(str(path)).elements[0].data
            support=np.column_stack([vertex[k] for k in ['x','y','z']]);masks=uav(labels)
        target=folder/(identity+'.npz')
        report=preparar_grafo(xyz,masks,target,dict(source,tipo='real',fuente=SOURCES['semanticbridge'] if identity.startswith('semanticbridge_') else SOURCES['uav'],originales=n,taxonomia_version=2))
        with np.load(target) as f:d={k:f[k] for k in f.files}
        geometria_densa(d,report,support);d['estructura'],_=estructura(d['xyz'],d['grupo_por_punto'])
        np.savez_compressed(target,**d);report.update(archivo='grafos/'+target.name,sha256=firma(target))
        guardar(target.with_suffix('.json'),report);reports.append(report);del support
    synth=spec['sinteticos'];total=synth['entrenamiento']+synth['validacion']
    for i in range(total):
        identity=f'contextual_{i:03d}';cached=existente(identity)
        if cached is not None:reports.append(cached);continue
        xyz,labels,params=generar(synth['semillas_inicio']+i);target=folder/(identity+'.npz')
        report=preparar_grafo(xyz,1<<labels,target,dict(id=identity,tipo='sintetico',division='train' if i<synth['entrenamiento'] else 'val',parametros=params,fuentes=SOURCES,taxonomia_version=2))
        with np.load(target) as f:d={k:f[k] for k in f.files}
        d['estructura'],_=estructura(d['xyz'],d['grupo_por_punto']);np.savez_compressed(target,**d)
        report.update(archivo='grafos/'+target.name,sha256=firma(target));guardar(target.with_suffix('.json'),report);reports.append(report)
    guardar(manifestpath,dict(clases=CLASSES,mascaras=MASKS.tolist(),puentes=reports,taxonomia_version=2,superestructura='union viga/tablero',Lurin_en_entrenamiento=False))
    print('Corpus final verificado:',len(reports),'escenas.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fuentes',type=Path);p.add_argument('--datos',type=Path,default=CORPUS)
    a=p.parse_args();preparar(a.fuentes,a.datos)


if __name__=='__main__':main()
