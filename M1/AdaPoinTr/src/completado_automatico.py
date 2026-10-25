"""Completado híbrido reproducible desde preparación, sin parámetros de Lurín.

AdaPoinTr propone puntos. El ajuste geométrico estima un conjunto de apoyos
alineados con cabezal a partir de observaciones, conserva los vacíos y rechaza
propuestas neuronales incompatibles. La reconstrucción global proviene del
ajuste, no se atribuye al aprendizaje neuronal.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from .diagnostico import MODULE, PROJECT, sha256
from .geometrias_sinteticas import prism
from .localizacion_apoyos import support_context
from .ajuste_estructura import fit_structure
from .secciones_columnas import fit_section
from .perfiles_viga import fit_cross_profile, profiled_beam, sample_union
from .exportacion import global_points
from .exportacion import DTYPE, write_bridge
from .seleccion_formas import augment_section,section_polygon,observed_beam_forms,loft_beam_parts,beam_underside,borrow_section_model

SEED=241143


def merge_density_peaks(context):
    """Fusiona dos máximos del mismo fuste si no hay valle de separación.

    Una sección circular ancha puede producir dos crestas laterales; las
    columnas distintas deben conservar el valle de baja densidad entre ellas.
    """
    peaks=context['density_peaks_s'];axis=context['density_hist_s'];hist=context['density_hist']
    if len(peaks)<2:return peaks
    ids=np.array([int(np.argmin(abs(axis-p))) for p in peaks]);groups=[[int(ids[0])]]
    for left,right in zip(ids[:-1],ids[1:]):
        valley=float(hist[left:right+1].min());should_merge=valley>.25*min(hist[left],hist[right])
        if should_merge:groups[-1].append(int(right))
        else:groups.append([int(right)])
    return np.array([np.average(axis[g],weights=hist[g]) for g in groups])


def choose_shared_section(sections):
    """La forma sencilla basta si la rectangular no mejora >25% el error.

    No confirma la clase real. Solo resuelve una hipótesis para superficies
    repetidas, con preferencia por evidencia individual suficientemente cubierta.
    """
    confident=[s['status'] for s in sections if s['status']!='incierta']
    if confident:
        return max(sorted(set(confident)),key=confident.count), 'observaciones con cobertura suficiente'
    ratios=[s['models']['cilindrica']['validation_score']/max(s['models']['rectangular']['validation_score'],1e-9)
            for s in sections if all(k in s['models'] for k in ('cilindrica','rectangular'))]
    if not ratios:raise ValueError('No hay evidencia de sección suficiente para reconstruir apoyos')
    return ('cilindrica' if np.median(ratios)<=1.25 else 'rectangular'), 'hipótesis compartida incierta; regla de parsimonia, no clase confirmada'


def build_geometry(context, full_local, seed=SEED):
    context=dict(context);context['density_peaks_s']=merge_density_peaks(context)
    if len(context['density_peaks_s'])<2:raise ValueError('No se distinguen al menos dos apoyos separados')
    local=context['local'];_,fit=fit_structure(local,context['density_peaks_s'],context['plane'],seed)
    # Una diagonal espuria no puede extender el cabezal fuera del contexto visto.
    # Limitar con evidencia evita extrapolaciones de varias luces desde un talud.
    polygon=np.array(fit['beam_polygon_s_z'],dtype=float)
    lo,hi=np.quantile(local[:,0],[.0005,.9995]);span=hi-lo
    polygon[3,0]=max(polygon[3,0],lo-.01*span);polygon[2,0]=min(polygon[2,0],hi+.01*span)
    polygon[0,0]=np.clip(polygon[0,0],polygon[3,0]+.05,(lo+hi)/2-.05)
    polygon[1,0]=np.clip(polygon[1,0],(lo+hi)/2+.05,polygon[2,0]-.05)
    fit['beam_polygon_s_z']=polygon.tolist()
    sections=[];stems=[]
    spacing=float(np.median(np.diff(context['density_peaks_s'])))
    for i,col in enumerate(fit['columns']):
        floor=col['fitted_bottom'];upper=min(fit['beam_bottom_z']-.2,floor+max(.6,.85*col['observed_height']))
        stem=local[(abs(local[:,0]-col['peak'])<min(.8,spacing*.4))&(local[:,2]>floor+.2)&(local[:,2]<upper)]
        base=fit_section(stem,[col['s'],col['d']],col['width_shared'],col['depth_shared'],seed+i)
        stems.append(stem);sections.append(augment_section(stem,base,col['width_shared'],col['depth_shared'],seed+i))
    shared,reason=choose_shared_section(sections)
    radii=[s['models']['cilindrica']['parameters'][2] for s in sections if 'cilindrica' in s['models']]
    common_radius=float(np.median(radii)) if radii else fit['columns'][0]['width_shared']/2
    columns=[];polygons=[]
    for i,(col,section,stem) in enumerate(zip(fit['columns'],sections,stems)):
        kind=section['status'] if section['status']!='incierta' else shared
        model=section['models'].get(kind);center=np.array([col['s'],col['d']]);status=section['status']
        if kind=='cilindrica':
            radius=common_radius
            if model and model['validation_score']<=.1*max(col['width_shared'],col['depth_shared']) and .75*radius<=model['parameters'][2]<=1.3*radius:
                radius=.5*(radius+model['parameters'][2])
            radius=min(radius,.4*spacing);initial=np.array(model['parameters'][:2]) if model else center
            lower=center-.6;upper=center+.6;initial=np.clip(initial,lower+1e-7,upper-1e-7)
            if len(stem)>30:
                _,ids=np.unique(np.floor(stem[:,:2]/.025).astype('i8'),axis=0,return_index=True);xy=stem[ids,:2]
                center=least_squares(lambda q:np.linalg.norm(xy-q,axis=1)-radius,initial,bounds=(lower,upper),loss='soft_l1',f_scale=.025,max_nfev=150).x
            angle=np.linspace(0,2*np.pi,96,endpoint=False);poly=np.c_[np.cos(angle),np.sin(angle)]*radius
        elif kind=='rectangular':
            q=np.array(model['parameters']) if model else np.r_[center,col['width_shared']/2,col['depth_shared']/2,0.]
            center=q[:2];a,b,theta=q[2:];a=min(a,.4*spacing)
            rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
            poly=np.array([[-a,-b],[a,-b],[a,b],[-a,b]])@rot.T;radius=float(abs(poly[:,1]).max())
        else:
            if model is None:model=borrow_section_model(sections,kind,center)
            if model is None:raise ValueError('Sección sin parámetros observados ni vecinos compatibles')
            q=np.array(model['parameters']);center=q[:2];poly=section_polygon(kind,q)
            extent=float(abs(poly[:,0]).max())
            if extent>.4*spacing:poly*=.4*spacing/extent
            radius=float(abs(poly[:,1]).max())
        columns.append({'support_id':i,'s':float(center[0]),'d':float(center[1]),'radius':float(radius),
                        'section':kind,'section_confidence':status,'fitted_bottom':col['fitted_bottom'],
                        'observed_points_for_fit':len(stem),'polygon_xy':poly.tolist(),
                        'parameter_source':model.get('parameter_source','ajuste observado y regularización compartida') if model else 'regularización con apoyos observados',
                        'borrowed_from_supports':model.get('borrowed_from_supports',[]) if model else []});polygons.append(poly)
    profile_fallback=None
    try:profile=fit_cross_profile(full_local,columns,fit['beam_bottom_z'],fit['beam_top_z'])
    except ValueError as error:
        # No se inventa un perfil escalonado sin evidencia: aproximación prismática explícita.
        profile_fallback=str(error);z0,z1=fit['beam_bottom_z'],fit['beam_top_z']
        slope,intercept=np.polyfit([c['s'] for c in columns],[c['d'] for c in columns],1)
        levels=np.linspace(z0,z1,7);widths=[];minimum=2*max(c['radius'] for c in columns)
        outer=(abs(full_local[:,0]-columns[0]['s'])<.4)|(abs(full_local[:,0]-columns[-1]['s'])<.4)
        for z in levels:
            band=full_local[outer&(abs(full_local[:,2]-z)<max(.12,(z1-z0)*.15))]
            if len(band)>=20:
                residual=abs(band[:,1]-(slope*band[:,0]+intercept));widths.append(max(minimum,2*float(np.quantile(residual,.95))))
            else:widths.append(max(minimum,fit['beam_depth']))
        widths=np.maximum.accumulate(np.minimum(widths,minimum*3.))
        profile={'levels_z':levels.tolist(),'relative_centers_d':[0.]*7,'depths':widths.tolist(),
                 'row_center_slope':float(slope),'row_center_intercept':float(intercept),'new_top_z':z1,'bottom_z':z0,
                 'source':'perfil simétrico aproximado con cuantiles por altura; sin transición superior distinguible'}
    profile=observed_beam_forms(full_local,columns,profile,fit['beam_polygon_s_z'])
    parts=loft_beam_parts(profile)
    lb,rb,rt,lt=[v[0] for v in fit['beam_polygon_s_z']];z0,z1=fit['beam_bottom_z'],profile['new_top_z']
    def underside(s):
        return beam_underside(s,profile)
    for i,(col,poly) in enumerate(zip(columns,polygons)):
        extent=float(abs(poly[:,0]).max());top=min(z1,max(underside(col['s']-extent),underside(col['s']+extent)));floor=col['fitted_bottom']
        if not floor<top<=z1:raise ValueError('Unión columna-viga incompatible con observaciones')
        col['fitted_top']=float(top);parts.append(prism(poly,top-floor,2,[col['s'],col['d'],(floor+top)/2],i+2))
    return parts,{'columns':columns,'shared_section_hypothesis':shared,'shared_section_reason':reason,
                  'section_comparisons':sections,'beam_profile':profile,'beam_longitudinal_polygon':fit['beam_polygon_s_z'],
                  'beam_profile_fallback':profile_fallback,'beam_bottom_z':z0,'beam_top_z':z1,
                  'scope':'un conjunto dominante de apoyos alineados y cabezal; no reconstruye toda tipología de puente'}


def neural_candidates(context,checkpoint,seed=SEED):
    import torch
    from .modelo import load_official
    torch.set_num_threads(4);torch.manual_seed(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,_=load_official('cuda');saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(saved['state_dict'],strict=True);model.eval();rng=np.random.default_rng(seed);predictions=[];samples=[]
    with torch.inference_mode():
        for _ in range(2):
            ids=rng.choice(len(context['normalized']),4096,replace=False);samples.append(context['ids'][ids])
            pred=model(torch.from_numpy(context['normalized'][ids][None]).cuda())[1][0].cpu().numpy()
            if not np.isfinite(pred).all():raise ValueError('AdaPoinTr produjo valores no finitos')
            predictions.append(pred)
    first,second=predictions
    stable=np.vstack([first[cKDTree(second).query(first)[0]<=.01],second[cKDTree(first).query(second)[0]<=.01]])
    return stable*context['scale']+context['center'],np.stack(samples)


def merge_surface(observed,surface,tags,neural,seed=SEED):
    """Propuestas neuronales cerca de superficie: proyección y trazabilidad separadas."""
    accepted=np.empty((0,3));accepted_tags=np.empty(0,dtype='i2')
    if len(neural):
        dist,nearest=cKDTree(surface).query(neural);mask=dist<=.08
        accepted=surface[nearest[mask]];accepted_tags=tags[nearest[mask]]
    # La geometría rellena superficies; AdaPoinTr aporta candidatos regularizados.
    points=np.vstack([accepted,surface]);owners=np.r_[accepted_tags,tags];source=np.r_[np.ones(len(accepted),dtype='u1'),np.full(len(surface),2,dtype='u1')]
    keep=cKDTree(observed).query(points)[0]>.035;points=points[keep];owners=owners[keep];source=source[keep]
    _,ids=np.unique(np.floor(points/.025).astype('i8'),axis=0,return_index=True);ids=np.sort(ids)
    return points[ids],owners[ids],source[ids],len(accepted)


def verify_output(output,xyz,rgb,reduction,new_count):
    with output.open('rb') as f:
        while True:
            line=f.readline()
            if not line or f.tell()>10000:raise ValueError('Cabecera PLY inválida')
            if line.strip()==b'end_header':break
        offset=f.tell()
    count=len(xyz)+new_count
    if output.stat().st_size!=offset+count*DTYPE.itemsize:raise ValueError('Tamaño PLY inválido')
    cloud=np.memmap(output,mode='r',dtype=DTYPE,offset=offset,shape=(count,))
    for j,k in enumerate(('x','y','z')):
        if not np.array_equal(cloud[k][:len(xyz)],xyz[:,j]) or not np.isfinite(cloud[k]).all():raise ValueError('Coordenadas alteradas')
    for j,k in enumerate(('red','green','blue')):
        if not np.array_equal(cloud[k][:len(xyz)],rgb[:,j]):raise ValueError('Colores alterados')
    if not np.array_equal(cloud['reduccion_id'][:len(xyz)],reduction) or not np.array_equal(cloud['punto_id'][:len(xyz)],np.arange(len(xyz))):raise ValueError('Identidades alteradas')
    if cloud['origen'][:len(xyz)].any() or not (cloud['origen'][len(xyz):]==1).all():raise ValueError('Procedencia incorrecta')
    del cloud


def run(input_npz,output_dir,checkpoint,seed=SEED):
    input_npz=Path(input_npz).resolve();output_dir=Path(output_dir).resolve();checkpoint=Path(checkpoint).resolve()
    if output_dir.exists():raise ValueError('La salida ya existe. Usa --salida con una carpeta nueva; no se sobrescriben resultados.')
    before=sha256(input_npz);checkpoint_hash=sha256(checkpoint)
    training_report=checkpoint.parent/'resultado.json'
    if training_report.exists():
        expected=json.loads(training_report.read_text())['checkpoint_sha256']
        if checkpoint.name=='mejor.pt' and checkpoint_hash!=expected:raise ValueError('Los pesos no coinciden con el entrenamiento registrado')
    with np.load(input_npz,allow_pickle=False) as data:
        xyz=data['xyz'];rgb=data['rgb'];reduction=data['indices_reduccion'];meta=json.loads(str(data['metadatos']))
    if xyz.ndim!=2 or xyz.shape[1]!=3 or len(xyz)<4096 or rgb.shape!=xyz.shape or len(reduction)!=len(xyz) or not np.isfinite(xyz).all():raise ValueError('Preparación incompatible')
    if xyz.dtype!=np.float32:raise ValueError('Se espera la preparación oficial float32 para conservar coordenadas exactas')
    plane=float(meta['orientacion']['plano_z_modo'])
    # Escala canónica inferida de la entrada, no de una referencia completa.
    depth=float(plane-np.quantile(xyz[:,2],.0005));factor=depth/6.3
    if not np.isfinite(factor) or factor<=0:raise ValueError('No se distingue altura bajo el tablero')
    canonical=xyz.astype('f8')/factor;context=support_context(canonical,plane/factor,seed)
    xy=canonical[:,:2]-context['origin_xy'];full_local=np.column_stack([xy@context['axis_xy'],xy@context['normal_xy'],canonical[:,2]])
    print('Apoyos localizados. Ajustando secciones y perfil observado...',flush=True)
    parts,fit=build_geometry(context,full_local,seed)
    print('AdaPoinTr: dos muestras del conjunto; después, ajuste geométrico.',flush=True)
    neural,input_ids=neural_candidates(context,checkpoint,seed)
    surface,tags=sample_union(parts,150000,seed)
    estimated,owners,sources,accepted=merge_surface(context['local'],surface,tags,neural,seed)
    if not len(estimated):raise ValueError('No se generaron estimaciones compatibles')
    # Ningún punto estimado bajo el cabezal puede formar paredes entre apoyos.
    below=estimated[:,2]<fit['beam_bottom_z']-1e-6
    if (below&(owners==1)).any():raise ValueError('La viga invade huecos libres')
    world=(global_points(estimated,context).astype('f8')*factor).astype('f4')
    shape_codes={'rectangular':1,'cilindrica':2,'eliptica':4,'octagonal':5}
    labels=np.array([3 if t==1 else shape_codes[fit['columns'][int(t)-2]['section']] for t in owners],dtype='u1')
    supports=np.where(owners==1,-1,owners-2).astype('i2')
    output_dir.mkdir(parents=True);output=output_dir/'puente_completado.ply'
    write_bridge(output,xyz,rgb,reduction,world,labels,labels,supports);verify_output(output,xyz,rgb,reduction,len(world))
    np.savez_compressed(output_dir/'estimados.npz',xyz=world,local_canonico=estimated,origen=np.ones(len(world),dtype='u1'),
                        fuente_estimacion=sources,apoyo_id=supports,network_input_global_ids=input_ids)
    if sha256(input_npz)!=before:raise ValueError('La preparación original cambió durante el proceso')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(14,7),sharex=True,sharey=True);observed=context['local']
    for ax in axes:ax.scatter(observed[:,0]*factor,observed[:,2]*factor,s=.2,c='#1984c6');ax.set_aspect('equal');ax.set_ylabel('Z, unidades originales')
    axes[0].set_title('Entrada observada');axes[1].scatter(estimated[:,0]*factor,estimated[:,2]*factor,s=.25,c='#f08d23');axes[1].set_title('Completado híbrido automático: naranja estimado');axes[1].set_xlabel('Eje del conjunto, unidades originales');fig.tight_layout();fig.savefig(output_dir/'comparacion.png',dpi=140);plt.close(fig)
    record={'method':'AdaPoinTr + ajuste geométrico automático; la forma global se obtiene del ajuste observado, no de AdaPoinTr solo',
            'input_npz':str(input_npz),'input_sha256':before,'input_original_unchanged':True,'checkpoint':str(checkpoint),'checkpoint_sha256':checkpoint_hash,
            'output_sha256':sha256(output),'original_points_preserved':len(xyz),'estimated_points_added':len(world),'seed':seed,
            'canonical_factor':factor,'original_coordinates_colors_ids_verified':True,'manual_bridge_context_used':False,
            'previous_approximations_loaded':False,'real_missing_surface_reference_available':False,
            'units':'las originales, sin afirmar metros certificados','neural_candidates_near_fitted_surface':accepted,
            'neural_projected_points_after_merge':int((sources==1).sum()),'geometric_surface_points_after_merge':int((sources==2).sum()),
            'geometry':fit,'scope':fit['scope'],'completion_scope':'solo conjunto dominante detectado; resto del puente conservado',
            'shape_code_map':{**shape_codes,'viga':3},'shape_selection_source_sha256':sha256(MODULE/'src/seleccion_formas.py'),
            'source_sha256':sha256(Path(__file__)),
            'helper_sources_sha256':{name:sha256(MODULE/'src'/name) for name in ('localizacion_apoyos.py','ajuste_estructura.py','secciones_columnas.py','perfiles_viga.py','exportacion.py')},
            'provenance':'origen PLY 0 observado / 1 estimado; estimados.npz fuente 1 propuesta neuronal proyectada / 2 superficie geométrica; formas propuestas no son etiquetas semánticas'}
    (output_dir/'ejecucion.json').write_text(json.dumps(record,indent=2,ensure_ascii=False),encoding='utf8')
    print(f'Originales conservados: {len(xyz):,}. Estimados: {len(world):,}. Apoyos: {len(fit["columns"])}.',flush=True)
    print('Salida: '+str(output),flush=True)
    return record


def main(argv=None):
    cfg=json.loads((MODULE/'configs/puentes.json').read_text())
    parser=argparse.ArgumentParser(description='Completado híbrido automático desde la salida de limpieza; requiere entorno nativo AdaPoinTr y GPU.')
    parser.add_argument('--entrada',type=Path,default=PROJECT/cfg['input_npz'])
    parser.add_argument('--salida',type=Path,default=MODULE/'resultados/completado')
    parser.add_argument('--checkpoint',type=Path,default=MODULE/'modelos/completado_contextual/mejor.pt')
    parser.add_argument('--semilla',type=int,default=SEED)
    args=parser.parse_args(argv);run(args.entrada,args.salida,args.checkpoint,args.semilla)


if __name__=='__main__':main()
