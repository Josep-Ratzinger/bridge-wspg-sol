"""Controles paramétricos nuevos de selección automática, sin entrenar pesos.

Referencias conocidas solo para medir. No pasan a build_geometry. Incluye
variación de secciones, cabezales, terminaciones, orientación y escala.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from ..src.diagnostico import MODULE,sha256
from ..src.geometrias_sinteticas import prism
from ..src.seleccion_formas import section_polygon,loft_beam_parts
from ..src.completado_automatico import support_context,build_geometry,merge_surface,SEED
from ..src.perfiles_viga import sample_union
from ..src.entrenamiento import cloud_metrics


def scene(index):
    rng=np.random.default_rng(241160+index);shape=('cilindrica','rectangular','eliptica','octagonal')[index%4]
    beam=('rectangular','trapezoidal','hombro','I')[index//4];n=int(rng.integers(4,8));spacing=float(rng.uniform(2.1,3.2));half=(n-1)*spacing/2
    levels=np.linspace(4.5,5.9,13);t=(levels-levels[0])/np.ptp(levels)
    if beam=='rectangular':depth=np.full(len(t),1.7)
    elif beam=='trapezoidal':depth=1.3+.9*t
    elif beam=='hombro':depth=1.3+.3*t+1.5*np.maximum(t-.5,0)
    else:depth=np.where((t<=.2)|(t>=.8),2.,.8)
    end=('recto','inclinado','curvo')[index%3];out=np.full(len(t),.65) if end=='recto' else .4+.5*(t if end=='inclinado' else t*t)
    profile={'levels_z':levels.tolist(),'depths':depth.tolist(),'left_s':(-half-out).tolist(),'right_s':(half+out).tolist(),
             'relative_centers_d':[0.]*len(t),'row_center_slope':0.,'row_center_intercept':0.}
    parts=loft_beam_parts(profile);centers=(np.arange(n)-(n-1)/2)*spacing
    params=[0,0,.52] if shape=='cilindrica' else [0,0,.65 if shape=='eliptica' else .52,.34 if shape=='eliptica' else .52,.12]
    poly=section_polygon(shape,params)
    for i,s in enumerate(centers):parts.append(prism(poly,4.5,2,[s,0,2.25],i+2))
    full,tags=sample_union(parts,90000,241170+index);keep=np.zeros(len(full),bool)
    for i in range(n):
        fraction=1. if i in (0,n-1) else float(rng.choice([.1,.25,.5,.75]))
        keep|=(tags==i+2)&(full[:,2]<=4.5*fraction)
    keep|=(tags==1)&(abs(full[:,0])>half*.7);keep&=rng.random(len(full))>.25
    for _ in range(2):
        ids=np.flatnonzero(keep);anchor=full[int(rng.choice(ids))];distance=np.linalg.norm(full[ids]-anchor,axis=1)
        keep[ids[np.argsort(distance)[:int(len(ids)*.04)]]]=False
    theta=float(rng.uniform(-np.pi,np.pi));rotation=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1.]])
    scale=float(rng.uniform(.7,1.6));translation=rng.uniform(-5,5,3)
    world=full@rotation*scale+translation
    return world[keep],world,(6.1*scale+translation[2]),{'shape':shape,'beam':beam,'end':end,'count':n,'scale':scale,'rotation':theta}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--salida',default='formas_geometricas');args=parser.parse_args()
    if Path(args.salida).name!=args.salida:raise ValueError('Salida debe ser nombre de carpeta')
    out=MODULE/'evaluacion'/args.salida
    if out.exists():raise ValueError('No sobrescribir controles')
    rows=[]
    for i in range(16):
        observed,reference,plane,truth=scene(i);factor=(plane-np.quantile(observed[:,2],.0005))/6.3
        row={'case':i,'reference_only_for_metrics':truth,'stages':{}}
        try:
            ctx=support_context(observed/factor,plane/factor,SEED);xy=observed[:,:2]/factor-ctx['origin_xy'];local=np.c_[xy@ctx['axis_xy'],xy@ctx['normal_xy'],observed[:,2]/factor]
            tolerance=.01*float(np.ptp(observed,axis=0).max());row['partial']=cloud_metrics(observed,reference,tolerance)
            for name,builder in [('metodo_aprobado',build_geometry)]:
                try:
                    parts,fit=builder(ctx,local);surface,tags=sample_union(parts,40000,SEED);new,_,_,_=merge_surface(ctx['local'],surface,tags,np.empty((0,3)))
                    xy=ctx['origin_xy']+new[:,0,None]*ctx['axis_xy']+new[:,1,None]*ctx['normal_xy'];new=np.c_[xy,new[:,2]]*factor
                    hidden=reference[cKDTree(observed).query(reference)[0]>tolerance]
                    stage={'success':True,'combined':cloud_metrics(np.vstack([observed,new]),reference,tolerance),
                           'missing_recovery':float((cKDTree(new).query(hidden)[0]<=tolerance).mean()) if len(new) and len(hidden) else 0.,
                           'sections':[c['section'] for c in fit['columns']],'detected_columns':len(fit['columns']),
                           'beam_selection':fit['beam_profile'].get('shape_selection'),
                           'added_precision':float((cKDTree(reference).query(new)[0]<=tolerance).mean()) if len(new) else None}
                except ValueError as e:stage={'success':False,'error':str(e)}
                row['stages'][name]=stage
        except ValueError as e:row['localization_error']=str(e)
        rows.append(row);print(json.dumps({'case':i,'truth':truth,'results':{k:{f:v[f] for f in ('success','combined','sections') if f in v} for k,v in row['stages'].items()}}),flush=True)
    out.mkdir(parents=True);report={'method':'control de la etapa geométrica del método aprobado; AdaPoinTr no modificado ni reentrenado en este control',
                        'new_reference_scenes':True,'real_missing_geometry_validated':False,'cases':rows,
                        'selection_source_sha256':sha256(MODULE/'src/seleccion_formas.py'),'completion_source_sha256':sha256(MODULE/'src/completado_automatico.py')}
    (out/'evaluacion.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8')


if __name__=='__main__':main()
