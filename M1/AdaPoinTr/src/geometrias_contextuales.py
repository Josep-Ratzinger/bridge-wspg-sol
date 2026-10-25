"""Conjuntos propios de investigación; perfiles convexos y pérdidas simultáneas.

No son planos de Lurín ni cotas normativas. Cada identidad precede a sus máscaras.
El generador anterior queda intacto para conservar sus hashes y experimentos.
"""
import numpy as np
from .geometrias_sinteticas import prism
from .perfiles_viga import profiled_beam, sample_union


def recipe(split, index):
    if split not in ('train', 'val', 'test'):
        raise ValueError('Partición desconocida')
    rng = np.random.default_rng(np.random.SeedSequence([241140, ('train','val','test').index(split), index]))
    n = int(rng.integers(3, 11)); radius = float(rng.uniform(.3, .65))
    spacing = float(rng.uniform(2.1, 3.5)); h = float(rng.uniform(3., 7.))
    cap_h = float(rng.uniform(.6, 1.5)); half = (n-1)*spacing/2
    taper = float(rng.uniform(.2, 1.2)); top_half = half+radius+rng.uniform(.2,.7)
    bottom_half = max(half-spacing*.55, top_half-taper)
    shape = ('circular', 'rectangular', 'octagonal')[index % 3]
    depth = float(rng.uniform(radius*2.3, radius*3.3))
    profile = {'levels_z':[h,h+cap_h*.35,h+cap_h*.8,h+cap_h],
               'relative_centers_d':[0.,0.,0.,0.],
               'depths':[depth,depth*1.12,depth*1.5,depth*1.55],
               'row_center_slope':0.,'row_center_intercept':0.,'new_top_z':h+cap_h}
    polygon = [[-bottom_half,h],[bottom_half,h],[top_half,h+cap_h],[-top_half,h+cap_h]]
    columns=[]
    for i in range(n):
        s=(i-(n-1)/2)*spacing
        def underside(x):
            return h+cap_h*max(0.,abs(x)-bottom_half)/(top_half-bottom_half)
        columns.append({'s':s,'d':0.,'radius':radius,'top':max(underside(s-radius),underside(s+radius)),'bottom':0.})
    return {'base_id':f'contextual_{split}_{index:06d}', 'split':split,'shape':shape,
            'columns':columns,'profile':profile,'polygon':polygon,'radius':radius,
            'provenance':'variaciones paramétricas propias; sin medidas ni referencia completa de Lurín'}


def parts_for(p):
    parts=[profiled_beam(p['polygon'],p['profile'])]
    for i,col in enumerate(p['columns']):
        r=col['radius']
        if p['shape']=='rectangular':poly=np.array([[-r,-r*.85],[r,-r*.85],[r,r*.85],[-r,r*.85]])
        else:
            a=np.linspace(0,2*np.pi,64 if p['shape']=='circular' else 8,endpoint=False)
            poly=np.c_[np.cos(a),np.sin(a)]*r
        parts.append(prism(poly,col['top']-col['bottom'],2,[col['s'],col['d'],(col['top']+col['bottom'])/2],i+2))
    return parts


def pair(split, index, variant=0):
    p=recipe(split,index);rng=np.random.default_rng(np.random.SeedSequence([241141,index,variant,('train','val','test').index(split)]))
    full,tags=sample_union(parts_for(p),16384,int(rng.integers(2**31)))
    mode=variant%3;keep=np.zeros(len(full),bool);fractions=[]
    for i,col in enumerate(p['columns']):
        fraction=.1 if mode==2 else float(rng.choice([.1,.25,.5,.75,1.]))
        if i in (0,len(p['columns'])-1) and mode==0:fraction=1.
        fractions.append(fraction);keep|=(tags==i+2)&(full[:,2]<=fraction*col['top'])
    if mode==0:keep|=(tags==1)&(abs(full[:,0])>np.max(np.abs(p['polygon']),axis=0)[0]*.36)
    elif mode==1:keep|=(tags==1)&(rng.random(len(full))<.45)
    # Ocultación de una cara y huecos locales además de truncamiento vertical.
    keep&=(full[:,1]>-p['radius']*.45)
    for _ in range(2):
        ids=np.flatnonzero(keep)
        if len(ids)<64:raise ValueError('Pocos puntos para la máscara')
        anchor=full[int(rng.choice(ids))];dist=np.linalg.norm(full[ids]-anchor,axis=1)
        keep[ids[np.argsort(dist)[:max(1,int(len(ids)*.1))]]]=False
    remaining=np.flatnonzero(keep);ids=rng.choice(remaining,4096,replace=len(remaining)<4096)
    angle=rng.uniform(-np.pi,np.pi);rotation=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1.]])
    partial=full[ids]@rotation;partial+=rng.normal(0,.003,partial.shape)
    reference=full@rotation;center=(partial.min(0)+partial.max(0))/2;scale=float(np.ptp(partial,axis=0).max())
    return {'partial':((partial-center)/scale).astype('f4'),'complete':((reference-center)/scale).astype('f4'),
            'base_id':p['base_id'],'case_id':p['base_id']+f'_{variant}', 'shape':p['shape'],'variant':variant,
            'occlusion_fraction':float((~keep).mean()),'visible_height_fractions':fractions,
            'reference_source_indices':ids,'visible_mask':keep,'center':center,'scale':scale,'recipe':p}
