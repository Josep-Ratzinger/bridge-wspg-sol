"""Perfil de cabezal inferido de restos laterales, sin rellenar el espacio superior."""
import numpy as np
from scipy.spatial import ConvexHull
from M1.AdaPoinTr.src.geometrias_sinteticas import contains


def fit_cross_profile(observed,columns,old_bottom,old_top):
    slope,intercept=np.polyfit([c['s'] for c in columns],[c['d'] for c in columns],1)
    profiles=[]
    for col in (columns[0],columns[-1]):
        p=observed[(abs(observed[:,0]-col['s'])<.35)&(abs(observed[:,1]-col['d'])<2.2)&(observed[:,2]>=old_bottom-.15)&(observed[:,2]<=old_top)]
        profiles.append(np.column_stack([p[:,1]-(slope*p[:,0]+intercept),p[:,2]]))
    p=np.vstack(profiles);edges=np.arange(old_bottom,old_top+.1,.1);density=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        band=p[(p[:,1]>=lo)&(p[:,1]<hi)];inner=int((abs(band[:,0])<.85).sum());outer=int(((abs(band[:,0])>1.05)&(abs(band[:,0])<1.95)).sum());density.append({'z_low':float(lo),'z_high':float(hi),'inner_points':inner,'outer_points':outer,'ratio':outer/max(inner,1)})
    top=None
    for row in density:
        if row['inner_points']>=30 and row['ratio']>.30:
            top=row['z_low'];break
    if top is None or top<old_bottom+.5:raise ValueError('No se distingue el limite entre cabezal y zona superior')
    levels=np.linspace(old_bottom,top,7);centers=[];depths=[]
    for z in levels:
        band=p[(abs(p[:,1]-z)<.09)&(p[:,1]<=top-.025)&(abs(p[:,0])<1.2)]
        if len(band)<20:band=p[(abs(p[:,1]-z)<.16)&(p[:,1]<=top)&(abs(p[:,0])<1.2)]
        if len(band)<10:raise ValueError('Perfil con pocos puntos')
        left,right=np.quantile(band[:,0],[.03,.97]);centers.append(float((left+right)/2));depths.append(float(right-left))
    required=max(2*(abs(c['d']-(slope*c['s']+intercept))+c['radius']) for c in columns)
    depths=np.maximum.accumulate(np.maximum(depths,required));centers=np.linspace(0.,float(np.median(centers[-3:])),len(centers))
    # La base toca los cilindros existentes; la parte alta sigue el contorno observado.
    return {'levels_z':levels.tolist(),'relative_centers_d':centers.tolist(),'depths':depths.tolist(),'row_center_slope':float(slope),'row_center_intercept':float(intercept),'old_top_z':old_top,'new_top_z':float(top),'bottom_z':old_bottom,'density_transition':density,'minimum_depth_for_existing_column_contacts':float(required),'source':'perfiles observados cerca de los dos apoyos extremos; umbral de extension lateral para separar cabezal de elementos superiores'}


def profiled_beam(longitudinal,profile):
    (lb,z0),(rb,_),(rt,_),(lt,_)=longitudinal;top=profile['new_top_z'];vertices=[]
    for z,offset,depth in zip(profile['levels_z'],profile['relative_centers_d'],profile['depths']):
        t=(z-z0)/(top-z0);left=lb+t*(lt-lb);right=rb+t*(rt-rb)
        for s,sign in ((left,-1),(right,-1),(right,1),(left,1)):
            d=profile['row_center_slope']*s+profile['row_center_intercept']+offset+sign*depth/2;vertices.append([s,d,z])
    vertices=np.array(vertices);hull=ConvexHull(vertices)
    return {'kind':'convex_profile','vertices':vertices,'faces':hull.simplices,'planes':hull.equations,'tag':1}


def inside(part,points):
    if part.get('kind')!='convex_profile':return contains(part,points)
    keep=np.ones(len(points),bool)
    for a,b,c,d in part['planes']:keep&=points[:,0]*a+points[:,1]*b+points[:,2]*c+d<=1e-7
    return keep


def sample_union(parts,n,seed=241137):
    rng=np.random.default_rng(seed);triangles=[];owners=[]
    for i,part in enumerate(parts):
        tri=part['vertices'][part['faces']];triangles.extend(tri);owners.extend([i]*len(tri))
    triangles=np.array(triangles);owners=np.array(owners);area=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1);area/=area.sum();collected=[];tags=[];count=0
    for _ in range(16):
        size=max(1024,int((n-count)*1.5));ids=rng.choice(len(triangles),size,p=area);tri=triangles[ids];uv=rng.random((size,2));u=np.sqrt(uv[:,0]);v=uv[:,1];p=(1-u[:,None])*tri[:,0]+(u*(1-v))[:,None]*tri[:,1]+(u*v)[:,None]*tri[:,2];owner=owners[ids];keep=np.ones(size,bool)
        for i,part in enumerate(parts):
            candidate=(owner!=i)&keep;keep[candidate]&=~inside(part,p[candidate])
        collected.append(p[keep]);tags.append(np.array([parts[i]['tag'] for i in owner[keep]],dtype='i2'));count+=int(keep.sum())
        if count>=n:break
    if count<n:raise ValueError('Muestreo insuficiente')
    return np.concatenate(collected)[:n],np.concatenate(tags)[:n]
