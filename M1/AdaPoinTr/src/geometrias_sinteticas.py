"""Superficies parametricas externas, pares y recetas reproducibles."""
import hashlib,json
import numpy as np

FAMILIES=('columna','cabezal','apoyo_multicolumna','viga','tablero')

def parameters(family,index,seed=241120):
    rng=np.random.default_rng(np.random.SeedSequence([seed,FAMILIES.index(family),index]))
    p={'family':family,'base_id':f'{family}_{index:06d}','seed':seed,'units':'m','source_ids':['yang2026'],'own_variations':[]}
    u=lambda a,b:float(rng.uniform(a,b))
    if family=='columna':
        p.update(shape=('rectangular','circular','rectangular_peru_anchor')[index%3],height=u(5,15))
        if p['shape']=='circular':p['radius']=u(1,2)
        elif p['shape']=='rectangular':p.update(width=u(.8,2),depth=u(.8,2))
        else:
            p.update(height=10.75*u(.85,1.15),width=3.8*u(.85,1.15),depth=1.8*u(.85,1.15),source_ids=['rondon2021'],own_variations=['perturbacion 0.85-1.15 de ancla peruana'])
    elif family in ('cabezal','apoyo_multicolumna'):
        p.update(shape=('rectangular','trapezoidal')[index%2],length=u(8,15),depth=u(1.2,2))
        p['height']=u(.7,2) if p['shape']=='rectangular' else u(.7,1)
        p['bottom_length']=p['length'] if p['shape']=='rectangular' else p['length']*u(.7,.9)
        p['own_variations']=['poligono longitudinal simplificado del cabezal']
        if family=='apoyo_multicolumna':
            p.update(column_height=u(5,15),column_width=u(.8,1.5),column_depth=u(.8,min(1.8,p['depth']-.1)))
            usable=p['bottom_length']-.2;maximum=min(10,int((usable+.25)/(p['column_width']+.25)))
            if maximum<2:raise ValueError('Cabezal demasiado corto para columnas separadas')
            p['columns']=int(rng.integers(2,maximum+1))
            p['spacing']=(usable-p['column_width'])/(p['columns']-1)
            p['own_variations']+=['numero de columnas y separacion con holgura geometrica; no dimensionamiento resistente']
    elif family=='viga':
        p.update(shape=('rectangular','I','cajon')[index%3],height=u(1.5,2),length=u(8,30))
        p.update(width=p['height']*u(.35,.7),flange=p['height']*u(.12,.2))
        p['web']=p['width']*u(.18,.28);p['source_ids']+=['mtc_tipologias']
        p['own_variations']=['longitud, ancho, espesores y seccion simplificada; no cotas de planos peruanos']
    elif family=='tablero':
        p.update(shape='parche_losa',length=u(1,3),width=u(1,3),height=.4,physical_deck_width=u(6,12))
        p['own_variations']=['recorte local rectangular 1-3 m; no tablero completo del puente']
    validate(p)
    return p

def validate(p):
    if p['family'] not in FAMILIES:raise ValueError('Familia no admitida')
    for key in ('height','length','width','depth','radius','column_height','column_width','column_depth','web','flange'):
        if key in p and (not np.isfinite(p[key]) or p[key]<=0):raise ValueError('Dimension invalida: '+key)
    if p['family']=='apoyo_multicolumna':
        if p['spacing']-p['column_width']<.25-1e-9:raise ValueError('Columnas sin hueco libre')
        if p['column_depth']>=p['depth']:raise ValueError('Columna fuera del cabezal')
        span=(p['columns']-1)*p['spacing']+p['column_width']
        if span>p['bottom_length']-.2+1e-8:raise ValueError('Columnas fuera del apoyo inferior')

def prism(polygon,length,axis,center,tag):
    polygon=np.asarray(polygon,dtype=float)
    # Poligonos convexos en sentido antihorario; todos los del generador lo son.
    edge=np.roll(polygon,-1,axis=0)-polygon
    if any(edge[i,0]*edge[(i+1)%len(edge),1]-edge[i,1]*edge[(i+1)%len(edge),0]<-1e-9 for i in range(len(edge))):raise ValueError('Poligono no convexo')
    dims=[d for d in range(3) if d!=axis];vertices=[]
    for side in (-length/2,length/2):
        layer=np.zeros((len(polygon),3));layer[:,dims]=polygon;layer[:,axis]=side
        vertices.extend(layer+np.asarray(center))
    n=len(polygon);triangles=[]
    for i in range(1,n-1):triangles.extend([(0,i+1,i),(n,n+i,n+i+1)])
    for i in range(n):j=(i+1)%n;triangles.extend([(i,j,n+j),(i,n+j,n+i)])
    faces=np.asarray(triangles)
    if axis==1:faces=faces[:,[0,2,1]]
    return {'vertices':np.asarray(vertices),'faces':faces,'polygon':polygon,'length':length,'axis':axis,'center':np.asarray(center),'tag':tag}

def box(sizes,center,tag):
    x,y,z=sizes
    return prism([[-x/2,-z/2],[x/2,-z/2],[x/2,z/2],[-x/2,z/2]],y,1,center,tag)

def components(p):
    validate(p);family=p['family'];parts=[]
    if family=='columna':
        if p['shape']=='circular':
            angle=np.linspace(0,2*np.pi,64,endpoint=False);polygon=np.c_[np.cos(angle),np.sin(angle)]*p['radius']
            parts=[prism(polygon,p['height'],2,[0,0,p['height']/2],0)]
        else:parts=[box([p['width'],p['depth'],p['height']],[0,0,p['height']/2],0)]
    elif family in ('cabezal','apoyo_multicolumna'):
        z=p.get('column_height',0);h=p['height'];a=p['bottom_length']/2;b=p['length']/2
        parts=[prism([[-a,0],[a,0],[b,h],[-b,h]],p['depth'],1,[0,0,z],1)]
        if family=='apoyo_multicolumna':
            for i in range(p['columns']):
                x=(i-(p['columns']-1)/2)*p['spacing']
                parts.append(box([p['column_width'],p['column_depth'],z],[x,0,z/2],2+i))
    elif family=='viga':
        l,w,h,t,s=p['length'],p['width'],p['height'],p['flange'],p['web']
        if p['shape']=='rectangular':parts=[box([l,w,h],[0,0,h/2],0)]
        else:
            parts=[box([l,w,t],[0,0,t/2],0),box([l,w,t],[0,0,h-t/2],0)]
            if p['shape']=='I':parts.append(box([l,s,h-2*t],[0,0,h/2],0))
            else:
                for sign in (-1,1):parts.append(box([l,s,h-2*t],[0,sign*(w-s)/2,h/2],0))
    else:parts=[box([p['length'],p['width'],p['height']],[0,0,p['height']/2],0)]
    return parts

def contains(part,points,tolerance=1e-8):
    q=np.asarray(points)-part['center'];axis=part['axis'];dims=[i for i in range(3) if i!=axis]
    inside=abs(q[:,axis])<=part['length']/2+tolerance
    poly=part['polygon'];uv=q[:,dims]
    for a,b in zip(poly,np.roll(poly,-1,axis=0)):
        e=b-a;inside&=(e[0]*(uv[:,1]-a[1])-e[1]*(uv[:,0]-a[0]))>=-tolerance
    return inside

def sample_surface(parts,n,rng):
    triangles=[];normals=[];owners=[]
    for owner,part in enumerate(parts):
        tri=part['vertices'][part['faces']];cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);length=np.linalg.norm(cross,axis=1)
        triangles.extend(tri);normals.extend(cross/length[:,None]);owners.extend([owner]*len(tri))
    triangles=np.asarray(triangles);normals=np.asarray(normals);owners=np.asarray(owners)
    area=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1);area/=area.sum()
    collected=[];directions=[];tags=[];total=0
    for _ in range(16):
        count=max(1024,int((n-total)*1.5));choice=rng.choice(len(triangles),count,p=area);selected=triangles[choice]
        uv=rng.random((count,2));u=np.sqrt(uv[:,0]);v=uv[:,1]
        points=(1-u[:,None])*selected[:,0]+(u*(1-v))[:,None]*selected[:,1]+(u*v)[:,None]*selected[:,2]
        owner=owners[choice];keep=np.ones(count,bool)
        for i,part in enumerate(parts):
            candidate=owner!=i
            keep[candidate]&=~contains(part,points[candidate])
        collected.append(points[keep]);directions.append(normals[choice][keep]);tags.append(np.array([parts[i]['tag'] for i in owner[keep]],dtype='i2'))
        total+=int(keep.sum())
        if total>=n:break
    if total<n:raise ValueError('No se pudo muestrear superficie exterior suficiente')
    return np.concatenate(collected)[:n],np.concatenate(directions)[:n],np.concatenate(tags)[:n]

def pair(p,variant=0,n_complete=16384,n_partial=4096,noise=True):
    rng=np.random.default_rng(np.random.SeedSequence([p['seed'],FAMILIES.index(p['family']),int(p['base_id'].split('_')[-1]),variant,71]))
    full,normals,tags=sample_surface(components(p),n_complete,rng)
    # Occlusion de region espacial, con centros distintos segun punto de vista.
    angle=rng.uniform(0,2*np.pi);direction=np.array([np.cos(angle),np.sin(angle),rng.uniform(-.9,.4)])
    view=full@direction;seed=int(np.argmax(view)) if variant%3==0 else int(rng.integers(len(full)))
    remove_fraction=(.10,.20,.30,.40,.50)[variant%5]
    distance=np.linalg.norm(full-full[seed],axis=1);cut=np.quantile(distance,remove_fraction)
    local=distance>cut;front=normals@direction>=0
    mode=('hueco_local','caras_orientadas','caras_orientadas_y_hueco')[variant%3]
    visible=local if variant%3==0 else front if variant%3==1 else front&local
    retained=np.flatnonzero(visible)
    if len(retained)<256:raise ValueError('Vista con pocos puntos observados')
    # Densidad no uniforme para aproximar adquisicion: ponderacion de caras.
    weights=.25+.75*np.clip(normals[retained]@direction,0,1);weights/=weights.sum()
    indices=rng.choice(retained,n_partial,replace=len(retained)<n_partial,p=weights)
    angle=rng.uniform(0,2*np.pi);rotation=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    partial=full[indices]@rotation;target=full@rotation
    sigma=float(rng.uniform(0,.02)) if noise else 0.
    if sigma:partial+=rng.normal(0,sigma,partial.shape)
    # La transformacion depende SOLO de la entrada observada; no revela la
    # extension de una region faltante a traves de la referencia completa.
    center=(partial.min(0)+partial.max(0))/2;extent=float(np.ptp(partial,axis=0).max())
    reference=((target-center)/extent).astype('f4');partial=((partial-center)/extent).astype('f4')
    return {'complete':reference,'partial':partial,'source_indices':indices.astype('i4'),'visible_mask':visible,'part':tags,'center_m':center,'rotation':rotation,'scale_m':extent,'occlusion_fraction':float((~visible).mean()),'noise_sigma_m':sigma,'noise_sigma_normalized':sigma/extent,'occlusion_mode':mode,'unique_input_points':len(np.unique(indices))}

def fingerprint(p):
    data={k:v for k,v in p.items() if k not in ('base_id','seed','source_ids','own_variations')}
    return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()
