"""Selección geométrica interna con evidencia reservada; sin ficha de puente.

Biblioteca acotada de hipótesis, no reconocedor de cualquier forma estructural.
Los contornos de cabezal se comparan por bandas de altura no usadas al ajustar.
No accede a configuraciones de Lurín, aproximaciones previas ni referencias.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull


def section_polygon(kind, params, resolution=96):
    if kind not in ('cilindrica','rectangular','eliptica','octagonal'):raise ValueError('Sección fuera de la biblioteca')
    q=np.asarray(params,dtype=float)
    if kind=='cilindrica':
        a=np.linspace(0,2*np.pi,resolution,endpoint=False);return np.c_[np.cos(a),np.sin(a)]*q[2]
    a,b,theta=q[2:];rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    if kind=='rectangular':poly=np.array([[-a,-b],[a,-b],[a,b],[-a,b]])
    else:
        angle=np.linspace(0,2*np.pi,8 if kind=='octagonal' else resolution,endpoint=False)
        poly=np.c_[a*np.cos(angle),b*np.sin(angle)]
    return poly@rot.T


def borrow_section_model(sections,kind,center):
    """Para fuste escaso, dimensiones de vecinos observados de la misma forma."""
    neighbors=[(i,s['models'][kind]) for i,s in enumerate(sections) if s['status']==kind and kind in s['models']]
    if not neighbors:return None
    q=np.median(np.array([s['parameters'] for _,s in neighbors]),axis=0);q[:2]=center
    return {'parameters':q.tolist(),'parameter_source':'mediana de apoyos observados de la misma sección; no dimensiones manuales',
            'borrowed_from_supports':[i for i,_ in neighbors]}


def polygon_distance(points, poly):
    best=np.full(len(points),np.inf)
    for a,b in zip(poly,np.roll(poly,-1,axis=0)):
        edge=b-a;t=np.clip((points-a)@edge/(edge@edge),0,1)
        best=np.minimum(best,np.linalg.norm(points-a-t[:,None]*edge,axis=1))
    return best


def augment_section(points, base, width, depth, seed):
    """Añade elipse y octágono cuando la cobertura permite distinguirlos."""
    if not base['models'] or len(points)<200:return base
    rng=np.random.default_rng(seed);scale=max(width,depth);points=np.asarray(points)
    lo,hi=np.quantile(points[:,2],[.02,.98]);bins=np.clip(((points[:,2]-lo)/max(hi-lo,1e-8)*8).astype(int),0,7)
    def balance(p):
        _,ids=np.unique(np.floor(p/(scale*.025)).astype('i8'),axis=0,return_index=True);p=p[ids]
        return p[rng.choice(len(p),min(900,len(p)),replace=False)]
    train=balance(points[bins%2==0,:2]);test=balance(points[bins%2==1,:2])
    if min(len(train),len(test))<40:return base
    center=np.asarray(base['models']['cilindrica']['parameters'][:2]);radius=base['models']['cilindrica']['parameters'][2]
    angle=np.sort(np.mod(np.arctan2(test[:,1]-center[1],test[:,0]-center[0]),2*np.pi))
    coverage=float(360-np.degrees(np.diff(np.r_[angle,angle[0]+2*np.pi]).max()))
    base['additional_shapes_coverage_degrees']=coverage
    # Una sola cara de un octágono/elipse no basta para identificarlos.
    if coverage<270:
        base['additional_shapes_status']='cobertura insuficiente; no se fuerza una forma más compleja';return base
    models=base['models'];lower=np.r_[center-scale*.75,scale*.18,scale*.18,-np.pi/2]
    upper=np.r_[center+scale*.75,scale*.95,scale*.95,np.pi/2]
    for kind in ('eliptica','octagonal'):
        def residual(q,p):
            if kind=='eliptica':
                c,s=np.cos(q[4]),np.sin(q[4]);uv=(p-q[:2])@np.array([[c,-s],[s,c]])
                return (np.linalg.norm(uv/q[2:4],axis=1)-1)*min(q[2],q[3])
            return polygon_distance(p-q[:2],section_polygon(kind,q))
        candidates=[]
        for theta in (0.,np.pi/8,-np.pi/8):
            start=np.r_[center,np.clip([radius,radius],lower[2:4]+1e-7,upper[2:4]-1e-7),theta]
            fit=least_squares(lambda q:residual(q,train),start,bounds=(lower,upper),loss='soft_l1',f_scale=.015*scale,max_nfev=90)
            errors=abs(residual(fit.x,test));score=float(np.median(errors)+.25*np.quantile(errors,.8))
            candidates.append((score,fit.x,errors))
        score,parameters,errors=min(candidates,key=lambda v:v[0])
        models[kind]={'parameters':parameters.tolist(),'validation_score':score,'median_validation_error':float(np.median(errors)),
                      'p80_validation_error':float(np.quantile(errors,.8)),'coverage_degrees':coverage}
    old=min(('cilindrica','rectangular'),key=lambda k:models[k]['validation_score'])
    winner=min(('eliptica','octagonal'),key=lambda k:models[k]['validation_score'])
    error=models[winner]['validation_score'];old_error=models[old]['validation_score']
    gain=old_error-error
    if error<.04*scale and gain>max(.003*scale,.25*old_error):
        if winner=='eliptica':
            a,b=models[winner]['parameters'][2:4]
            if max(a,b)/min(a,b)<1.15:return base
        base.update(status=winner,hypothesis=winner,parameters=models[winner]['parameters'],
                    reason='menor error en bandas reservadas, ventaja absoluta/relativa y cobertura angular suficiente')
    return base


def basis(t,kind):
    t=np.asarray(t,dtype=float)
    if kind=='rectangular':return np.ones((len(t),1))
    if kind=='trapezoidal':return np.c_[np.ones(len(t)),t]
    if kind=='curvo':return np.c_[np.ones(len(t)),t,t*t]
    if kind.startswith('hombro_'):
        hinge=float(kind.split('_')[1]);return np.c_[np.ones(len(t)),t,np.maximum(t-hinge,0)]
    if kind.startswith('perfil_I_'):
        thickness=float(kind.split('_')[-1]);return np.c_[np.ones(len(t)),((t<=thickness)|(t>=1-thickness)).astype(float)]
    raise ValueError('Perfil no conocido')


def select_curve(z,values,noise,allow_i=False):
    """LOO por nivel: el nivel evaluado nunca interviene en ese ajuste."""
    z=np.asarray(z);values=np.asarray(values);t=(z-z.min())/max(float(np.ptp(z)),1e-12)
    kinds=['rectangular','trapezoidal','curvo','hombro_0.25','hombro_0.5','hombro_0.75']
    if allow_i:kinds+=['perfil_I_0.15','perfil_I_0.25']
    candidates={}
    for kind in kinds:
        design=basis(t,kind);k=design.shape[1]
        if len(z)<=k+1:continue
        errors=[]
        for i in range(len(z)):
            keep=np.arange(len(z))!=i
            coef=np.linalg.lstsq(design[keep],values[keep],rcond=None)[0]
            errors.append(abs(float(design[i]@coef)-values[i]))
        coef=np.linalg.lstsq(design,values,rcond=None)[0];pred=design@coef
        cv=float(np.mean(errors));penalty=noise*.25*k
        candidates[kind]={'validation_error':cv,'complexity_penalty':float(penalty),'score':cv+penalty,
                          'parameters':coef.tolist(),'parameters_count':k,'predicted_values':pred.tolist()}
    if not candidates:raise ValueError('Pocos niveles para comparar perfiles')
    best=min(candidates,key=lambda k:candidates[k]['score']);minimum=candidates[best]['score']
    equivalent=[k for k in candidates if candidates[k]['score']<=minimum+noise*.25]
    winner=min(equivalent,key=lambda k:(candidates[k]['parameters_count'],candidates[k]['score']))
    return np.array(candidates[winner]['predicted_values']),{'selected':winner,'candidates':candidates,'noise_tolerance':float(noise),
            'confidence':'aproximada' if len(equivalent)>1 else 'preferida por evidencia reservada','equivalent_hypotheses':equivalent}


def observed_beam_forms(observed,columns,prior,longitudinal):
    """Anchura transversal y dos extremos por altura, usando solo observaciones."""
    profile=dict(prior);z=np.asarray(profile['levels_z']);slope=profile['row_center_slope'];intercept=profile['row_center_intercept']
    if len(z)<5:z=np.linspace(z.min(),z.max(),7)
    height=float(np.ptp(z));spacing=float(np.median(np.diff([c['s'] for c in columns])))
    half_band=max(.055,height*.085);widths=[];left=[];right=[];counts=[];endpoint_counts=[]
    lower_s,upper_s=columns[0]['s'],columns[-1]['s'];rowcenter=lambda p:slope*p[:,0]+intercept
    depth_prior=float(max(profile['depths']));contact=2*max(c['radius'] for c in columns)
    near_ends=(abs(observed[:,0]-lower_s)<min(.45,.2*spacing))|(abs(observed[:,0]-upper_s)<min(.45,.2*spacing))
    for level in z:
        band=observed[(abs(observed[:,2]-level)<half_band)&near_ends]
        delta=band[:,1]-rowcenter(band);delta=delta[abs(delta)<max(depth_prior*.8,contact)]
        if len(delta)>=30:
            a,b=np.quantile(delta,[.03,.97]);width=(b-a) if a<-.1*contact and b>.1*contact else 2*max(abs(a),abs(b))
            widths.append(float(np.clip(width,contact*.25,depth_prior*1.5)));counts.append(len(delta))
        else:widths.append(np.nan);counts.append(len(delta))
        # No usar extensiones de la viga longitudinal/pista alejadas de la fila.
        b=observed[(abs(observed[:,2]-level)<half_band)]
        delta=abs(b[:,1]-rowcenter(b));b=b[delta<max(contact*.55,depth_prior*.35)]
        l=b[(b[:,0]>=lower_s-spacing*.65)&(b[:,0]<lower_s+spacing*.35),0]
        r=b[(b[:,0]<=upper_s+spacing*.65)&(b[:,0]>upper_s-spacing*.35),0]
        left.append(float(np.quantile(l,.035)) if len(l)>=20 else np.nan)
        right.append(float(np.quantile(r,.965)) if len(r)>=20 else np.nan);endpoint_counts.append([len(l),len(r)])
    widths=np.array(widths);good=np.isfinite(widths)
    width_selection={'selected':'prior_por_evidencia_insuficiente','confidence':'incierta'}
    if good.sum()>=5:
        raw=widths[good];noise=max(.015*contact,.02*float(np.median(raw)))
        _,width_selection=select_curve(z[good],raw,noise,allow_i=True)
        model=width_selection['candidates'][width_selection['selected']]
        t=(z-z[good].min())/max(float(np.ptp(z[good])),1e-12)
        depths=basis(t,width_selection['selected'])@model['parameters']
        depths=np.clip(depths,contact*.25,depth_prior*1.5);depths[0]=max(depths[0],contact)
    else:depths=np.interp(z,np.asarray(prior['levels_z']),prior['depths'])
    # Cada extremo tiene su propia selección: vertical, inclinado, hombro o curvo.
    endpoints=[];selections=[]
    for side,raw in enumerate((np.array(left),np.array(right))):
        valid=np.isfinite(raw)
        if valid.sum()>=5:
            _,choice=select_curve(z[valid],raw[valid],max(.025,.03*contact))
            model=choice['candidates'][choice['selected']];t=(z-z[valid].min())/max(float(np.ptp(z[valid])),1e-12)
            fitted=basis(t,choice['selected'])@model['parameters']
            # No extender fuera de la banda observada para inventar una punta.
            fitted=np.clip(fitted,raw[valid].min()-.04*contact,raw[valid].max()+.04*contact)
        else:
            fallback=lower_s-contact*.5 if side==0 else upper_s+contact*.5
            fitted=np.full(len(z),float(np.nanmedian(raw)) if valid.any() else fallback)
            choice={'selected':'extremo_recto_por_evidencia_insuficiente','confidence':'incierta'}
        endpoints.append(fitted);selections.append(choice)
    l,r=endpoints
    if np.any(r-l<spacing):raise ValueError('Extremos incompatibles con una viga compartida')
    profile.update(levels_z=z.tolist(),depths=np.asarray(depths).tolist(),relative_centers_d=np.interp(z,prior['levels_z'],prior['relative_centers_d']).tolist(),
                   left_s=l.tolist(),right_s=r.tolist(),shape_selection={'cross_section':width_selection,'left_end':selections[0],'right_end':selections[1]},
                   observed_profile={'widths':[float(v) if np.isfinite(v) else None for v in widths],'left_s':[float(v) if np.isfinite(v) else None for v in left],
                                     'right_s':[float(v) if np.isfinite(v) else None for v in right],'width_counts':counts,'endpoint_counts':endpoint_counts},
                   source='comparación automática de perfiles y extremos por bandas de altura observadas; sin geometría completa ni ficha manual')
    return profile


def loft_beam_parts(profile):
    """Tramos convexos por banda; conserva perfiles no convexos sin envolverlos."""
    layers=[]
    for z,l,r,offset,depth in zip(profile['levels_z'],profile['left_s'],profile['right_s'],profile['relative_centers_d'],profile['depths']):
        layer=[]
        for s,sign in ((l,-1),(r,-1),(r,1),(l,1)):
            d=profile['row_center_slope']*s+profile['row_center_intercept']+offset+sign*depth/2;layer.append([s,d,z])
        layers.append(layer)
    parts=[]
    for lower,upper in zip(layers[:-1],layers[1:]):
        vertices=np.vstack([lower,upper]);hull=ConvexHull(vertices)
        parts.append({'kind':'convex_profile','vertices':vertices,'faces':hull.simplices,'planes':hull.equations,'tag':1})
    return parts


def beam_underside(position,profile):
    z=np.asarray(profile['levels_z']);left=np.asarray(profile['left_s']);right=np.asarray(profile['right_s'])
    for i in range(len(z)-1):
        if left[i]<=position<=right[i]:return float(z[i])
        roots=[]
        for edge in (left,right):
            if abs(edge[i+1]-edge[i])>1e-12:
                t=(position-edge[i])/(edge[i+1]-edge[i])
                if 0<=t<=1:roots.append(t)
        for t in sorted(roots):
            l=left[i]+t*(left[i+1]-left[i]);r=right[i]+t*(right[i+1]-right[i])
            if l-1e-7<=position<=r+1e-7:return float(z[i]+t*(z[i+1]-z[i]))
    return float(z[-1])
