"""Referencias estructurales automáticas desde XYZ; no recibe clases ni puente.

Ámbito: puente vial de tablero continuo, gravedad en Z. Se estima superficie,
recorrido/bordes y persistencia vertical; los ajustes inciertos se atenúan.
No presupone número de pilares ni secciones cilíndricas/rectangulares.
"""
import numpy as np
from scipy import ndimage

NOMBRES=['altura_centro','altura_inferior','altura_superior','extension_vertical',
    'extension_longitudinal','extension_transversal','lateral_absoluto','margen_extremo',
    'fraccion_sobre_superficie','fraccion_bajo_superficie','cercania_borde','cercania_mediana',
    'persistencia_vertical','compacidad_apoyo','eje_vertical','normal_vertical',
    'linealidad','planaridad','confianza_superficie','confianza_extremos']


def calcular(xyz,groups):
    xyz=np.asarray(xyz,dtype='f8');groups=np.asarray(groups,dtype='i8');n=int(groups.max())+1
    count=np.bincount(groups,minlength=n)
    if not len(xyz) or (count==0).any() or not np.isfinite(xyz).all():raise ValueError('XYZ/grupos inválidos')
    origin=np.median(xyz,axis=0);q=xyz-origin
    _,vec=np.linalg.eigh(np.cov(q[:,:2].T));axis=vec[:,-1]
    if axis[np.argmax(abs(axis))]<0:axis=-axis
    rotation=np.array([[axis[0],-axis[1]],[axis[1],axis[0]]]);q[:,:2]=q[:,:2]@rotation
    low,high=np.percentile(q,[.5,99.5],axis=0);size=(high-low).clip(1e-5)
    # Equal XY cells avoid giving the densest reconstructed patch all votes.
    nx,ny,nz=128,48,96
    ix=np.clip(((q[:,0]-low[0])/size[0]*nx).astype(int),0,nx-1)
    iy=np.clip(((q[:,1]-low[1])/size[1]*ny).astype(int),0,ny-1)
    iz=np.clip(((q[:,2]-low[2])/size[2]*nz).astype(int),0,nz-1)
    hist=np.bincount((ix*ny+iy)*nz+iz,minlength=nx*ny*nz).reshape(nx*ny,nz)
    total=hist.sum(1);cumulative=hist.cumsum(1);zbin=(cumulative<.8*total[:,None]).sum(1).clip(0,nz-1)
    cells=np.flatnonzero(total>=3)
    points=np.column_stack([low[0]+(cells//ny+.5)*size[0]/nx,
        low[1]+(cells%ny+.5)*size[1]/ny,low[2]+(zbin[cells]+.5)*size[2]/nz])
    tol=max(size[2]*.016,size[0]*.001,1e-6);rng=np.random.default_rng(173)
    candidates=[];best=0
    if len(points)>=12:
        for _ in range(240):
            sample=points[rng.choice(len(points),3,replace=False)];A=np.column_stack([np.ones(3),sample[:,:2]])
            try:coef=np.linalg.solve(A,sample[:,2])
            except np.linalg.LinAlgError:continue
            if np.linalg.norm(coef[1:])>.6:continue
            residual=points[:,2]-(coef[0]+points[:,:2]@coef[1:]);mask=abs(residual)<tol*1.5
            if mask.sum()<12 or np.ptp(points[mask,0])<.35*size[0] or np.ptp(points[mask,1])<.12*size[1]:continue
            score=int(mask.sum());best=max(best,score);candidates.append((score,coef,mask))
    confidence=0.
    if candidates:
        eligible=[c for c in candidates if c[0]>=max(best*.28,len(points)*.035)]
        score,coef,inliers=max(eligible,key=lambda c:c[0]*np.exp(np.clip((c[1][0]-low[2])/size[2],-2,2))/(1+10*np.square(c[1][1:]).sum()))
        confidence=float(min(1,score/max(len(points)*.25,1)))
    else:
        coef=np.array([np.percentile(q[:,2],65),0.,0.]);inliers=np.ones(len(points),bool)
    # Gentle quadratic profile supports grade/vertical curvature without a
    # per-bridge plane, height, bounds or annotation file.
    def design(p):
        u=p[:,0]/size[0];v=p[:,1]/size[1]
        return np.column_stack([np.ones(len(p)),u,v,u*u,u*v,v*v])
    coeff=np.array([coef[0],coef[1]*size[0],coef[2]*size[1],0,0,0.])
    A=design(points)
    if len(points)>=30:
        for _ in range(4):
            good=abs(points[:,2]-A@coeff)<tol*2
            if good.sum()<30:break
            trial=np.linalg.lstsq(A[good],points[good,2],rcond=None)[0]
            if np.max(abs(trial[3:]))>size[2]*2:break
            coeff=trial
    surface=design(q)@coeff;gap=surface-q[:,2]
    inliers=abs(points[:,2]-A@coeff)<tol*2
    stations=[];left=[];right=[]
    if inliers.any():
        for a,b in zip(np.linspace(low[0],high[0],25)[:-1],np.linspace(low[0],high[0],25)[1:]):
            p=points[inliers&(points[:,0]>=a)&(points[:,0]<b)]
            if len(p)>=4:
                stations.append((a+b)/2);left.append(np.percentile(p[:,1],3));right.append(np.percentile(p[:,1],97))
    if len(stations)>=4:
        station=np.asarray(stations);centerline=(np.asarray(left)+right)/2
        centercoef=np.polyfit(station/size[0],centerline,min(3,len(station)-1))
        halfwidth=max(float(np.median((np.asarray(right)-left)/2)),size[1]*.1)
    else:centercoef=np.array([0.]);halfwidth=size[1]*.4;confidence*=.25
    lateral=(q[:,1]-np.polyval(centercoef,q[:,0]/size[0]))/halfwidth
    below=(gap>tol*2)&(abs(lateral)<1.2)
    H=max(float(np.percentile(gap[below],99)) if below.any() else size[2],tol*5)
    H=min(H,halfwidth*4);H=max(H,tol*5)
    under=(gap>max(tol*2,.04*H))&(gap<.65*H)&(abs(lateral)<1.15)
    if under.sum()>=100:
        corelo,corehi=np.percentile(q[under,0],[3,97]);coreconfidence=float(np.clip((corehi-corelo)/size[0],0,1))
        if corehi-corelo<size[0]*.3:corelo,corehi=low[0],high[0];coreconfidence=0.
    else:corelo,corehi=low[0],high[0];coreconfidence=0.
    corelen=max(corehi-corelo,1e-6)
    margin=np.minimum(q[:,0]-corelo,corehi-q[:,0])/corelen
    # Distinct height slices distinguish vertical supports from a horizontal
    # floor, including irregular/flared supports and partial faces.
    px,py=256,128
    gx=np.clip(((q[:,0]-low[0])/size[0]*px).astype(int),0,px-1)
    gy=np.clip(((q[:,1]-low[1])/size[1]*py).astype(int),0,py-1)
    gz=np.clip((gap/H*16).astype(int),0,15)
    body=(gap>.15*H)&(gap<1.1*H)
    occupancy=np.bincount(((gx[body]*py+gy[body])*16+gz[body]),minlength=px*py*16).reshape(px,py,16)
    persistence=(occupancy>0).sum(2)/16
    smoothed=ndimage.maximum_filter(persistence,size=3)
    components,num=ndimage.label(smoothed>.28,np.ones((3,3)))
    compact=np.zeros(num+1)
    for k,slices in enumerate(ndimage.find_objects(components),1):
        if slices is None:continue
        extent=max((slices[0].stop-slices[0].start)*size[0]/px,(slices[1].stop-slices[1].start)*size[1]/py)
        compact[k]=np.clip(1-extent/(H*.7),0,1)
    pointpersist=smoothed[gx,gy];pointcompact=compact[components[gx,gy]]
    def mean(v):return np.bincount(groups,weights=v,minlength=n)/count
    centers=np.column_stack([mean(q[:,a]) for a in range(3)])
    lo=np.full((n,3),np.inf);hi=-lo.copy();cov=np.zeros((n,3,3));delta=q-centers[groups]
    for a in range(3):
        np.minimum.at(lo[:,a],groups,q[:,a]);np.maximum.at(hi[:,a],groups,q[:,a])
        for b in range(a,3):cov[:,a,b]=mean(delta[:,a]*delta[:,b]);cov[:,b,a]=cov[:,a,b]
    eigen,vectors=np.linalg.eigh(cov);eigen=eigen.clip(0);den=eigen[:,-1].clip(1e-9)
    extension=hi-lo;zg=mean(gap)/H
    features=np.column_stack([-zg,-zg-extension[:,2]/(2*H),-zg+extension[:,2]/(2*H),
        extension[:,2]/H,extension[:,0]/(halfwidth*2),extension[:,1]/(halfwidth*2),
        mean(abs(lateral)),mean(margin),mean(gap<-tol),mean(gap>tol*2),
        mean(np.exp(-((abs(lateral)-1)/.18)**2)),mean(np.exp(-(lateral/.15)**2)),
        mean(pointpersist),mean(pointcompact),abs(vectors[:,2,-1]),abs(vectors[:,2,0]),
        (eigen[:,-1]-eigen[:,-2])/den,(eigen[:,-2]-eigen[:,0])/den,
        np.full(n,confidence),np.full(n,coreconfidence)])
    features=np.nan_to_num(features).clip(-5,5).astype('f4')
    meta=dict(version=1,clases_leidas=False,ajuste_superficie=confidence,confianza_extremos=coreconfidence,
        altura_soporte=H,semiancho=halfwidth,extremos=[float(corelo),float(corehi)],
        superficie_coeficientes=coeff.tolist(),centrolinea_coeficientes=centercoef.tolist(),
        traslacion=origin.tolist(),rotacion_xy=rotation.tolist(),gravedad='Z',
        alcance='tablero vial continuo; ajuste incierto atenúa SOL, sin garantía para puentes fuera de este ámbito')
    return features,meta


def costes(features):
    """Violaciones suaves por nodo; no son etiquetas asignadas por geometría."""
    f=np.asarray(features);cost=np.zeros((len(f),10),dtype='f4')
    above=np.clip(f[:,0]*6,0,1);deep=np.clip(-f[:,0]*2,0,1)
    upright=f[:,12]*(.35+.65*f[:,13])*np.clip(f[:,3]*4,0,1)
    interior=np.clip(f[:,7]*4,0,1)*f[:,19]
    edge=np.clip(f[:,10],0,1);middle=np.clip(f[:,11],0,1);conf=f[:,18]
    cost[:,0]=above # absence of persistence is not proof of absence of a column
    # A broad, nearly horizontal sheet at the roadway level is evidence
    # against a girder; a vertical girder face alone is not a pillar.
    surface_sheet=np.exp(-(f[:,0]/.03)**2)*f[:,15]*np.clip(1-f[:,3]*6,0,1)*(1-f[:,8])
    cost[:,1]=upright*.9+above*.5+np.clip(-f[:,2]-.5,0,1)*.5+surface_sheet
    cost[:,2]=np.clip(abs(f[:,0])*6-.25,0,1)+upright*.5+np.clip(-f[:,7]*3,0,1)*f[:,19]*.25
    cost[:,3]=np.clip(-f[:,2]*6,0,1)+(1-edge)*np.clip(f[:,8]*2,0,1)*.7
    cost[:,4]=interior*np.clip(f[:,9]*3,0,1)+deep*.7
    cost[:,5]=np.clip(-f[:,2]*6,0,1)+(1-middle)*.7
    cost[:,6]=np.clip(f[:,1]+.6,0,1)+upright*.7
    cost[:,7]=interior*upright # supports in the interior are not end abutments
    return (cost*conf[:,None]).clip(0,2)

