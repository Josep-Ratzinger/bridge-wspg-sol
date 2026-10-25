"""Ajuste de secciones con observaciones reservadas por altura."""
import numpy as np
from scipy.optimize import least_squares

def boundary_residual(points,parameters,kind):
    center=parameters[:2];q=points-center
    if kind=='cilindrica':return np.linalg.norm(q,axis=1)-parameters[2]
    a,b,angle=parameters[2:];c,s=np.cos(angle),np.sin(angle);rot=np.array([[c,-s],[s,c]])
    local=q@rot;delta=abs(local)-np.array([a,b])
    return np.linalg.norm(np.maximum(delta,0),axis=1)+np.minimum(delta.max(1),0)

def fit_section(points,prior_center,prior_width,prior_depth,seed=241133):
    points=np.asarray(points);center=np.asarray(prior_center,dtype=float);scale=max(prior_width,prior_depth);rng=np.random.default_rng(seed)
    if len(points)<160 or np.ptp(points[:,2])<.15:
        return {'status':'incierta','reason':'observaciones insuficientes para comparar secciones','hypothesis':'rectangular','parameters':[float(center[0]),float(center[1]),prior_width/2,prior_depth/2,0.],'models':{},'points':len(points)}
    zlo,zhi=np.quantile(points[:,2],[.02,.98]);bins=np.minimum(7,np.maximum(0,((points[:,2]-zlo)/max(zhi-zlo,1e-8)*8).astype(int)))
    train=points[bins%2==0,:2];test=points[bins%2==1,:2]
    if min(len(train),len(test))<50:return {'status':'incierta','reason':'sin suficientes bandas de altura para validacion','hypothesis':'rectangular','parameters':[float(center[0]),float(center[1]),prior_width/2,prior_depth/2,0.],'models':{},'points':len(points)}
    def balanced(p):
        _,indices=np.unique(np.floor(p/(scale*.025)).astype('i8'),axis=0,return_index=True);p=p[indices]
        return p[rng.choice(len(p),min(2500,len(p)),replace=False)]
    train=balanced(train);test=balanced(test);models={}
    for kind in ('cilindrica','rectangular'):
        if kind=='cilindrica':
            lower=[center[0]-.75*scale,center[1]-.75*scale,.18*scale];upper=[center[0]+.75*scale,center[1]+.75*scale,.95*scale]
            starts=[[center[0],center[1],.5*scale]]
        else:
            lower=[center[0]-.75*scale,center[1]-.75*scale,.18*scale,.18*scale,-np.pi/4];upper=[center[0]+.75*scale,center[1]+.75*scale,.95*scale,.95*scale,np.pi/4]
            starts=[[center[0],center[1],prior_width/2,prior_depth/2,angle] for angle in (0.,-.35,.35)]
        fits=[]
        for start in starts:
            optimized=least_squares(lambda x:boundary_residual(train,x,kind),start,bounds=(lower,upper),loss='soft_l1',f_scale=scale*.025,max_nfev=160)
            errors=abs(boundary_residual(test,optimized.x,kind));score=float(np.median(errors)+.25*np.quantile(errors,.8))
            fits.append((score,optimized.x,errors))
        score,parameters,errors=min(fits,key=lambda t:t[0]);models[kind]={'parameters':parameters.tolist(),'validation_score':score,'median_validation_error':float(np.median(errors)),'p80_validation_error':float(np.quantile(errors,.8))}
    winner=min(models,key=lambda k:models[k]['validation_score']);loser=next(k for k in models if k!=winner);best=models[winner];advantage=(models[loser]['validation_score']-best['validation_score'])/max(models[loser]['validation_score'],1e-8)
    params=np.array(best['parameters']);evidence=abs(boundary_residual(test,params,winner))<=max(.05*scale,2*best['median_validation_error']);p=test[evidence]
    if winner=='cilindrica':
        angle=np.sort(np.mod(np.arctan2(p[:,1]-params[1],p[:,0]-params[0]),2*np.pi));coverage=360-np.degrees(np.diff(np.r_[angle,angle[0]+2*np.pi]).max()) if len(angle)>2 else 0.;sufficient=coverage>=100.
        evidence_record={'angular_coverage_degrees':float(coverage)}
    else:
        c,s=np.cos(params[4]),np.sin(params[4]);q=(p-params[:2])@np.array([[c,-s],[s,c]]);dist=np.column_stack([abs(q[:,0]-params[2]),abs(q[:,0]+params[2]),abs(q[:,1]-params[3]),abs(q[:,1]+params[3])]);labels=dist.argmin(1);counts=np.bincount(labels,minlength=4);faces=int((counts>=max(10,len(p)*.1)).sum());sufficient=faces>=2;evidence_record={'supported_faces':faces}
    confident=bool(advantage>=.2 and best['validation_score']<=.06*scale and sufficient)
    return {'status':winner if confident else 'incierta','reason':'ventaja y cobertura suficientes' if confident else 'ventaja, ajuste o cobertura insuficiente para afirmar la seccion','hypothesis':winner,'parameters':best['parameters'],'models':models,'relative_advantage':float(advantage),'evidence':evidence_record,'points':len(points),'validation_points':len(test),'reference_scope':'ajuste a superficies observadas, no clase semantica confirmada'}
