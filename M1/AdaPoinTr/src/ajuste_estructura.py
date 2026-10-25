"""Estimación inicial de alineación, bases y cabezal desde observaciones."""
import numpy as np
from .geometrias_sinteticas import prism,box

def diagonal_edge(points,side,top,bottom,rng):
    s=points[:,0];cut=np.quantile(s,.3 if side=='left' else .7)
    p=points[((s<cut) if side=='left' else (s>cut))&(points[:,2]>top-.45*(top-bottom))]
    p=p[rng.choice(len(p),min(len(p),6000),replace=False)];best=None
    for _ in range(400):
        a,b=p[rng.choice(len(p),2,replace=False)];ds=b[0]-a[0]
        if abs(ds)<.5:continue
        slope=(b[2]-a[2])/ds
        if not ((-1.5<slope<-.10) if side=='left' else (.10<slope<1.5)):continue
        intercept=a[2]-slope*a[0];error=abs(p[:,2]-slope*p[:,0]-intercept)/np.sqrt(1+slope*slope);inside=error<.08
        if best is None or inside.sum()>best[0]:best=(int(inside.sum()),inside)
    if best is None or best[0]<30:return None
    selected=p[best[1]];slope,intercept=np.polyfit(selected[:,0],selected[:,2],1)
    return {'slope':float(slope),'intercept':float(intercept),'inliers':len(selected),'lower_z':float(np.quantile(selected[:,2],.05)),'median_residual':float(np.median(abs(selected[:,2]-slope*selected[:,0]-intercept)))}

def fit_structure(local,peaks,plane,seed=241132):
    rng=np.random.default_rng(seed);base=float(np.quantile(local[:,2],.005));s=local[:,0]
    outer=(s<np.quantile(s,.15))|(s>np.quantile(s,.85));top=float(np.quantile(local[outer,2],.9995));top=min(top,float(plane)-.15)
    left=diagonal_edge(local,'left',top,base,rng);right=diagonal_edge(local,'right',top,base,rng)
    bottoms=[line['lower_z'] for line in (left,right) if line]
    cap_bottom=float(np.median(bottoms)) if bottoms else top-.22*(top-base)
    cap_bottom=float(np.clip(cap_bottom,top-.45*(top-base),top-.08*(top-base)))
    spacing=float(np.median(np.diff(peaks)));observations=[]
    for peak in peaks:
        p=local[(abs(s-peak)<min(.8,.4*spacing))&(local[:,2]<cap_bottom-.15)]
        if len(p)<30:continue
        bottom=float(np.quantile(p[:,2],.02));height=float(np.quantile(p[:,2],.98)-bottom)
        stem=p[(p[:,2]>bottom+.2)&(p[:,2]<bottom+max(.5,min(2.3,.8*height)))]
        if len(stem)<30:stem=p
        low,high=np.quantile(stem[:,:2],[.05,.95],axis=0)
        observations.append({'peak':float(peak),'s':float((low[0]+high[0])/2),'d':float((low[1]+high[1])/2),'width':float(high[0]-low[0]),'depth':float(high[1]-low[1]),'bottom':bottom,'observed_height':height,'points':len(p)})
    if len(observations)<2:raise ValueError('No hay suficientes apoyos observados para hipotesis compartida')
    good=[o for o in observations if o['observed_height']>.35*(top-base)] or observations
    width=float(np.clip(np.median([o['width'] for o in good]),.3,.55*spacing));depth=float(np.clip(np.median([o['depth'] for o in good]),.45*width,1.6*width))
    bottom_slope,bottom_intercept=np.polyfit([o['s'] for o in observations],[o['bottom'] for o in observations],1)
    col_min=min(o['s'] for o in observations)-width/2;col_max=max(o['s'] for o in observations)+width/2
    left_top=(top-left['intercept'])/left['slope'] if left else col_min-.4
    right_top=(top-right['intercept'])/right['slope'] if right else col_max+.4
    left_bottom=(cap_bottom-left['intercept'])/left['slope'] if left else col_min
    right_bottom=(cap_bottom-right['intercept'])/right['slope'] if right else col_max
    left_top=min(float(left_top),col_min);right_top=max(float(right_top),col_max)
    left_bottom=float(np.clip(left_bottom,left_top+.1,(left_top+right_top)/2-.1));right_bottom=float(np.clip(right_bottom,(left_top+right_top)/2+.1,right_top-.1))
    dc=float(np.median([o['d'] for o in good]));cap_depth=max(depth*1.2,float(np.quantile(local[outer,1],.9)-np.quantile(local[outer,1],.1)))
    cap_depth=float(min(cap_depth,depth*2.))
    polygon=[[left_bottom,cap_bottom],[right_bottom,cap_bottom],[right_top,top],[left_top,top]]
    parts=[prism(polygon,cap_depth,1,[0,dc,0],1)];columns=[]
    def underside(position):
        if position<left_bottom:return cap_bottom+(top-cap_bottom)*(left_bottom-position)/(left_bottom-left_top)
        if position>right_bottom:return cap_bottom+(top-cap_bottom)*(position-right_bottom)/(right_top-right_bottom)
        return cap_bottom
    for i,o in enumerate(observations):
        floor=float(bottom_slope*o['s']+bottom_intercept);ceiling=float(max(underside(o['s']-width/2),underside(o['s']+width/2)))
        parts.append(box([width,depth,ceiling-floor],[o['s'],o['d'],(ceiling+floor)/2],2+i));columns.append({**o,'fitted_bottom':floor,'fitted_top':ceiling,'width_shared':width,'depth_shared':depth})
    return parts,{'columns':columns,'beam_polygon_s_z':polygon,'beam_depth':cap_depth,'beam_center_d':dc,'beam_bottom_z':cap_bottom,'beam_top_z':top,'left_edge_fit':left,'right_edge_fit':right,'fallback_beam_depth_used':not bool(bottoms),'assumptions':['columnas de seccion rectangular compartida','una viga continua de perfil trapezoidal','eje y posiciones inferidos de observaciones, no medidas certificadas'],'support_count':len(columns)}
