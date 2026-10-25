"""Localización geométrica automática del conjunto dominante de apoyos."""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

def support_context(xyz,plane,seed=241131):
    deep=xyz[xyz[:,2]<plane-3.]
    if len(deep)<1000:raise ValueError('No hay evidencia suficiente de apoyos bajo el tablero')
    rng=np.random.default_rng(seed);sample=deep[rng.choice(len(deep),min(20000,len(deep)),replace=False),:2];best=None
    for _ in range(256):
        a,b=sample[rng.choice(len(sample),2,replace=False)];v=b-a;length=np.linalg.norm(v)
        if length<5.:continue
        v/=length;n=np.array([-v[1],v[0]]);inside=abs((sample-a)@n)<.8;score=int(inside.sum())
        if best is None or score>best[0]:best=(score,inside)
    if best is None:raise ValueError('No se detecto una alineacion de apoyos')
    points=sample[best[1]];origin=np.median(points,axis=0);_,vectors=np.linalg.eigh(np.cov(points.T));axis=vectors[:,-1]
    if axis[np.argmax(abs(axis))]<0:axis=-axis
    normal=np.array([-axis[1],axis[0]]);ds=(deep[:,:2]-origin)@normal;ss=(deep[:,:2]-origin)@axis;inliers=abs(ds)<1.2
    lower,upper=np.quantile(ss[inliers],[.005,.995])+np.array([-.6,.6]);bottom=float(np.quantile(deep[inliers,2],.005))-.15
    local=np.column_stack(((xyz[:,:2]-origin)@axis,(xyz[:,:2]-origin)@normal,xyz[:,2]))
    halfwidth=max(1.2,float(np.quantile(abs(ds[inliers]),.99))+.5)
    ids=np.flatnonzero((local[:,0]>=lower)&(local[:,0]<=upper)&(abs(local[:,1])<=halfwidth)&(local[:,2]>=bottom)&(local[:,2]<plane-.2))
    roi=local[ids];center=(roi.min(0)+roi.max(0))/2;scale=float(np.ptp(roi,axis=0).max());normalized=((roi-center)/scale).astype('f4')
    if len(roi)<4096:raise ValueError('Conjunto con pocos puntos')
    # Solo diagnostico de periodicidad: no se usa para inventar columnas.
    low=roi[roi[:,2]<plane-2.];edges=np.arange(lower,upper+.1,.1);hist,_=np.histogram(low[:,0],edges);smoothed=gaussian_filter1d(hist.astype(float),1.2)
    peaks,_=find_peaks(smoothed,distance=12,prominence=max(5.,smoothed.max()*.015))
    return {'ids':ids,'local':roi,'normalized':normalized,'center':center,'scale':scale,'origin_xy':origin,'axis_xy':axis,'normal_xy':normal,'plane':plane,'bounds':[lower,upper,-halfwidth,halfwidth,bottom,plane-.2],'density_peaks_s':((edges[:-1]+edges[1:])/2)[peaks],'density_hist_s':(edges[:-1]+edges[1:])/2,'density_hist':smoothed,'ransac_inlier_fraction':best[0]/len(sample)}
