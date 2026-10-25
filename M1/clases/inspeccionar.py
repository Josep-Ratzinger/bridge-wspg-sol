"""Vistas de la nube observada para la revisión de clases, sin leer el SPG."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODULE=Path(__file__).resolve().parent
PROJECT=MODULE.parents[1]


def main():
    source=PROJECT/'M1/limpieza_y_preparacion/resultados/preparado/preparacion.npz'
    with np.load(source) as d:x=d['xyz'];rgb=d['rgb'];meta=json.loads(str(d['metadatos']))
    rng=np.random.default_rng(241170);ids=rng.choice(len(x),min(300000,len(x)),replace=False);p=x[ids];c=rgb[ids]/255
    out=MODULE/'resultados/revision';out.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(3,1,figsize=(17,12))
    axes[0].scatter(p[:,0],p[:,1],s=.15,c=c);axes[0].set(title='Planta observada (RGB original)',xlabel='X',ylabel='Y');axes[0].set_aspect('equal')
    axes[1].scatter(p[:,0],p[:,2],s=.15,c=c);axes[1].set(title='Vista longitudinal observada',xlabel='X',ylabel='Z');axes[1].set_aspect('equal')
    keep=abs(x[:,0]-8.2)<1.;p=x[keep];c=rgb[keep]/255
    axes[2].scatter(p[:,1],p[:,2],s=.35,c=c);axes[2].set(title='Sección transversal observada cerca de los apoyos',xlabel='Y',ylabel='Z');axes[2].set_aspect('equal')
    for ax in axes:ax.grid(alpha=.1)
    fig.tight_layout();fig.savefig(out/'vistas_observadas.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(16,9))
    for ax,(lo,hi) in zip(axes,[(.2,1.6),(-1.5,-.3),(-.15,.15)]):
        mask=(x[:,2]>lo)&(x[:,2]<hi);v=x[mask];color=rgb[mask]/255
        if len(v)>150000:
            i=rng.choice(len(v),150000,replace=False);v=v[i];color=color[i]
        ax.scatter(v[:,0],v[:,1],s=.25,c=color);ax.set_aspect('equal');ax.set(title=f'Planta: Z entre {lo} y {hi}',xlabel='X',ylabel='Y');ax.grid(alpha=.1)
    fig.tight_layout();fig.savefig(out/'bandas_altura.png',dpi=150);plt.close(fig)
    print('Plano dominante:',meta['orientacion']['plano_z_modo'])
    print('Percentiles XYZ:',np.quantile(x,[.001,.01,.5,.99,.999],axis=0))


if __name__=='__main__':main()
