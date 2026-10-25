"""Inspección de las propuestas y cobertura, independiente de los colores SPG."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from .anotacion import MODULE,PROJECT


def main():
    cfg=json.loads((MODULE/'configs/lurin.json').read_text(encoding='utf8'))
    with np.load(PROJECT/'M1/SPG/resultados/entrada/nube_completada.npz') as d:x=d['xyz'];origin=d['origen']
    with np.load(MODULE/'resultados/clases.npz') as d:labels=d['clase_propuesta'];reference=d['clase_referencia_preliminar']
    colors=np.full((len(x),3),145,dtype='u1')
    for c in cfg['clases']:colors[labels==c['id']]=c['rgb']
    rng=np.random.default_rng(241171);ids=rng.choice(len(x),250000,replace=False)
    fig,axes=plt.subplots(3,1,figsize=(17,12))
    for ax,inds,a,b,title in [(axes[0],ids,0,1,'Clases propuestas por punto: planta'),(axes[1],ids,0,2,'Clases propuestas por punto: vista longitudinal'),
                               (axes[2],np.flatnonzero(abs(x[:,0]-8.2)<1.1),1,2,'Columnas y cabezal: corte transversal')]:
        ax.scatter(x[inds,a],x[inds,b],s=.25,c=colors[inds]/255);ax.set_aspect('equal');ax.set_title(title);ax.grid(alpha=.12)
        ax.set_xlabel(('X','Y','Z')[a]);ax.set_ylabel(('X','Y','Z')[b])
    legend=[Line2D([],[],linestyle='',marker='o',color=np.array(c['rgb'])/255,label=c['nombre'].replace('_',' ')) for c in cfg['clases']]
    fig.legend(handles=legend,loc='lower center',ncol=4);fig.suptitle('Anotación asistida corregida con el usuario; gris sin asignar. Referencia preliminar.')
    fig.tight_layout(rect=[0,.065,1,.96]);fig.savefig(MODULE/'resultados/revision/clases_propuestas.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(16,7));keep=(reference[ids]>=0)&(origin[ids]==0)
    for ax,a,b in [(axes[0],0,1),(axes[1],0,2)]:
        ax.scatter(x[ids,a],x[ids,b],s=.15,color='#cccccc');selected=ids[keep]
        ax.scatter(x[selected,a],x[selected,b],s=.3,c=colors[selected]/255);ax.set_aspect('equal');ax.grid(alpha=.1)
    fig.suptitle('Cobertura de referencia PRELIMINAR: solo regiones observadas reconocibles; gris no evaluado')
    fig.tight_layout();fig.savefig(MODULE/'resultados/revision/cobertura_referencia.png',dpi=140);plt.close(fig)


if __name__=='__main__':main()
