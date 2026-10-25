"""Figuras de la versión final: referencia, WSPG y WSPG+SOL."""
import argparse,json
import numpy as np
from plyfile import PlyData
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from .datos import ROOT,BASE,RESULTADOS,PALETA
PALETTE=PALETA/255
CLASSES=['columnas','vigas/cabezales','tablero','barandas','accesos','separador','suelo/base','estribos','vegetación','señales']
def comparacion():
    report={'WSPG':'WSPG','WSPG_SOL':'WSPG_SOL'}
    names=[report['WSPG'],report['WSPG_SOL']]
    predicted=[PlyData.read(str(RESULTADOS/n/'puente_etiquetado.ply'))['vertex'].data for n in names]
    xyz=np.column_stack([predicted[0][k] for k in ['x','y','z']])
    with np.load(ROOT/'M1/clases/resultados/clases.npz') as f:assisted=f['clase_propuesta'];origin=f['origen']
    arrays=[assisted]+[v['clase_predicha'] for v in predicted]
    titles=['Anotación asistida previa · evaluación provisional','WSPG · contexto estructural aprendido','WSPG + SOL · relaciones explícitas, comparación pareada']
    fig,axes=plt.subplots(3,2,figsize=(16,11));rng=np.random.default_rng(42)
    for col in range(2):
        region=(xyz[:,0]>5.8)&(xyz[:,0]<10.7) if col else np.ones(len(xyz),bool);indices=[]
        for value,limit in [(0,120000),(1,80000)]:
            ids=np.flatnonzero(region&(origin==value));indices.append(rng.choice(ids,min(len(ids),limit),replace=False))
        idx=np.concatenate(indices);a,b=(1,2) if col else (0,1)
        for row in range(3):
            labels=arrays[row][idx];color=np.full((len(idx),3),.7);valid=labels>=0;color[valid]=PALETTE[labels[valid]]
            ax=axes[row,col];ax.scatter(xyz[idx,a],xyz[idx,b],c=color,s=.15,rasterized=True);ax.set_aspect('equal')
            ax.set_title(titles[row]+(' · apoyos YZ' if col else ' · planta XY'),fontsize=10)
            ax.set_xlabel('Y' if col else 'X');ax.set_ylabel('Z' if col else 'Y')
    fig.suptitle('Lurín · todos los puntos y el completado conservados\n'+'Entrenamiento externo; aceptación en Lurín con referencia asistida, no test independiente',fontsize=13)
    fig.legend(handles=[Patch(color=PALETTE[k],label=c) for k,c in enumerate(CLASSES)],loc='lower center',ncol=5,fontsize=8)
    fig.tight_layout(rect=[0,.06,1,.95]);fig.savefig(RESULTADOS/'comparacion.png',dpi=140);plt.close(fig)
    region=(xyz[:,1]>-12.5)&(xyz[:,1]<-9.5)&(xyz[:,0]>-25)&(xyz[:,0]<38)&(xyz[:,2]>-.5)&(xyz[:,2]<3.)
    idx=np.flatnonzero(region);idx=rng.choice(idx,min(len(idx),150000),replace=False)
    fig,axes=plt.subplots(3,1,figsize=(14,7))
    for row,ax in enumerate(axes):
        labels=arrays[row][idx];color=np.full((len(idx),3),.7);valid=labels>=0;color[valid]=PALETTE[labels[valid]]
        ax.scatter(xyz[idx,0],xyz[idx,2],c=color,s=.3,rasterized=True);ax.set_aspect('equal');ax.set_title(titles[row],fontsize=10)
    fig.suptitle('Detalle de barandas · solo ventana visual, PLY completo',fontsize=12)
    fig.legend(handles=[Patch(color=PALETTE[k],label=c) for k,c in enumerate(CLASSES)],loc='lower center',ncol=5,fontsize=8)
    fig.tight_layout(rect=[0,.07,1,.95]);fig.savefig(RESULTADOS/'barandas.png',dpi=140);plt.close(fig)

def main():
 comparacion()
if __name__=='__main__':main()
