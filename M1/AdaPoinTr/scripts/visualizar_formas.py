"""Detalle visual de la selección de formas; no modifica la inferencia."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..src.diagnostico import MODULE
from ..src.completado_automatico import support_context,SEED


def main():
    out=MODULE/'resultados/completado';record=json.loads((out/'ejecucion.json').read_text(encoding='utf8'))
    with np.load(record['input_npz']) as d:x=d['xyz'];plane=json.loads(str(d['metadatos']))['orientacion']['plano_z_modo']
    factor=record['canonical_factor'];context=support_context(x.astype('f8')/factor,plane/factor,SEED)
    xy=x[:,:2]/factor-context['origin_xy'];local=np.c_[xy@context['axis_xy'],xy@context['normal_xy'],x[:,2]/factor]
    new=record['geometry']['beam_profile']
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for j,index in enumerate((0,-1)):
        col=record['geometry']['columns'][index];s=col['s'];d=col['d'];z0,z1=new['levels_z'][0],new['levels_z'][-1]
        p=local[(abs(local[:,0]-s)<1.65)&(abs(local[:,1]-d)<1.3)&(local[:,2]>z0-.5)&(local[:,2]<z1+.6)]
        ax=axes[0,j];ax.scatter(p[:,0]*factor,p[:,2]*factor,s=.5,c='#1984c6',label='Observado')
        side='left_s' if j==0 else 'right_s'
        ax.plot(np.array(new[side])*factor,np.array(new['levels_z'])*factor,color='#f08d23',lw=2.5,label='Nuevo extremo')
        ax.set(title='Extremo '+('izquierdo' if j==0 else 'derecho'),xlabel='Eje del conjunto',ylabel='Z');ax.set_aspect('equal');ax.grid(alpha=.15)
        p=local[(abs(local[:,0]-s)<.35)&(abs(local[:,1]-d)<1.5)&(local[:,2]>z0-.3)&(local[:,2]<z1+.35)]
        ax=axes[1,j];ax.scatter(p[:,1]*factor,p[:,2]*factor,s=.5,c='#1984c6')
        z=np.array(new['levels_z']);center=new['row_center_slope']*s+new['row_center_intercept']+np.array(new['relative_centers_d']);width=np.array(new['depths'])
        outline=np.r_[np.c_[center-width/2,z],np.c_[(center+width/2)[::-1],z[::-1]],[[center[0]-width[0]/2,z[0]]]]
        ax.plot(outline[:,0]*factor,outline[:,1]*factor,c='#f08d23',lw=2.5);ax.set(title='Sección transversal estimada',xlabel='Transversal',ylabel='Z');ax.set_aspect('equal');ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=8);fig.suptitle('Contornos seleccionados desde la nube preparada; dimensiones aproximadas, unidades originales')
    fig.tight_layout();fig.savefig(out/'detalles_formas.png',dpi=150);plt.close(fig)


if __name__=='__main__':main()
