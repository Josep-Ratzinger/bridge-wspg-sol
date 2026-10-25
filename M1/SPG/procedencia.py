"""Procedencia observada/estimada; nunca etiquetas semánticas de referencia."""
import numpy as np


def resumir_origen(origen,etiquetas):
    origen=np.asarray(origen);etiquetas=np.asarray(etiquetas)
    if origen.shape!=etiquetas.shape or origen.ndim!=1 or not len(origen):raise ValueError('Origen y grupos deben tener igual tamaño no vacío')
    if not np.issubdtype(origen.dtype,np.integer) or not np.isin(origen,[0,1]).all():raise ValueError('Origen solo admite 0 observado y 1 estimado')
    if not np.issubdtype(etiquetas.dtype,np.integer) or etiquetas.min()<0:raise ValueError('Grupos inválidos')
    n=int(etiquetas.max())+1
    observados=np.bincount(etiquetas[origen==0].astype('i8'),minlength=n).astype('u8')
    estimados=np.bincount(etiquetas[origen==1].astype('i8'),minlength=n).astype('u8')
    total=observados+estimados
    if (total==0).any():raise ValueError('Grupos no consecutivos')
    return {'observados_por_superpunto':observados,'estimados_por_superpunto':estimados,
            'fraccion_estimada':(estimados/total).astype('f4')}
