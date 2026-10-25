"""Normalización y supervisión parcial compartidas por entrenamiento e inferencia."""
import numpy as np
import torch
from .datos import MASKS,CLASSES
def centros_relativos(centros):
    low,high=np.percentile(centros,[1,99],axis=0)
    return ((centros-(low+high)/2)/(high-low).clip(.01)).clip(-2,2).astype('f4')

def tensors(d,scaler,device):
    def t(v,dtype=torch.float32): return torch.as_tensor(v,dtype=dtype,device=device)
    x=[t(((d['puntos']-scaler['pm'])/scaler['ps']).clip(-10,10)),t(d['centros']),
       t(d['embedding_habilitado']),t(d['aristas'],torch.long),
       t(((d['atributos_aristas']-scaler['em'])/scaler['es']).clip(-10,10))]
    if 'atributos_nodo' in d:x.append(t(((d['atributos_nodo']-scaler['nm'])/scaler['ns']).clip(-10,10)))
    return x,t(d['histogramas'])

def allowed_matrix(device):
    return torch.tensor([[bool(int(mask)&(1<<c)) for c in range(10)] for mask in MASKS],device=device)

def partial_loss(logits,hist,weights,allowed):
    logp=logits.log_softmax(1)
    # CE exacta para etiquetas unitarias; -log(sum p_clases_permitidas) para
    # etiquetas amplias. No se reparten respuestas ficticias entre subclases.
    terms=torch.stack([-torch.logsumexp(logp[:,m],dim=1) for m in allowed],1)
    w=weights
    mw=(allowed*w[None,:]).sum(1)/allowed.sum(1)
    counts=hist*mw[None,:]
    return (counts*terms).sum()/counts.sum().clamp_min(1)

def resumen(confusion):
    tp=np.diag(confusion); support=confusion.sum(1); union=support+confusion.sum(0)-tp
    iou=np.divide(tp,union,out=np.zeros(10,dtype='f8'),where=union>0)
    # Media sobre clases con referencia exacta, conjunto declarado por split.
    present=support>0
    return dict(mIoU=float(iou[present].mean()) if present.any() else None,
        IoU={CLASSES[k]:float(iou[k]) if present[k] else None for k in range(10)},
        exactitud=float(tp.sum()/confusion.sum()) if confusion.sum() else None,
        matriz_confusion=confusion.tolist(),puntos_etiqueta_exacta=int(confusion.sum()),
        clases_ausentes_no_evaluadas=[CLASSES[k] for k in np.flatnonzero(~present)])
