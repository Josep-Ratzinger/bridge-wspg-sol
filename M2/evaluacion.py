"""Comparación de WSPG y WSPG+SOL con la referencia de Lurín."""
import argparse,json
from pathlib import Path
import numpy as np
from .datos import ROOT,BASE,RESULTADOS,CLASSES,firma,guardar
from .utilidades import resumen
def metricas_lurin(pred,ref,origin,xyz):
    valid=(ref>=0)&(origin==0)
    cm=np.bincount(ref[valid].astype('i8')*10+pred[valid],minlength=100).reshape(10,10)
    report=resumen(cm);gt=cm.sum(1);p=cm.sum(0);tp=np.diag(cm)
    report['recall']={CLASSES[k]:float(tp[k]/gt[k]) if gt[k] else None for k in range(10)}
    report['precision']={CLASSES[k]:float(tp[k]/p[k]) if p[k] else None for k in range(10)}
    # Lurín has a longitudinal X axis; continuity evaluates annotated bins,
    # not a claim of reconstructing a physical railing absent from the cloud.
    rail=valid&(ref==3);lo,hi=np.percentile(xyz[rail,0],[0,100]);station=np.clip(((xyz[:,0]-lo)/max(hi-lo,1e-6)*100).astype(int),0,99)
    side=(xyz[:,1]>np.median(xyz[rail,1])).astype(int);bins=station+100*side
    expected=np.bincount(bins[rail],minlength=200);correct=np.bincount(bins[rail&(pred==3)],minlength=200)
    evaluated=expected>=20;covered=correct>=.5*expected
    report['barandas_tramos_cobertura50']=float(covered[evaluated].mean()) if evaluated.any() else None
    report['barandas_tramos_cobertura90']=float((correct[evaluated]>=.9*expected[evaluated]).mean()) if evaluated.any() else None
    report['barandas_tramos_evaluados']=int(evaluated.sum())
    report['puntos_evaluados']=int(valid.sum());report['referencia']='asistida preliminar, observados; no verdad independiente'
    return report

def no_regresion(candidate,reference,tolerance=0.):
    drops={k:float(candidate['IoU'][k]-reference['IoU'][k]) for k in CLASSES[:7]
        if reference['IoU'][k] is not None and candidate['IoU'][k] is not None and candidate['IoU'][k]+tolerance<reference['IoU'][k]}
    for k in CLASSES[:7]:
        if reference['IoU'][k] is not None and candidate['IoU'][k] is None:drops[k]=-reference['IoU'][k]
    previous=reference['barandas_tramos_cobertura50'];current=candidate['barandas_tramos_cobertura50']
    continuity=previous is None or (current is not None and current>=previous)
    return not drops and continuity,drops


def evaluar(spg=ROOT/'M1/SPG/resultados',referencia=ROOT/'M1/clases/resultados/clases.npz',resultados=RESULTADOS):
    spg=Path(spg);resultados=Path(resultados);partition=spg/'superpuntos/superpuntos.npz'
    with np.load(partition) as f:group=f['celda_a_superpunto']
    with np.load(spg/'entrada/nube_completada.npz') as f:xyz=f['xyz'];origin=f['origen']
    with np.load(referencia) as f:
        ref=f['clase_referencia_preliminar']
        if 'origen' in f.files and not np.array_equal(origin,f['origen']):raise ValueError('Referencia de otra nube')
    if len(ref)!=len(group):raise ValueError('La referencia debe conservar el orden de los puntos de M1')
    reports={}
    for name in ['WSPG','WSPG_SOL']:
        meta=json.loads((resultados/name/'verificacion.json').read_text(encoding='utf8'))
        if meta['entrada_sha256']['particion']!=firma(partition):raise ValueError('Predicción de otro SPG')
        with np.load(resultados/name/'predicciones.npz') as f:pred=f['clase_por_superpunto'][group]
        reports[name]=metricas_lurin(pred,ref,origin,xyz)
    control=json.loads((BASE/'configs/aceptacion.json').read_text(encoding='utf8'))
    same=control['particion_sha256']==firma(partition);safe=None
    if same:safe=no_regresion(reports['WSPG_SOL'],control['umbrales'])[0]
    paired,drops=no_regresion(reports['WSPG_SOL'],reports['WSPG'])
    report=dict(resultados=reports,seleccion='WSPG_SOL',referencia=str(referencia),puntos_evaluados=int(((ref>=0)&(origin==0)).sum()),
        cobertura_observados=float(((ref>=0)&(origin==0)).sum()/(origin==0).sum()),puntos_exportados=len(xyz),
        comparacion_pareada=True,Lurin_usado_para_entrenar=False,Lurin_usado_para_aceptar_version=True,
        prueba_independiente=False,no_regresion_control=safe,no_regresion_pareada=paired,regresiones_pareadas=drops,
        control_no_regresion=control['umbrales'] if same else None)
    guardar(resultados/'evaluacion.json',report)
    for name,r in reports.items():print(name,'mIoU Lurín:',round(r['mIoU']*100,2),'%',flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spg',type=Path,default=ROOT/'M1/SPG/resultados')
    p.add_argument('--referencia',type=Path,default=ROOT/'M1/clases/resultados/clases.npz');p.add_argument('--resultados',type=Path,default=RESULTADOS)
    a=p.parse_args();evaluar(a.spg,a.referencia,a.resultados)


if __name__=='__main__':main()
