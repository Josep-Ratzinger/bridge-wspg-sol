"""Pureza provisional sobre anotaciones asistidas, excluyendo lo completado."""
import json
import numpy as np
from .anotacion import MODULE,PROJECT,sha256
from M1.SPG.pureza import evaluar_pureza


def evaluar():
    cfg=json.loads((MODULE/'configs/lurin.json').read_text(encoding='utf8'));classes=[c['id'] for c in cfg['clases'] if c['id']>=0]
    source=PROJECT/'M1/SPG/resultados/entrada/nube_completada.npz';part=PROJECT/'M1/SPG/resultados/superpuntos/superpuntos.npz'
    ref=MODULE/'datos/referencia_preliminar.ply';out=MODULE/'resultados/pureza_preliminar'
    report=evaluar_pureza(source,part,ref,out)
    report.update({'fronteras_ambiguas_no_evaluadas':True,'propuestas_automaticas_no_son_ground_truth':True,
                   'lectura':'diagnóstico preliminar solo sobre regiones observadas anotadas; no extrapolar al puente completo',
                   'referencia_metodo':'regiones revisadas en vistas y fotogramas, ampliadas con correcciones visuales del usuario; fronteras no verificadas punto a punto',
                   'superpuntos_no_usados_para_asignar_clases':True})
    with np.load(MODULE/'resultados/clases.npz') as d:ref_classes=d['clase_referencia_preliminar'];origin=d['origen']
    with np.load(part) as d:groups=d['celda_a_superpunto']
    with np.load(out/'pureza.npz') as d:
        dominant=np.full(int(groups.max())+1,-1,dtype='i2');dominant[d['grupo_id']]=d['clase_mayoritaria']
    report['coincidencia_con_mayoria_por_clase']={str(k):float((dominant[groups[ref_classes==k]]==k).mean()) if (ref_classes==k).any() else None for k in classes}
    report['puntos_referencia_por_clase']={str(k):int((ref_classes==k).sum()) for k in classes}
    report['clases_presentes']=[k for k in classes if (ref_classes==k).any()]
    report['taxonomia']=cfg['clases']
    report['spg_sin_recalcular']=True
    report['cambio_de_taxonomia_y_cobertura_no_es_mejora_de_particion']=True
    report['porcentaje_puntos_observados_no_etiquetados']=float(((ref_classes<0)&(origin==0)).sum()/(origin==0).sum())
    (out/'pureza.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8')
    return report
