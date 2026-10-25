"""Rutas, clases y formatos de la versión final de M2."""
import hashlib
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'M2'
CORPUS=BASE/'datos/corpus'
MODELOS=BASE/'modelos'
RESULTADOS=BASE/'resultados'
CLASSES=['columnas_pilares','vigas_cabezales','tablero','barandas_laterales','pista_accesos','separador_central','superficie_inferior_suelo_base','estribos','vegetacion','senales']
MASKS=np.array([1<<k for k in range(10)]+[80,6],dtype='i4')
PALETA=np.array([[230,65,60],[245,155,35],[45,125,225],[150,70,200],[55,175,105],[35,200,215],[155,105,65],[200,170,120],[40,110,40],[245,210,30]],dtype='u1')
SOURCES={'semanticbridge':'https://github.com/mvg-inatech/3d_bridge_segmentation','uav':'https://zenodo.org/records/18531104','tipologias':'https://www.fhwa.dot.gov/bridge/Lrfd/us_ds8.cfm','secciones_apoyos':'https://www.dot.ny.gov/main/business-center/engineering/cadd-info/drawings/bridge-detail-sheets/pr-piers','curvaturas':'https://doi.org/10.1111/mice.13384'}
def guardar(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf8')
def firma(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()
