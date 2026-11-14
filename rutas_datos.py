"""Ubicación de las entradas y datasets de WSPG+SOL."""
import os
from pathlib import Path

DATOS = Path(os.environ.get(
    'WSPG_SOL_DATA',
    r'D:\USUARIO\Documentos\Nube de puntos tesis\WSPG_SOL_Data',
))
RECORTE = DATOS / 'M1/02_limpieza_y_preparacion/recorte_manual'
FUENTES_M2 = DATOS / 'M2/02_fuentes_reales'
