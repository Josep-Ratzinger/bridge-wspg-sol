"""Almacenamiento de M3. M3_DATA_ROOT permite cambiar de disco."""
import os
from pathlib import Path

DATOS = Path(os.environ.get(
    'M3_DATA_ROOT',
    r'D:\USUARIO\Documentos\Nube de puntos tesis\M3_Data',
))
