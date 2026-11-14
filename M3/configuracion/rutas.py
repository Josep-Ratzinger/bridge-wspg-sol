"""Entradas y salidas del experimento de aumento sintético."""
from pathlib import Path

from rutas_datos import DATOS

LOTES = DATOS / 'M3/01_lotes_sinteticos'
EXPERIMENTO = Path(__file__).resolve().parents[1] / 'experimento'
RESULTADOS = EXPERIMENTO.parent / 'evaluacion/resultados'
