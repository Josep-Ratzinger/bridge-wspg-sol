"""Lectura de coordenadas y comprobaciones de entrada, sin alterar el PLY."""

from pathlib import Path

import numpy as np
from plyfile import PlyData


def revisar_nube(archivo: Path) -> dict:
    """Resume la nube por bloques para limitar el uso de memoria."""
    ply = PlyData.read(str(archivo))
    if "vertex" not in ply:
        raise ValueError("El PLY no contiene el elemento vertex.")
    puntos = ply["vertex"].data
    campos = set(puntos.dtype.names or ())
    if not {"x", "y", "z"}.issubset(campos):
        raise ValueError("Faltan coordenadas x, y o z.")
    if len(puntos) == 0:
        raise ValueError("La nube está vacía.")
    minimos = np.full(3, np.inf)
    maximos = np.full(3, -np.inf)
    no_finitos = 0
    for inicio in range(0, len(puntos), 250_000):
        bloque = puntos[inicio:inicio + 250_000]
        xyz = np.column_stack([bloque[eje] for eje in ("x", "y", "z")])
        validos = np.isfinite(xyz).all(axis=1)
        no_finitos += int((~validos).sum())
        if validos.any():
            minimos = np.minimum(minimos, xyz[validos].min(axis=0))
            maximos = np.maximum(maximos, xyz[validos].max(axis=0))
    if not np.isfinite(minimos).all():
        raise ValueError("La nube no contiene coordenadas finitas.")
    return {
        "puntos": len(puntos),
        "no_finitos": no_finitos,
        "rgb": {"red", "green", "blue"}.issubset(campos),
        "minimos": minimos.tolist(),
        "maximos": maximos.tolist(),
    }
