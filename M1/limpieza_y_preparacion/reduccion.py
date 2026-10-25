"""Reducción por centroides de celdas; no realiza partición en superpuntos."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def agrupar_celdas(xyz: np.ndarray, tamano: float):
    """Cuadrícula anclada en (0, 0, 0), con intervalos [i*t, (i+1)*t).

    Devuelve centroides, índice de celda por punto y población de cada celda.
    El índice conserva el orden de los puntos de entrada.
    """
    if not np.isfinite(tamano) or tamano <= 0:
        raise ValueError("El tamaño de celda debe ser finito y mayor que cero.")
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0:
        raise ValueError("Se requiere una matriz no vacía de coordenadas N × 3.")
    if not np.isfinite(xyz).all():
        raise ValueError("Hay coordenadas no finitas; no se eliminan automáticamente.")
    indices = np.floor(xyz / tamano)
    if np.max(np.abs(indices)) >= 2**62:
        raise ValueError("El tamaño de celda es demasiado pequeño para estas coordenadas.")
    indices = indices.astype(np.int64)
    indices -= indices.min(axis=0)
    dimensiones = [int(v) + 1 for v in indices.max(axis=0)]
    if math.prod(dimensiones) >= 2**64:
        raise ValueError("La cuadrícula excede el rango de índices admitido.")
    indices = indices.astype(np.uint64)
    claves = (indices[:, 0] * dimensiones[1] + indices[:, 1]) * dimensiones[2] + indices[:, 2]
    del indices
    _, correspondencia, conteos = np.unique(claves, return_inverse=True, return_counts=True)
    correspondencia = correspondencia.reshape(-1)
    if len(conteos) >= 2**32:
        raise ValueError("Hay demasiadas celdas para la correspondencia de 32 bits.")
    centroides = np.column_stack([
        np.bincount(correspondencia, weights=xyz[:, eje], minlength=len(conteos)) / conteos
        for eje in range(3)
    ]).astype(np.float32)
    return centroides, correspondencia.astype(np.uint32), conteos.astype(np.uint32)


def reducir_nube(archivo: Path, salida: Path, tamano: float) -> dict:
    """Guarda una nube de cálculo y la correspondencia completa con el original.

    No promedia normales: se calculará la geometría sobre la nube reducida.
    Los colores son medias por celda; no participan en esta agrupación espacial.
    """
    archivo, salida = Path(archivo), Path(salida)
    destino_ply = salida / "nube_reducida.ply"
    destino_npz = salida / "reduccion.npz"
    for destino in (destino_ply, destino_npz):
        if destino.exists():
            raise ValueError(f"Ya existe {destino}. No se sobrescriben resultados automáticamente.")
    ply = PlyData.read(str(archivo))
    if "vertex" not in ply:
        raise ValueError("El archivo no contiene el elemento vertex.")
    vertices = ply["vertex"].data
    nombres = set(vertices.dtype.names or ())
    if not {"x", "y", "z"}.issubset(nombres):
        raise ValueError("Faltan coordenadas x, y o z.")
    xyz = np.column_stack([vertices[eje] for eje in ("x", "y", "z")])
    print(f"Agrupando {len(xyz):,} puntos en celdas de {tamano:g} unidades...", flush=True)
    centroides, mapa, conteos = agrupar_celdas(xyz, tamano)
    rgb = None
    if {"red", "green", "blue"}.issubset(nombres):
        rgb = np.column_stack([
            np.rint(np.bincount(mapa, weights=vertices[c], minlength=len(conteos)) / conteos)
            for c in ("red", "green", "blue")
        ]).astype(np.uint8)
    firma = hashlib.sha256()
    with archivo.open("rb") as f:
        for bloque in iter(lambda: f.read(8 * 1024 * 1024), b""):
            firma.update(bloque)
    informe = {
        "metodo": "centroide por celda; no son superpuntos",
        "tamano_celda": tamano,
        "unidades": "las mismas del archivo de entrada; escala métrica estimada externamente",
        "origen_cuadricula": [0, 0, 0],
        "archivo_entrada": archivo.name,
        "sha256_entrada": firma.hexdigest(),
        "puntos_originales": len(vertices),
        "puntos_reducidos": len(centroides),
        "porcentaje_conservado": 100 * len(centroides) / len(vertices),
        "numpy": np.__version__,
    }
    assert int(conteos.sum(dtype=np.uint64)) == len(vertices)
    campos = [(e, "<f4") for e in ("x", "y", "z")]
    if rgb is not None:
        campos += [(c, "u1") for c in ("red", "green", "blue")]
    datos = np.empty(len(centroides), dtype=campos)
    for i, eje in enumerate(("x", "y", "z")):
        datos[eje] = centroides[:, i]
    if rgb is not None:
        for i, color in enumerate(("red", "green", "blue")):
            datos[color] = rgb[:, i]
    salida.mkdir(parents=True, exist_ok=True)
    contenido = dict(xyz=centroides, punto_a_celda=mapa, conteos=conteos,
                     metadatos=np.array(json.dumps(informe, ensure_ascii=False)))
    if rgb is not None:
        contenido["rgb"] = rgb
    np.savez_compressed(destino_npz, **contenido)
    PlyData([PlyElement.describe(datos, "vertex")], text=False, byte_order="<").write(str(destino_ply))
    return informe
