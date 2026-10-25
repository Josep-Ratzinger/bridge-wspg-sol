"""Empaqueta puntos reducidos y geometría para la futura red WSPG."""
import json
from pathlib import Path
import numpy as np
from .particion import sha256


def muestrear_grupos(xyz, geof, etiquetas, cantidad=128, minimo=25, semilla=42):
    xyz, geof, etiquetas = map(np.asarray, (xyz, geof, etiquetas))
    if xyz.ndim != 2 or xyz.shape[1] != 3 or geof.shape != (len(xyz), 5):
        raise ValueError("Se requieren XYZ y cinco atributos geométricos por punto.")
    if not len(xyz) or not np.isfinite(xyz).all() or not np.isfinite(geof).all():
        raise ValueError("Las entradas deben ser finitas y no vacías.")
    if etiquetas.shape != (len(xyz),) or not np.issubdtype(etiquetas.dtype, np.integer) or etiquetas.min() < 0:
        raise ValueError("Asignación de superpuntos inválida.")
    if cantidad < 1 or minimo < 1:
        raise ValueError("Cantidad y mínimo deben ser positivos.")
    n = int(etiquetas.max()) + 1
    if n > len(xyz):
        raise ValueError("Identificadores de superpuntos no consecutivos.")
    tamanos = np.bincount(etiquetas.astype(np.int64), minlength=n)
    if (tamanos == 0).any():
        raise ValueError("Los identificadores deben ser consecutivos desde cero.")
    orden = np.argsort(etiquetas, kind="stable")
    limites = np.r_[0, np.cumsum(tamanos)]
    rng = np.random.default_rng(semilla)
    indices = np.empty((n, cantidad), dtype=np.uint32)
    centros = np.empty((n, 3), dtype=np.float64)
    escalas = np.empty(n, dtype=np.float64)
    puntos = np.empty((n, cantidad, 8), dtype=np.float32)
    for i in range(n):
        grupo = orden[limites[i]:limites[i+1]]
        centros[i] = xyz[grupo].mean(axis=0)
        escalas[i] = max(float(np.ptp(xyz[grupo], axis=0).max()), 1e-12)
        seleccion = rng.choice(grupo, cantidad, replace=len(grupo) < cantidad)
        indices[i] = seleccion
        # Adaptación explícita: XYZ centrado y dividido por extensión máxima
        # del grupo entero; los otros cinco atributos se conservan sin ponderar.
        puntos[i, :, :3] = (xyz[seleccion] - centros[i]) / escalas[i]
        puntos[i, :, 3:] = geof[seleccion]
    return dict(puntos=puntos, indices_reducidos=indices, centros=centros, escalas=escalas,
                tamanos_reducidos=tamanos.astype(np.uint64), embedding_habilitado=tamanos >= minimo)


def preparar_entrada(reduccion, geometria, particion, salida, semilla=42):
    reduccion, geometria, particion, salida = map(Path, (reduccion, geometria, particion, salida))
    destino = salida / "entrada_superpuntos.npz"
    if destino.exists():
        raise ValueError("La entrada ya existe; no se sobrescribe automáticamente.")
    firmas = {"reduccion": sha256(reduccion), "geometria": sha256(geometria), "particion": sha256(particion)}
    with np.load(reduccion, allow_pickle=False) as d:
        xyz = d["xyz"]
        procedencia = d['origen'] if 'origen' in d.files else None
    with np.load(geometria, allow_pickle=False) as d:
        geof = d["geof"]
        meta_geo = json.loads(str(d["metadatos"].item()))
        nombres = d["nombres"].tolist()
    with np.load(particion, allow_pickle=False) as d:
        etiquetas = d["celda_a_superpunto"]
        meta_part = json.loads(str(d["metadatos"].item()))
    if meta_geo["sha256_entrada"] != firmas["reduccion"] or meta_part["sha256_reduccion"] != firmas["reduccion"] or meta_part["sha256_geometria"] != firmas["geometria"]:
        raise ValueError("Las entradas no pertenecen a la misma ejecución.")
    if nombres != ["linealidad", "planaridad", "dispersion", "verticalidad", "elevacion"]:
        raise ValueError("Orden de atributos inesperado.")
    contenido = muestrear_grupos(xyz, geof, etiquetas, semilla=semilla)
    if procedencia is not None:
        from .procedencia import resumir_origen
        contenido.update(resumir_origen(procedencia,etiquetas))
        contenido['origen_muestreado'] = procedencia[contenido['indices_reducidos']]
    informe = {"superpuntos": len(contenido["puntos"]), "puntos_por_grupo": 128,
        "atributos": ["x_local", "y_local", "z_local"] + nombres, "semilla": semilla,
        "muestreo": "uniforme sin reemplazo si N>=128; con reemplazo si N<128",
        "normalizacion_xyz": "(xyz-centroide_grupo)/extension_maxima_grupo; adaptación propia",
        "minimo_embedding": 25, "grupos_pequenos": int((~contenido["embedding_habilitado"]).sum()),
        "regla_grupos_pequenos": "conservados; máscara para que la futura red asigne embedding cero, no implementada aún",
        "firmas_sha256": firmas, "estado": "entrada preparada para la red; no etiquetas semánticas",
        "procedencia_conservada": procedencia is not None,
        "origen_es_atributo_de_red": False}
    salida.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destino, **contenido, metadatos=np.array(json.dumps(informe, ensure_ascii=False)))
    return informe
