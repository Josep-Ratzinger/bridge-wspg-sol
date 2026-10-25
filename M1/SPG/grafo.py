"""Grafo dirigido por adyacencia Voronoi/Delaunay de los puntos reducidos.

Referencias: WSPG 2022 §3.3.3 y repositorio SPG de Landrieu/Simonovsky.
Implementación propia por agregación; 13 atributos con log-cocientes, perfil
SPG. No garantiza identidad numérica con los descriptores del código original.
"""
import json
import itertools
from pathlib import Path
import numpy as np
import scipy
from scipy.spatial import Delaunay, QhullError
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from .particion import sha256

NOMBRES = ["offset_media_x", "offset_media_y", "offset_media_z",
           "offset_std_x", "offset_std_y", "offset_std_z",
           "log_longitud", "log_superficie", "log_volumen", "log_conteo",
           "centroide_delta_x", "centroide_delta_y", "centroide_delta_z"]


def validar(xyz, etiquetas):
    xyz = np.asarray(xyz, dtype=np.float64)
    etiquetas = np.asarray(etiquetas)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError("Se requieren coordenadas finitas N x 3.")
    if etiquetas.shape != (len(xyz),) or not np.issubdtype(etiquetas.dtype, np.integer) or etiquetas.min() < 0:
        raise ValueError("Asignación de superpuntos inválida.")
    n = int(etiquetas.max()) + 1
    if n > len(xyz) or (np.bincount(etiquetas.astype(np.int64), minlength=n) == 0).any():
        raise ValueError("Identificadores de superpuntos no consecutivos.")
    return xyz, etiquetas.astype(np.int64), n


def aristas_voronoi(xyz, etiquetas):
    """Aristas únicas entre puntos de distintos grupos; no k-NN de centroides."""
    if len(xyz) < 4:
        raise ValueError("Delaunay 3D requiere al menos cuatro puntos.")
    try:
        tri = Delaunay(xyz)
    except QhullError as error:
        raise ValueError("Delaunay 3D falló: geometría degenerada o coplanar; no se altera con ruido artificial.") from error
    if len(tri.coplanar):
        raise ValueError("Delaunay omitió puntos coplanares/duplicados; revisar la entrada antes de construir el grafo.")
    tetra = tri.simplices
    del tri
    bloques = []
    n = np.uint64(len(xyz))
    for a, b in itertools.combinations(range(4), 2):
        u, v = tetra[:, a], tetra[:, b]
        distinto = etiquetas[u] != etiquetas[v]
        u, v = u[distinto], v[distinto]
        codigos = np.minimum(u, v).astype(np.uint64)*n + np.maximum(u, v).astype(np.uint64)
        bloques.append(np.unique(codigos))
    codigos = np.unique(np.concatenate(bloques))
    return (codigos // n).astype(np.uint32), (codigos % n).astype(np.uint32)


def atributos_nodos(xyz, etiquetas, n):
    tamanos = np.bincount(etiquetas, minlength=n)
    orden = np.argsort(etiquetas, kind="stable")
    limites = np.r_[0, np.cumsum(tamanos)]
    centros = np.zeros((n, 3)); formas = np.zeros((n, 4))
    for i in range(n):
        puntos = xyz[orden[limites[i]:limites[i+1]]]
        centros[i] = puntos.mean(axis=0)
        if len(puntos) > 1:
            local = puntos - centros[i]
            valores = np.maximum(np.linalg.eigvalsh(local.T@local/(len(puntos)-1)), 0)[::-1]
            formas[i, :3] = [valores[0], valores[0]*valores[1], np.prod(valores)]
        formas[i, 3] = len(puntos)
    return centros, formas, tamanos


def agregar_conexiones(xyz, etiquetas, u, v):
    xyz, etiquetas, n = validar(xyz, etiquetas)
    u, v = np.asarray(u), np.asarray(v)
    if u.shape != v.shape or u.ndim != 1 or not np.issubdtype(u.dtype, np.integer) or not np.issubdtype(v.dtype, np.integer):
        raise ValueError("Aristas puntuales inválidas.")
    if len(u) and (u.min() < 0 or v.min() < 0 or max(u.max(), v.max()) >= len(xyz)):
        raise ValueError("Índice de arista fuera de rango.")
    # Deduplicar y ordenar contactos puntuales no dirigidos.
    claves = np.minimum(u, v).astype(np.uint64)*np.uint64(len(xyz)) + np.maximum(u, v).astype(np.uint64)
    claves = np.unique(claves)
    u, v = (claves//len(xyz)).astype(np.int64), (claves%len(xyz)).astype(np.int64)
    mascara = etiquetas[u] != etiquetas[v]
    u, v = u[mascara], v[mascara]
    # Orientar cada contacto por identificador de superpunto, y agregar una vez.
    invertir = etiquetas[u] > etiquetas[v]
    a, b = np.where(invertir, v, u), np.where(invertir, u, v)
    claves = etiquetas[a].astype(np.uint64)*np.uint64(n) + etiquetas[b].astype(np.uint64)
    orden = np.argsort(claves)
    claves, a, b = claves[orden], a[orden], b[orden]
    centros, formas, tamanos = atributos_nodos(xyz, etiquetas, n)
    if not len(claves):
        return dict(origen=np.empty(0, np.uint32), destino=np.empty(0, np.uint32),
                    atributos=np.empty((0, 13), np.float32), contactos=np.empty(0, np.uint64),
                    centroides=centros, formas=formas, tamanos_reducidos=tamanos)
    inicios = np.r_[0, np.flatnonzero(np.diff(claves))+1]
    cuentas = np.diff(np.r_[inicios, len(claves)])
    src = (claves[inicios]//n).astype(np.uint32); dst = (claves[inicios]%n).astype(np.uint32)
    delta = xyz[a]-xyz[b]
    medias = np.add.reduceat(delta, inicios, axis=0)/cuentas[:, None]
    # Segundo paso centrado: evita cancelación de varianzas.
    residuo = delta - np.repeat(medias, cuentas, axis=0)
    std = np.sqrt(np.add.reduceat(residuo*residuo, inicios, axis=0)/cuentas[:, None])
    logs = np.log(formas + 1e-10)
    cocientes = logs[src]-logs[dst]
    diferencia = centros[src]-centros[dst]
    ida = np.column_stack((medias, std, cocientes, diferencia))
    vuelta = np.column_stack((-medias, std, -cocientes, -diferencia))
    return dict(origen=np.r_[src, dst], destino=np.r_[dst, src],
                atributos=np.vstack((ida, vuelta)).astype(np.float32), contactos=np.tile(cuentas, 2).astype(np.uint64),
                centroides=centros, formas=formas, tamanos_reducidos=tamanos)


def generar_grafo(reduccion, particion, salida):
    reduccion, particion, salida = map(Path, (reduccion, particion, salida))
    if any((salida/n).exists() for n in ("grafo.npz", "informe.json")):
        raise ValueError("Ya existe el grafo; no se sobrescribe automáticamente.")
    firma_red, firma_part = sha256(reduccion), sha256(particion)
    with np.load(reduccion, allow_pickle=False) as d:
        xyz = d["xyz"]
        procedencia = d['origen'] if 'origen' in d.files else None
    with np.load(particion, allow_pickle=False) as d:
        etiquetas = d["celda_a_superpunto"]
        meta = json.loads(str(d["metadatos"].item()))
    if meta["sha256_reduccion"] != firma_red:
        raise ValueError("La partición no corresponde a la reducción.")
    xyz, etiquetas, n = validar(xyz, etiquetas)
    print(f"Delaunay/Voronoi 3D sobre {len(xyz):,} puntos. Esta fase puede tardar y consumir varios GB de RAM...", flush=True)
    u, v = aristas_voronoi(xyz, etiquetas)
    print(f"Contactos entre grupos: {len(u):,}. Agregando los 13 atributos...", flush=True)
    datos = agregar_conexiones(xyz, etiquetas, u, v)
    if procedencia is not None:
        from .procedencia import resumir_origen
        datos.update(resumir_origen(procedencia,etiquetas))
    if not np.isfinite(datos["atributos"]).all():
        raise ValueError("Se generaron atributos no finitos.")
    red = csr_matrix((np.ones(len(datos["origen"]), dtype=np.uint8), (datos["origen"], datos["destino"])), shape=(n,n))
    componentes = connected_components(red, directed=False, return_labels=False)
    grados = np.bincount(datos["origen"], minlength=n)
    distancias = np.linalg.norm(xyz[u]-xyz[v], axis=1)
    informe = {"superpuntos": n, "conexiones_dirigidas": len(datos["origen"]),
        "contactos_puntuales_no_dirigidos": len(u), "componentes_conexas": int(componentes),
        "superpuntos_aislados": int((grados==0).sum()), "atributos": NOMBRES,
        "adyacencia": "Voronoi por dual Delaunay 3D de todos los puntos reducidos; no centroides ni k-NN",
        "longitud_contactos_p50_p95_max": np.percentile(distancias,[50,95,100]).tolist() if len(u) else [],
        "distancia_maxima": None, "nota_distancia": "sin corte; revisar conexiones largas por vacíos, sin asumir contacto físico",
        "formas_nodos": "lambda1; lambda1*lambda2; lambda1*lambda2*lambda3; cantidad reducida. Covarianza muestral; ceros en singletons",
        "perfil": "13 atributos SPG: media y desviación de offsets, 4 log-cocientes, diferencia de centroides; implementación propia, no identidad numérica con WSPG",
        "epsilon_log": 1e-10, "normalizacion": "sin estandarizar; ajustar scaler solo con entrenamiento en etapa posterior",
        "sha256_reduccion": firma_red, "sha256_particion": firma_part,
        "numpy": np.__version__, "scipy": scipy.__version__}
    salida.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(salida/"grafo.npz", **datos, nombres=np.array(NOMBRES), metadatos=np.array(json.dumps(informe, ensure_ascii=False)))
    (salida/"informe.json").write_text(json.dumps(informe, indent=2, ensure_ascii=False), encoding="utf-8")
    return informe
