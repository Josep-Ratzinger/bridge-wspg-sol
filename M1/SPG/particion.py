"""Partición geométrica con cut-pursuit-py 1.0.12 (solucionador alternativo).

Referencia WSPG: Yang et al. 2022, secciones 3.3.2 y 4.1-4.2.2.1.
Cinco atributos; elevación z-minmax propia. Peso verticalidad 1 por defecto.
Pesos de distancia: partition/partition.py de loicland/superpoint_graph.
No reproduce el binario libcp original ni utiliza etiquetas semánticas.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from plyfile import PlyData, PlyElement
import cut_pursuit_py


def sha256(archivo):
    firma = hashlib.sha256()
    with Path(archivo).open("rb") as f:
        for parte in iter(lambda: f.read(8 * 1024 * 1024), b""):
            firma.update(parte)
    return firma.hexdigest()


def grafo_vecinos(xyz, vecinos=10, bloque=10000):
    """Lista k-NN agrupada por origen, requisito del envoltorio instalado.

    Se conservan ambos sentidos cuando son vecinos recíprocos: cada entrada
    aporta un término a la penalización, como en la lista k-NN de referencia.
    No se aplica distancia máxima ni se borran puntos aislados.
    """
    xyz = np.ascontiguousarray(xyz, dtype=np.float64)
    n = len(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("Se requieren coordenadas finitas N × 3.")
    if vecinos < 1 or vecinos >= n or bloque < 1:
        raise ValueError("Vecinos debe estar entre 1 y N-1; bloque debe ser positivo.")
    if n * vecinos >= 2**32:
        raise ValueError("El grafo excede los índices de 32 bits del solucionador.")
    arbol = cKDTree(xyz)
    origen = np.repeat(np.arange(n, dtype=np.uint32), vecinos)
    destino = np.empty(n * vecinos, dtype=np.uint32)
    distancias = np.empty(n * vecinos, dtype=np.float32)
    for inicio in range(0, n, bloque):
        fin = min(inicio + bloque, n)
        dist, ind = arbol.query(xyz[inicio:fin], k=vecinos + 1, workers=1)
        propio = ind == np.arange(inicio, fin)[:, None]
        orden = np.argsort(propio, axis=1, kind="stable")[:, :vecinos]
        destino[inicio * vecinos:fin * vecinos] = np.take_along_axis(ind, orden, axis=1).ravel()
        distancias[inicio * vecinos:fin * vecinos] = np.take_along_axis(dist, orden, axis=1).ravel()
    return origen, destino, distancias


def resolver_particion(geof, origen, destino, distancias, regularizacion=0.03,
                       peso_verticalidad=1.0, verbose=True):
    """Suma de pérdidas cuadráticas y penalización l0 sobre el grafo.

    Peso final por entrada: regularizacion / (1 + distancia / media_distancias).
    El envoltorio de la versión 1.0.12 NO multiplica el vector de pesos por
    reg_strength: ese argumento es solo el peso homogéneo alternativo cuando
    no hay vector. Por ello incorporamos regularizacion en los pesos y pasamos
    1.0 al argumento reg_strength. No se modifica la biblioteca instalada.
    """
    atributos = np.array(geof, dtype=np.float32, order="C", copy=True)
    if atributos.ndim != 2 or atributos.shape[1] != 5 or not np.isfinite(atributos).all():
        raise ValueError("Se requieren cinco atributos geométricos finitos por punto.")
    if not np.isfinite(regularizacion) or regularizacion <= 0:
        raise ValueError("La regularización debe ser finita y positiva.")
    if not np.isfinite(peso_verticalidad) or peso_verticalidad <= 0:
        raise ValueError("El peso de verticalidad debe ser finito y positivo.")
    origen = np.ascontiguousarray(origen, dtype=np.uint32)
    destino = np.ascontiguousarray(destino, dtype=np.uint32)
    distancias = np.asarray(distancias, dtype=np.float32)
    if origen.ndim != 1 or destino.shape != origen.shape or distancias.shape != origen.shape or len(origen) == 0:
        raise ValueError("Las listas de aristas y distancias deben tener la misma longitud no nula.")
    if origen.max() >= len(atributos) or destino.max() >= len(atributos):
        raise ValueError("Hay índices de aristas fuera de la nube.")
    if np.any(origen[1:] < origen[:-1]):
        raise ValueError("Las aristas deben estar agrupadas en orden de origen.")
    if not np.isfinite(distancias).all() or np.any(distancias < 0):
        raise ValueError("Las distancias deben ser finitas y no negativas.")
    media = float(distancias.mean(dtype=np.float64))
    if media <= 0:
        raise ValueError("El grafo no tiene extensión espacial.")
    pesos = np.asarray(regularizacion / (1.0 + distancias / media), dtype=np.float32)
    atributos[:, 3] *= peso_verticalidad
    etiquetas = cut_pursuit_py.perform_cut_pursuit(
        reg_strength=1.0, D=5, pc_vec=atributos, edge_weights=pesos,
        Eu=origen, Ev=destino, verbose=verbose,
    )
    etiquetas = np.asarray(etiquetas)
    if etiquetas.shape != (len(atributos),):
        raise ValueError("El solucionador devolvió una partición de tamaño incorrecto.")
    # Garantizar que cada superpunto es conexo en el grafo, incluso si el
    # solucionador asigna un mismo identificador a regiones desconectadas.
    mismo = etiquetas[origen] == etiquetas[destino]
    interno = csr_matrix((np.ones(int(mismo.sum()), dtype=np.uint8),
                          (origen[mismo], destino[mismo])), shape=(len(atributos), len(atributos)))
    numero, etiquetas_conexas = connected_components(interno, directed=False)
    divisiones = int(numero - len(np.unique(etiquetas)))
    return etiquetas_conexas.astype(np.uint32), media, divisiones


def generar_superpuntos(reduccion, geometria, salida, vecinos=10,
                        regularizacion=0.03, peso_verticalidad=1.0):
    """Exporta la partición, su correspondencia original y una vista por colores."""
    reduccion, geometria, salida = Path(reduccion), Path(geometria), Path(salida)
    for nombre in ("superpuntos.npz", "superpuntos.ply"):
        if (salida / nombre).exists():
            raise ValueError(f"Ya existe {salida / nombre}. No se sobrescribe automáticamente.")
    with np.load(geometria, allow_pickle=False) as datos:
        geof = datos["geof"]
        meta_geo = json.loads(str(datos["metadatos"].item()))
        nombres = datos["nombres"].tolist()
        validos = datos["valido"]
    if nombres != ["linealidad", "planaridad", "dispersion", "verticalidad", "elevacion"]:
        raise ValueError("Se requieren 5 atributos, incluida elevacion. Recalcula geometria con --elevacion z-minmax.")
    firma_reduccion = sha256(reduccion)
    if meta_geo["sha256_entrada"] != firma_reduccion:
        raise ValueError("La geometría no corresponde al archivo de reducción indicado.")
    with np.load(reduccion, allow_pickle=False) as datos:
        xyz = datos["xyz"]
        mapa = datos["punto_a_celda"]
        conteos = datos["conteos"]
        procedencia = datos["origen"] if "origen" in datos.files else None
    if len(xyz) != len(geof) or validos.shape != (len(xyz),) or not validos.all():
        raise ValueError("La geometría tiene tamaño incorrecto o contiene vecindarios degenerados.")
    if mapa.ndim != 1 or len(mapa) == 0 or not np.issubdtype(mapa.dtype, np.integer) or mapa.min() < 0 or mapa.max() >= len(xyz):
        raise ValueError("La correspondencia con los puntos originales no es válida.")
    if conteos.shape != (len(xyz),) or not np.array_equal(np.bincount(mapa, minlength=len(xyz)), conteos):
        raise ValueError("Los conteos no coinciden con la correspondencia de reducción.")
    print(f"Construyendo grafo: {len(xyz):,} puntos y {vecinos} vecinos por punto...", flush=True)
    origen, destino, distancias = grafo_vecinos(xyz, vecinos)
    print(f"Entradas del grafo: {len(origen):,}. Ejecutando cut pursuit...", flush=True)
    etiquetas, media, divisiones = resolver_particion(
        geof, origen, destino, distancias, regularizacion, peso_verticalidad,
    )
    numero = int(etiquetas.max()) + 1
    originales = etiquetas[mapa]
    tamanos_reducidos = np.bincount(etiquetas, minlength=numero).astype(np.uint64)
    tamanos_originales = np.bincount(originales, minlength=numero).astype(np.uint64)
    if int(tamanos_originales.sum()) != len(mapa):
        raise ValueError("La partición no conserva el total de puntos originales.")
    informe = {
        "superpuntos": numero, "puntos_reducidos": len(xyz), "puntos_originales": len(mapa),
        "vecinos_grafo": vecinos, "entradas_grafo": len(origen), "distancia_media_aristas": media,
        "regularizacion": regularizacion, "peso_verticalidad": peso_verticalidad,
        "perfil": "WSPG 2022 adaptado; no reproducción exacta",
        "elevacion": meta_geo.get("elevacion", {}),
        "hilos_openmp": __import__("os").environ.get("OMP_NUM_THREADS", "no fijado"),
        "lambda_distancia": 1.0, "radio_maximo_grafo": None,
        "pesos": "regularizacion/(1+distancia/media_distancias); reciprocidad conservada",
        "solucionador": "cut-pursuit-py", "version": cut_pursuit_py.__version__,
        "limite_iteraciones_biblioteca": 20, "tolerancia_biblioteca": 1e-4,
        "semilla_solucionador": "no expuesta por la biblioteca; no se garantiza partición idéntica",
        "semilla_colores": 42, "numpy": np.__version__, "scipy": scipy.__version__,
        "implementacion": "adaptación con solucionador alternativo, no reproducción exacta de libcp original",
        "divisiones_para_conectividad": divisiones,
        "sha256_reduccion": firma_reduccion, "sha256_geometria": sha256(geometria),
        "geometria": meta_geo,
        "mediana_puntos_reducidos_por_superpunto": float(np.median(tamanos_reducidos)),
        "superpuntos_con_un_punto_reducido": int((tamanos_reducidos == 1).sum()),
    }
    salida.mkdir(parents=True, exist_ok=True)
    print(f"Guardando {numero:,} superpuntos...", flush=True)
    extra = {}
    if procedencia is not None:
        from .procedencia import resumir_origen
        extra = resumir_origen(procedencia, etiquetas)
        informe['procedencia'] = 'conteos observados/estimados por superpunto; no etiquetas semánticas'
    np.savez_compressed(salida / "superpuntos.npz", celda_a_superpunto=etiquetas,
                        punto_a_superpunto=originales, tamanos_reducidos=tamanos_reducidos,
                        tamanos_originales=tamanos_originales,
                        **extra, metadatos=np.array(json.dumps(informe, ensure_ascii=False)))
    colores = np.random.default_rng(42).integers(40, 256, size=(numero, 3), dtype=np.uint8)
    campos = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
              ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("superpunto", "<u4")]
    if procedencia is not None:campos += [('origen','u1')]
    puntos = np.empty(len(xyz), dtype=campos)
    for i, eje in enumerate(("x", "y", "z")):
        puntos[eje] = xyz[:, i]
    for i, c in enumerate(("red", "green", "blue")):
        puntos[c] = colores[etiquetas, i]
    puntos["superpunto"] = etiquetas
    if procedencia is not None:puntos['origen'] = procedencia
    PlyData([PlyElement.describe(puntos, "vertex")], text=False, byte_order="<").write(str(salida / "superpuntos.ply"))
    return informe
