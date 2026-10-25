"""Descriptores geométricos de SPG calculados por vecindarios k-NN.

Referencia de las fórmulas: compute_geof de loicland/superpoint_graph,
partition/ply_c/ply_c.cpp (rama ssp+spg). Implementación NumPy propia.
Se utilizan raíces de los autovalores para linealidad, planaridad y dispersión.
La verticalidad sigue el vector de autovectores absolutos ponderados por los
AUTOVALORES, como en ese código; no es 1 - abs(normal_z).
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.spatial import cKDTree
from plyfile import PlyData, PlyElement

NOMBRES_LOCALES = ("linealidad", "planaridad", "dispersion", "verticalidad")
NOMBRES = (*NOMBRES_LOCALES, "elevacion")


def elevacion_z_minmax(xyz):
    """Adaptación: (z-z_min)/(z_max-z_min) sobre TODA la nube reducida.

    WSPG 2022 enumera elevación, pero no especifica esta fórmula. Su referencia
    Guinard y Landrieu 2017 usa altura respecto a un modelo de carretera.
    No disponemos de ese terreno; este descriptor NO es altura sobre terreno
    ni cota geográfica. Es sensible a puntos extremos y al marco Z elegido.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError("La elevación requiere coordenadas finitas N × 3 no vacías.")
    z = xyz[:, 2]
    minimo, maximo = float(z.min()), float(z.max())
    elevacion = np.zeros(len(z), dtype=np.float32)
    if maximo > minimo:
        elevacion = ((z - minimo) / (maximo - minimo)).astype(np.float32)
    return elevacion, {"modo": "z-minmax", "z_min": minimo, "z_max": maximo,
        "formula": "(z-z_min)/(z_max-z_min); cero si z_max=z_min",
        "procedencia": "adaptación propia, no especificada operativamente por WSPG 2022",
        "limitacion": "no es altura sobre terreno; sensible a extremos y orientación Z"}



def descriptores(vecindarios):
    """Recibe B vecindarios de M puntos: matriz B × M × 3.

    Cada vecindario incluye el propio punto. Los autovalores se ordenan de
    menor a mayor. Los vecindarios sin extensión reciben ceros y valido=False.
    La verticalidad se refiere al eje Z de entrada, no a gravedad medida.
    """
    posiciones = np.asarray(vecindarios, dtype=np.float64)
    centradas = posiciones - posiciones.mean(axis=1, keepdims=True)
    cov = np.einsum("bni,bnj->bij", centradas, centradas) / posiciones.shape[1]
    valores, vectores = np.linalg.eigh(cov)
    valores = np.maximum(valores, 0.)
    raices = np.sqrt(valores)
    validos = valores[:, 2] > 0
    geof = np.zeros((len(posiciones), 4), dtype=np.float64)
    den = raices[validos, 2]
    geof[validos, 0] = (raices[validos, 2] - raices[validos, 1]) / den
    geof[validos, 1] = (raices[validos, 1] - raices[validos, 0]) / den
    geof[validos, 2] = raices[validos, 0] / den
    vector = np.einsum("bij,bj->bi", np.abs(vectores), valores)
    norma = np.linalg.norm(vector, axis=1)
    geof[validos, 3] = vector[validos, 2] / norma[validos]
    return np.clip(geof, 0, 1).astype(np.float32), validos


def calcular_geometria(xyz, vecinos=45, bloque=10000, progreso=None):
    """Consulta k vecinos distintos del propio índice, sin radio máximo.

    El soporte espacial varía con la densidad: radio[i] es la distancia al
    último vecino usado. No se selecciona ni modifica el número de vecinos
    de forma automática. Se conserva el orden de la nube reducida.
    """
    xyz = np.ascontiguousarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("Se requieren coordenadas finitas con forma N × 3.")
    if vecinos < 3 or vecinos >= len(xyz):
        raise ValueError("El número de vecinos debe ser al menos 3 y menor que el de puntos.")
    if bloque < 1:
        raise ValueError("El tamaño de bloque debe ser positivo.")
    arbol = cKDTree(xyz)
    geof = np.empty((len(xyz), 4), dtype=np.float32)
    radios = np.empty(len(xyz), dtype=np.float32)
    validos = np.empty(len(xyz), dtype=bool)
    for inicio in range(0, len(xyz), bloque):
        fin = min(inicio + bloque, len(xyz))
        distancias, indices = arbol.query(xyz[inicio:fin], k=vecinos + 1, workers=1)
        # El propio índice no siempre aparece primero si hay coordenadas repetidas.
        propio = indices == np.arange(inicio, fin)[:, None]
        orden = np.argsort(propio, axis=1, kind="stable")[:, :vecinos]
        indices = np.take_along_axis(indices, orden, axis=1)
        distancias = np.take_along_axis(distancias, orden, axis=1)
        posiciones = np.concatenate((xyz[inicio:fin, None, :], xyz[indices]), axis=1)
        geof[inicio:fin], validos[inicio:fin] = descriptores(posiciones)
        radios[inicio:fin] = distancias[:, -1]
        if progreso:
            progreso(fin, len(xyz))
    return geof, radios, validos


def procesar_geometria(archivo: Path, salida: Path, vecinos=45, bloque=10000, elevacion="z-minmax"):
    """Lee reduccion.npz y exporta atributos de cálculo y PLY de inspección."""
    if elevacion != "z-minmax":
        raise ValueError("Modo de elevación no implementado; se admite z-minmax como adaptación explícita.")
    archivo, salida = Path(archivo), Path(salida)
    destinos = [salida / "geometria.npz", salida / "geometria.ply"]
    for destino in destinos:
        if destino.exists():
            raise ValueError(f"Ya existe {destino}. No se sobrescribe automáticamente.")
    with np.load(archivo, allow_pickle=False) as datos:
        xyz = datos["xyz"]
        origen = json.loads(str(datos["metadatos"].item()))
    print(f"Calculando geometría de {len(xyz):,} puntos con {vecinos} vecinos...", flush=True)
    ultimo = [-1]
    def progreso(fin, total):
        decena = int(10 * fin / total)
        if decena != ultimo[0]:
            print(f"Progreso: {100 * fin / total:.1f}%", flush=True)
            ultimo[0] = decena
    geof, radios, validos = calcular_geometria(xyz, vecinos, bloque, progreso)
    if not np.isfinite(geof).all() or not np.isfinite(radios).all():
        raise ValueError("Se encontraron resultados no finitos; no se guardó la geometría.")
    altura, meta_altura = elevacion_z_minmax(xyz)
    geof = np.column_stack((geof, altura)).astype(np.float32)
    firma = hashlib.sha256()
    with archivo.open("rb") as f:
        for parte in iter(lambda: f.read(8 * 1024 * 1024), b""):
            firma.update(parte)
    informe = {
        "entrada": archivo.name,
        "sha256_entrada": firma.hexdigest(),
        "reduccion": origen,
        "vecinos_excluyendo_propio": vecinos,
        "puntos_por_covarianza": vecinos + 1,
        "bloque": bloque,
        "eje_vertical": "Z del marco local; no es una vertical física verificada",
        "formula_dimensionalidad": "s_i=sqrt(lambda_i), lambda1>=lambda2>=lambda3; L=(s1-s2)/s1; P=(s2-s3)/s1; S=s3/s1",
        "formula_verticalidad": "u=sum_i(lambda_i*abs(v_i)); V=u_z/||u||",
        "factor_verticalidad_particion": "no aplicado; los atributos se guardan sin ponderar",
        "elevacion": meta_altura,
        "perfil": "WSPG 2022 adaptado: elevación propia y vecindario 45 heredado de SPG",
        "dimensiones": 5,
        "referencia": "https://github.com/loicland/superpoint_graph/blob/ssp%2Bspg/partition/ply_c/ply_c.cpp",
        "numpy": np.__version__, "scipy": scipy.__version__,
        "vecindarios_degenerados": int((~validos).sum()),
        "radio_percentiles_5_50_95_max": np.percentile(radios, [5, 50, 95, 100]).tolist(),
    }
    salida.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destinos[0], geof=geof, nombres=np.array(NOMBRES),
                        radio_vecindario=radios, valido=validos,
                        metadatos=np.array(json.dumps(informe, ensure_ascii=False)))
    campos = [(e, "<f4") for e in ("x", "y", "z", *NOMBRES, "radio_vecindario")]
    puntos = np.empty(len(xyz), dtype=campos)
    for i, eje in enumerate(("x", "y", "z")):
        puntos[eje] = xyz[:, i]
    for i, nombre in enumerate(NOMBRES):
        puntos[nombre] = geof[:, i]
    puntos["radio_vecindario"] = radios
    PlyData([PlyElement.describe(puntos, "vertex")], text=False, byte_order="<").write(str(destinos[1]))
    return informe
