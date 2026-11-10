"""Genera los dos lotes de aumento del M3 a partir del generador de M2."""
import argparse
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
from .rutas import DATOS
from scipy.spatial import cKDTree


def escena(generar, semilla, situacion):
    xyz, clases, params = generar(semilla)
    rng = np.random.default_rng(semilla + 1)
    tablero = xyz[clases == 2]
    arbol = cKDTree(tablero[:, :2])
    # Aproxima la cota local del tablero usando varias observaciones cercanas.
    def cota(xy):
        _, vecinos = arbol.query(xy, k=12)
        return np.quantile(tablero[vecinos, 2], .85, axis=-1)

    barandas = xyz[clases == 3]
    if situacion == 'baranda_tablero':
        piso = cota(barandas[:, :2])
        barandas[:, 2] = piso + .50 * (barandas[:, 2] - piso)
        centro = np.quantile(barandas[:, 0], rng.uniform(.20, .80))
        ocultos = (abs(barandas[:, 0] - centro) < params['longitud'] * .10)
        conservar = np.ones(len(xyz), dtype=bool)
        conservar[np.flatnonzero(clases == 3)] = ~(ocultos & (rng.random(len(barandas)) < .65))
        xyz[clases == 3] = barandas
        xyz, clases = xyz[conservar], clases[conservar]

    if situacion == 'estribo_terreno':
        estribos = xyz[clases == 7]
        limite = np.quantile(estribos[:, 2], .30)
        bajos = estribos[estribos[:, 2] <= limite]
        pie = bajos[rng.integers(len(bajos), size=2200)].copy()
        pie[:, 0] += np.sign(pie[:, 0]) * rng.uniform(.10, 2.2, len(pie))
        pie[:, 1] += rng.normal(0, .22, len(pie))
        pie[:, 2] = limite + rng.normal(0, .09, len(pie))
        ocultar = (clases == 7) & (xyz[:, 2] < limite) & (rng.random(len(xyz)) < .85)
        xyz = np.vstack((xyz[~ocultar], pie))
        clases = np.r_[clases[~ocultar], np.full(len(pie), 6)]

    # Los dos lotes incluyen señales y vegetación, ausentes del generador base.
    extras, etiquetas = [], []
    for extremo in (.18, .82):
        borde = barandas[np.argmin(abs(barandas[:, 0] - np.quantile(barandas[:, 0], extremo)))]
        x, y = borde[:2]
        z = cota([[x, y]])[0]
        altura = rng.uniform(1.5, 2.7)
        t = rng.uniform(0, 2*np.pi, 130)
        poste = np.column_stack((x + .04*np.cos(t), y + .04*np.sin(t), z + rng.uniform(0, altura, len(t))))
        placa = np.column_stack((x + rng.normal(0, .015, 140), y + rng.uniform(-.35, .35, 140), z + altura + rng.uniform(-.24, .24, 140)))
        distancia = .85 if situacion == 'senal_entorno' else 3.0
        copa = rng.normal(0, [.45, .60, .55], (700, 3))
        copa += [x, y + np.sign(y) * distancia, z + altura]
        extras.extend((poste, placa, copa))
        etiquetas.extend((np.full(len(poste), 9), np.full(len(placa), 9), np.full(len(copa), 8)))
    return np.vstack([xyz, *extras]).astype('f4'), np.concatenate([clases, *etiquetas]), params


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--salida', type=Path, default=DATOS / 'datos_sinteticos')
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    from M2.corpus import preparar_grafo
    from M2.datos import CLASSES, MASKS, firma, guardar
    from M2.estructura import calcular
    from M2.sinteticos import generar

    situaciones = ('baranda_tablero', 'senal_entorno', 'estribo_terreno')
    for lote in ('uniforme', 'dirigido'):
        carpeta = args.salida / lote
        grafos = carpeta / 'grafos'
        grafos.mkdir(parents=True, exist_ok=True)
        filas = []
        for i in range(12):
            semilla = (91000 if lote == 'uniforme' else 92000) + i * 30
            situacion = situaciones[i % len(situaciones)] if lote == 'dirigido' else 'escena_general'
            for intento in range(30):
                xyz, clases, params = escena(generar, semilla + intento, situacion)
                if len(np.unique(clases)) == len(CLASSES) and len(xyz) >= 22000:
                    break
            else:
                raise RuntimeError('No se encontró una escena con las diez clases')
            rng = np.random.default_rng(params['semilla'])
            indices = rng.choice(len(xyz), 22000, replace=False)
            identidad = f'm3_{lote}_{i:02d}'
            destino = grafos / f'{identidad}.npz'
            with TemporaryDirectory(dir=carpeta) as temporal:
                bruto = Path(temporal) / destino.name
                fila = preparar_grafo(xyz[indices], 1 << clases[indices], bruto, dict(
                    id=identidad, tipo='sintetico', division='train', situacion=situacion,
                    parametros=params, taxonomia_version=2))
                with np.load(bruto) as f:
                    datos = {k:f[k] for k in f.files}
                assert fila['puntos'] == 22000 and all(fila['histograma_mascaras'][:10])
                datos['estructura'], _ = calcular(datos['xyz'], datos['grupo_por_punto'])
                np.savez_compressed(bruto, **datos)
                bruto.replace(destino)
            fila.update(archivo=f'grafos/{destino.name}', sha256=firma(destino))
            filas.append(fila)
        guardar(carpeta / 'manifiesto.json', dict(
            lote=lote, clases=CLASSES, mascaras=MASKS.tolist(), taxonomia_version=2,
            puentes=filas, Lurin_en_entrenamiento=False,
            puntos_por_clase=dict(zip(CLASSES, np.sum([f['histograma_mascaras'][:10] for f in filas], axis=0).tolist()))))


if __name__ == '__main__':
    main()



