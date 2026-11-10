"""Entrena y evalúa las tres condiciones del experimento de aumento M3."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from .rutas import DATOS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--datos', type=Path, default=DATOS / 'datos_sinteticos')
    parser.add_argument('--salida', type=Path, default=DATOS / 'experimento')
    parser.add_argument('--exportar-ply', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    from M2 import entrenamiento
    from M2.corpus import cargar
    from M2.datos import CORPUS, MODELOS, ROOT, firma, guardar
    from M2.evaluacion import metricas_lurin
    from M2.inferencia import aplicar, cargar_modelo, leer_spg
    from M2.utilidades import resumen, tensors

    output = args.salida
    epochs = 220
    models = output / 'modelos'
    results = output / 'resultados'
    data_root = args.datos
    arms = ('sin_aumento', 'uniforme', 'dirigido')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(4)
    _, base_data = cargar()
    extras = {arm: cargar(data_root / arm)[1] for arm in arms[1:]}
    if any(len(scenes) != 12 for scenes in extras.values()):
        parser.error('Cada lote debe tener 12 escenas')
    init = MODELOS / 'inicializacion_wspg.pt'
    base, scaler = entrenamiento.preparar_tensores(
        base_data + extras['uniforme'] + extras['dirigido'], device, init)
    original_plan = entrenamiento.planificar
    base_sha = firma(CORPUS / 'manifiesto.json')
    plan = {}
    for arm in arms:
        added = extras.get(arm)
        plan[arm] = dict(corpus_base_sha256=base_sha, lote_sha256=firma(
            data_root / arm / 'manifiesto.json') if added else None)
        # Tres escenas del corpus base y una del lote adicional por época.
        def sampler(data, rng, added=added):
            selected = original_plan(data, rng)
            index = sampler.epoch % 12
            sampler.epoch += 1
            if added is not None:
                selected[-1] = added[index]
            return selected

        sampler.epoch = 0
        entrenamiento.planificar = sampler
        try:
            entrenamiento.entrenar_rama(arm, .5, base_data, base, scaler, device, epochs,
                                        models, base_sha, init)
        finally:
            entrenamiento.planificar = original_plan
    guardar(output / 'protocolo.json', dict(
        condiciones=plan, epocas_maximas=epochs, semilla=614, densidad_por_escena=22000,
        slots_sinteticos=4, slots_adicionales_en_variantes=1,
        validacion_utilizada_en_diagnostico=True, prueba_independiente=False))

    spg = ROOT / 'M1/SPG/resultados'
    with np.load(spg / 'superpuntos/superpuntos.npz') as f:
        groups = f['celda_a_superpunto']
    with np.load(spg / 'entrada/nube_completada.npz') as f:
        xyz, origin = f['xyz'], f['origen']
    with np.load(ROOT / 'M1/clases/resultados/clases.npz') as f:
        ref = f['clase_referencia_preliminar']
        np.testing.assert_array_equal(origin, f['origen'])

    # Frontera según contacto entre etiquetas exactas en el mismo superpunto
    # o entre superpuntos conectados. Se mide solo sobre puntos observados.
    valid = (ref >= 0) & (origin == 0)
    presence = np.zeros((int(groups.max()) + 1, 10), dtype=bool)
    presence[groups[valid], ref[valid]] = True
    with np.load(spg / 'grafo/grafo.npz') as f:
        src, dst = f['origen'], f['destino']
    nearby = presence.copy()
    np.logical_or.at(nearby, src, presence[dst])
    np.logical_or.at(nearby, dst, presence[src])
    boundary = valid & (nearby.sum(1)[groups] > 1)
    rail_deck_boundary = valid & (ref == 3) & nearby[groups, 2]

    def completar(report):
        cm = np.asarray(report['matriz_confusion'])
        classes = list(report['IoU'])
        denom = cm.sum(0) + cm.sum(1)
        report['F1'] = {name: float(2 * cm[k, k] / denom[k]) if cm[k].sum() else None
                        for k, name in enumerate(classes)}
        report['baranda_como_tablero'] = dict(
            puntos=int(cm[3, 2]), total_baranda=int(cm[3].sum()),
            fraccion=float(cm[3, 2] / cm[3].sum()) if cm[3].sum() else None)
        return report

    lurin_data = None if args.exportar_ply else leer_spg(spg)[0]
    reports, validation = {}, {}
    for arm in arms:
        model, model_scaler, model_device, _ = cargar_modelo(arm, models)
        confusion = np.zeros((10, 10), dtype='i8')
        with torch.inference_mode():
            for scene in base_data:
                if (scene['division'], scene['tipo']) != ('val', 'real'):
                    continue
                features, _ = tensors(scene, model_scaler, model_device)
                labels = model(*features).argmax(1).cpu().numpy()
                for k in range(10):
                    np.add.at(confusion[k], labels, scene['histogramas'][:, k])
        validation[arm] = completar(resumen(confusion))
        folder = results / arm
        if not args.exportar_ply:
            features, _ = tensors(lurin_data, model_scaler, model_device)
            with torch.inference_mode():
                pred = model(*features).argmax(1).cpu().numpy().astype('i2')
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder / 'predicciones.npz', clase_por_superpunto=pred)
        else:
            pred, _ = aplicar(arm, spg, folder, models)
        pred = pred[groups]
        reports[arm] = completar(metricas_lurin(pred, ref, origin, xyz))
        border_cm = np.bincount(ref[boundary].astype('i8') * 10 + pred[boundary],
                                minlength=100).reshape(10, 10)
        reports[arm]['frontera_spg'] = completar(resumen(border_cm))
        reports[arm]['baranda_como_tablero_en_borde'] = dict(
            puntos=int((rail_deck_boundary & (pred == 2)).sum()),
            total_baranda=int(rail_deck_boundary.sum()))
        cfg = json.loads((models / arm / 'entrenamiento.json').read_text(encoding='utf8'))
        reports[arm].update(epoca=cfg['epoca'], mIoU_val_real=cfg['mIoU_val_real'])
        print(arm, reports[arm]['mIoU'], reports[arm]['IoU'], flush=True)
    guardar(results / 'evaluacion_lurin.json', reports)
    guardar(results / 'validacion_externa.json', validation)

if __name__ == '__main__':
    main()







