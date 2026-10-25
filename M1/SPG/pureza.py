"""Pureza semántica de la partición sobre puntos reducidos con etiquetas de referencia."""
import json
from pathlib import Path
import numpy as np
from plyfile import PlyData, PlyElement
from .particion import sha256


def plantilla(reduccion, salida):
    reduccion, salida = Path(reduccion), Path(salida)
    if salida.exists():
        raise ValueError("La plantilla ya existe; no se sobrescribe.")
    with np.load(reduccion, allow_pickle=False) as d:
        xyz = d["xyz"]
        rgb = d["rgb"] if "rgb" in d.files else None
        origen = d['origen'] if 'origen' in d.files else None
    campos = [(e,"f8") for e in "xyz"] + [("punto_id","u4"),("clase","i4")]
    if rgb is not None:
        campos += [(c,"u1") for c in ("red","green","blue")]
    if origen is not None:campos += [('origen','u1')]
    puntos = np.zeros(len(xyz), dtype=campos)
    for i,e in enumerate("xyz"):
        puntos[e] = xyz[:,i]
    puntos["punto_id"] = np.arange(len(xyz), dtype=np.uint32)
    puntos["clase"] = -1
    if origen is not None:puntos['origen'] = origen
    if rgb is not None:
        for i,c in enumerate(("red","green","blue")):
            puntos[c] = rgb[:,i]
    salida.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(puntos,"vertex")], text=False,
            comments=["sha256_reduccion="+sha256(reduccion), "clase=-1 sin etiqueta; >=0 clase de referencia; conservar punto_id y coordenadas"]).write(str(salida))
    return len(xyz)


def calcular_pureza(grupos, clases, total_por_grupo):
    grupos, clases, total = map(np.asarray, (grupos, clases, total_por_grupo))
    if grupos.shape != clases.shape or grupos.ndim != 1 or not np.issubdtype(grupos.dtype,np.integer) or not np.issubdtype(clases.dtype,np.integer):
        raise ValueError("Grupos y clases deben ser vectores enteros del mismo tamaño.")
    if total.ndim != 1 or not len(total) or (total <= 0).any() or (grupos < 0).any() or (grupos >= len(total)).any():
        raise ValueError("Conteos o grupos inválidos.")
    conocidas = clases >= 0
    g, c = grupos[conocidas], clases[conocidas]
    if not len(g):
        raise ValueError("No hay etiquetas: clase=-1 significa desconocido, no una clase.")
    pares, conteos = np.unique(np.column_stack((g,c)), axis=0, return_counts=True)
    etiquetados = np.bincount(g, minlength=len(total))
    if (etiquetados > total).any():
        raise ValueError("Hay más etiquetas que puntos del grupo.")
    mayorias = np.zeros(len(total),dtype=np.int64)
    dominante = np.full(len(total),-1,dtype=np.int64)
    # Empates: menor identificador semántico, sin modificar la pureza.
    for (grupo,clase), cantidad in zip(pares,conteos):
        if cantidad > mayorias[grupo]:
            mayorias[grupo], dominante[grupo] = cantidad, clase
    evaluados = etiquetados > 0
    pureza = np.full(len(total),np.nan)
    pureza[evaluados] = mayorias[evaluados]/etiquetados[evaluados]
    resumen = {"puntos_etiquetados": int(len(g)), "cobertura_puntos": float(len(g)/total.sum()),
        "grupos_evaluados": int(evaluados.sum()), "grupos_totales": len(total),
        "grupos_totalmente_etiquetados": int((etiquetados==total).sum()),
        "pureza_ponderada_puntos": float(mayorias.sum()/len(g)),
        "pureza_media_grupos_evaluados": float(pureza[evaluados].mean()),
        "grupos_con_mas_de_una_clase": int((np.bincount(pares[:,0],minlength=len(total))>1).sum()),
        "clases_presentes": np.unique(c).tolist(),
        "alcance": "todos los puntos reducidos" if len(g)==total.sum() else "solo subconjunto etiquetado; no extrapolar al puente completo",
        "interpretacion": "pureza de partición, no exactitud de una red; grupos pequeños pueden dar pureza alta"}
    return resumen, dict(pureza=pureza, clase_mayoritaria=dominante, etiquetados=etiquetados,
        cobertura_por_grupo=etiquetados/total, histograma_grupo_clase=pares, histograma_conteos=conteos)


def evaluar_pureza(reduccion, particion, etiquetas_ply, salida, campo="clase"):
    reduccion,particion,etiquetas_ply,salida = map(Path,(reduccion,particion,etiquetas_ply,salida))
    if any((salida/n).exists() for n in ("pureza.json","pureza.npz")):
        raise ValueError("Ya existe la evaluación de pureza; no se sobrescribe.")
    firma = sha256(reduccion)
    with np.load(reduccion,allow_pickle=False) as d:
        xyz = d["xyz"]
        origen = d['origen'] if 'origen' in d.files else None
    with np.load(particion,allow_pickle=False) as d:
        grupos = d["celda_a_superpunto"]
        meta = json.loads(str(d["metadatos"].item()))
    if meta["sha256_reduccion"] != firma or len(grupos)!=len(xyz):
        raise ValueError("La partición no corresponde a la reducción.")
    ply = PlyData.read(str(etiquetas_ply),mmap=False)
    tipo = next((c.split('=',1)[1] for c in ply.comments if c.startswith('tipo_etiquetas=')), 'no_declarada')
    if tipo == 'propuesta_automatica':raise ValueError('Las propuestas automáticas no son referencia para validar pureza')
    preliminar = tipo == 'anotacion_asistida_preliminar'
    vertices = ply["vertex"].data
    if not {"punto_id",campo,"x","y","z"}.issubset(vertices.dtype.names):
        raise ValueError("Faltan punto_id, XYZ o el campo de clase de referencia.")
    ids, clases = vertices["punto_id"], vertices[campo]
    for datos in (ids,clases):
        if not np.isfinite(datos).all() or not np.equal(datos,np.floor(datos)).all():
            raise ValueError("IDs y etiquetas deben ser enteros finitos, aunque el PLY los guarde como float.")
    if not len(ids) or ids.min()<0 or ids.max()>=len(xyz) or len(np.unique(ids))!=len(ids):
        raise ValueError("punto_id fuera de rango o duplicado; no mezclar recortes solapados.")
    ids, clases = ids.astype(np.int64),clases.astype(np.int64)
    if (clases < -1).any():
        raise ValueError("Usa -1 para desconocido y clases >=0.")
    referencia = np.column_stack([vertices[e] for e in "xyz"])
    if not np.allclose(referencia,xyz[ids],rtol=0,atol=1e-5):
        raise ValueError("Las coordenadas no corresponden a punto_id: no transformar ni remuestrear al etiquetar.")
    n_grupos = int(grupos.max())+1
    if origen is not None:
        observados = origen==0
        total_original = np.bincount(grupos[observados].astype(np.int64),minlength=n_grupos)
        grupo_id = np.flatnonzero(total_original>0)
        remap = np.full(n_grupos,-1,dtype=np.int64);remap[grupo_id] = np.arange(len(grupo_id))
        elegibles = observados[ids]
        resumen,datos = calcular_pureza(remap[grupos[ids[elegibles]]],clases[elegibles],total_original[grupo_id])
    else:
        grupo_id = np.arange(n_grupos)
        total = np.bincount(grupos.astype(np.int64))
        resumen,datos = calcular_pureza(grupos[ids].astype(np.int64),clases,total)
    datos['grupo_id'] = grupo_id
    resumen.update({'tipo_referencia':tipo,'evaluacion_preliminar':preliminar,'validacion_humana_experta':False if preliminar else None,
                    'grupos_totales_spg':n_grupos,'puntos_estimados_excluidos':int((origen==1).sum()) if origen is not None else 0})
    if preliminar:
        resumen['alcance'] += '; anotación asistida preliminar, no validación independiente ni de fronteras ambiguas'
    resumen.update({"sha256_reduccion":firma,"sha256_particion":sha256(particion),
        "sha256_etiquetas":sha256(etiquetas_ply),"campo_clase":campo,
        "unidad": "puntos observados de entrada SPG, peso uniforme" if origen is not None else "puntos reducidos, peso uniforme; no pureza de los puntos originales",
        "formula_ponderada":"suma de conteos de clase mayoritaria por grupo / total de puntos etiquetados"})
    salida.mkdir(parents=True,exist_ok=True)
    (salida/"pureza.json").write_text(json.dumps(resumen,indent=2,ensure_ascii=False),encoding="utf-8")
    np.savez_compressed(salida/"pureza.npz",**datos)
    return resumen
