"""Entrada del proyecto. Ejecutar con --help para ver las opciones."""

import argparse
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
LIMPIEZA = RAIZ / "M1" / "limpieza_y_preparacion"
SPG = RAIZ / "M1" / "SPG"
CLASES = RAIZ / "M1" / "clases"


def main():
    parser = argparse.ArgumentParser(
        description="Procesamiento de nubes de puentes para SPG, WSPG y SOL."
    )
    comandos = parser.add_subparsers(dest="comando", required=True)
    clases = comandos.add_parser('clases', help='Anotación asistida de Lurín y pureza preliminar; no entrenamiento WSPG.')
    clases.add_argument('accion', nargs='?', choices=['ejecutar','proponer','evaluar'], default='ejecutar')
    adapointr = comandos.add_parser("adapointr", help="Diagnostico y verificacion tecnica de AdaPoinTr.")
    adapointr.add_argument("accion", choices=["diagnosticar", "verificar-modelo"])
    adapointr.add_argument("--config", type=Path, default=None)
    revisar = comandos.add_parser("revisar", help="Comprobar un archivo PLY sin modificarlo.")
    revisar.add_argument("archivo", type=Path, help="Ruta de la nube de puntos PLY.")
    reducir = comandos.add_parser("reducir", help="Reducir densidad por celdas, conservando la correspondencia.")
    reducir.add_argument("archivo", type=Path, help="Ruta del PLY original.")
    reducir.add_argument("--celda", type=float, required=True, help="Lado de celda en unidades de la nube.")
    reducir.add_argument("--salida", type=Path, default=LIMPIEZA / "resultados")
    geometria = comandos.add_parser("geometria", help="Calcular descriptores geométricos de la nube reducida.")
    geometria.add_argument("archivo", type=Path, help="Ruta de reduccion.npz.")
    geometria.add_argument("--vecinos", type=int, default=45, help="Vecinos por punto, sin contar el propio (45).")
    geometria.add_argument("--elevacion", choices=["z-minmax"], required=True,
                           help="Adaptación explícita: altura Z normalizada por nube; no es terreno ni fórmula verificada de WSPG.")
    geometria.add_argument("--bloque", type=int, default=10000, help="Puntos procesados por bloque (10000).")
    geometria.add_argument("--salida", type=Path, default=SPG / "resultados" / "geometria")
    particion = comandos.add_parser("particionar", help="Generar superpuntos geométricos con cut pursuit.")
    particion.add_argument("reduccion", type=Path, help="Archivo reduccion.npz.")
    particion.add_argument("geometria", type=Path, help="Archivo geometria.npz.")
    particion.add_argument("--vecinos", type=int, default=10, help="Vecinos para el grafo de partición (10).")
    particion.add_argument("--regularizacion", type=float, default=0.03, help="Penalización de cortes; valor elegido en WSPG 2022 (0.03).")
    particion.add_argument("--peso-verticalidad", type=float, default=1.0, help="Multiplicador de verticalidad (1, sin ponderación adicional).")
    particion.add_argument("--salida", type=Path, default=SPG / "resultados" / "superpuntos")
    preparar = comandos.add_parser("preparar", help="Muestrear 128 puntos con 8 atributos por superpunto.")
    preparar.add_argument("reduccion", type=Path)
    preparar.add_argument("geometria", type=Path)
    preparar.add_argument("particion", type=Path)
    preparar.add_argument("--semilla", type=int, default=42)
    preparar.add_argument("--salida", type=Path, default=SPG / "resultados" / "entrada_red")
    grafo = comandos.add_parser("grafo", help="Construir adyacencia Voronoi y 13 atributos por conexión.")
    grafo.add_argument("reduccion", type=Path)
    grafo.add_argument("particion", type=Path)
    grafo.add_argument("--salida", type=Path, default=SPG / "resultados" / "grafo")
    plantilla = comandos.add_parser("pureza-plantilla", help="Exportar nube reducida con IDs y clase de referencia inicialmente desconocida.")
    plantilla.add_argument("reduccion", type=Path)
    plantilla.add_argument("--salida", type=Path, default=CLASES / "datos" / "para_etiquetar.ply")
    pureza = comandos.add_parser("pureza", help="Evaluar partición con etiquetas semánticas de referencia.")
    pureza.add_argument("reduccion", type=Path)
    pureza.add_argument("particion", type=Path)
    pureza.add_argument("etiquetas", type=Path)
    pureza.add_argument("--campo", default="clase")
    pureza.add_argument("--salida", type=Path, default=CLASES / "resultados" / "pureza")
    limpieza = comandos.add_parser("limpiar", help="Filtrar ruido residual y preparar una normalización reversible antes del completado.")
    limpieza.add_argument("archivo", type=Path, nargs="?", default=LIMPIEZA / "procesamiento_manual" / "puente_orientacion_revision.ply")
    limpieza.add_argument("reduccion", type=Path, nargs="?", default=LIMPIEZA / "resultados" / "reduccion.npz")
    limpieza.add_argument("--vecinos", type=int, default=20)
    limpieza.add_argument("--desviaciones", type=float, default=3.0)
    limpieza.add_argument("--radio", type=float, default=None, help="Unidades de nube; por defecto cuatro veces el tamaño de vóxel.")
    limpieza.add_argument("--minimo-vecinos", type=int, default=3)
    limpieza.add_argument("--salida", type=Path, default=LIMPIEZA / "resultados" / "preparado")
    args = parser.parse_args()
    if args.comando == 'clases':
        from M1.clases.__main__ import main as ejecutar_clases
        ejecutar_clases([args.accion])
        return
    if args.comando == "adapointr":
        from M1.AdaPoinTr.__main__ import main as ejecutar_adapointr
        opciones = [args.accion]
        if args.config is not None:
            opciones.extend(["--config", str(args.config)])
        ejecutar_adapointr(opciones)
        return
    if args.comando == "limpiar":
        try:
            from M1.limpieza_y_preparacion.limpieza import limpiar_preparar
            informe = limpiar_preparar(args.archivo, args.reduccion, args.salida, args.vecinos,
                                      args.desviaciones, args.radio, args.minimo_vecinos)
        except (OSError, ValueError, KeyError, MemoryError) as error:
            parser.exit(1, f"No se pudo completar la preparación: {error}\n")
        print(f"Salida: {args.salida.resolve()}")
        print("puente_preparado.ply: XYZ y RGB en su marco local; puntos_retirados.ply: revisión en rojo.")
        print("preparacion.npz: datos de cálculo, normalización reversible y correspondencias.")
        print("Revisar visualmente antes del completado. La escala disponible sigue siendo estimada.")
        return
    if args.comando in ("grafo", "pureza-plantilla", "pureza"):
        try:
            if args.comando == "grafo":
                from M1.SPG.grafo import generar_grafo
                informe = generar_grafo(args.reduccion, args.particion, args.salida)
                print(f"Superpuntos: {informe['superpuntos']:,}; conexiones dirigidas: {informe['conexiones_dirigidas']:,}")
                print(f"Componentes conexas: {informe['componentes_conexas']}; aislados: {informe['superpuntos_aislados']}")
                print("grafo.npz: nodos, conexiones y 13 atributos; informe.json: trazabilidad y diagnóstico.")
            elif args.comando == "pureza-plantilla":
                from M1.SPG.pureza import plantilla
                cantidad = plantilla(args.reduccion, args.salida)
                print(f"Plantilla: {cantidad:,} puntos. clase=-1 sin etiquetar; conservar punto_id y XYZ.")
            else:
                from M1.SPG.pureza import evaluar_pureza
                informe = evaluar_pureza(args.reduccion, args.particion, args.etiquetas, args.salida, args.campo)
                nombre_metrica = 'Pureza PRELIMINAR' if informe.get('evaluacion_preliminar') else 'Pureza ponderada'
                print(f"{nombre_metrica}: {informe['pureza_ponderada_puntos']:.3%}; cobertura etiquetada: {informe['cobertura_puntos']:.2%}")
                print(f"Alcance: {informe['alcance']}")
            print(f"Salida: {args.salida.resolve()}")
        except (OSError, ValueError, KeyError, MemoryError) as error:
            parser.exit(1, f"No se pudo ejecutar {args.comando}: {error}\n")
        return

    if args.comando == "preparar":
        try:
            from M1.SPG.preparacion import preparar_entrada
            informe = preparar_entrada(args.reduccion, args.geometria, args.particion, args.salida, args.semilla)
        except (OSError, ValueError, KeyError) as error:
            parser.exit(1, f"No se pudo preparar la entrada: {error}\n")
        print(f"Superpuntos preparados: {informe['superpuntos']:,}; 128 puntos y 8 atributos por grupo.")
        print(f"Grupos menores de 25 puntos, conservados con máscara: {informe['grupos_pequenos']:,}")
        print(f"Salida: {args.salida.resolve()}")
        print("Todavía no se ha entrenado la red ni asignado clases semánticas.")
        return


    if args.comando == "particionar":
        # Un hilo nativo evita competencia excesiva en la primera ejecución.
        # La biblioteca no expone una semilla del solucionador.
        import os
        os.environ["OMP_NUM_THREADS"] = "1"
        try:
            from M1.SPG.particion import generar_superpuntos
            informe = generar_superpuntos(args.reduccion, args.geometria, args.salida,
                        args.vecinos, args.regularizacion, args.peso_verticalidad)
        except ModuleNotFoundError as error:
            parser.exit(1, f"Falta una dependencia: {error.name}. Instala requirements.txt.\n")
        except (OSError, ValueError, KeyError) as error:
            parser.exit(1, f"No se pudo generar la partición: {error}\n")
        print(f"Superpuntos: {informe['superpuntos']:,}")
        print(f"Puntos originales representados: {informe['puntos_originales']:,}")
        print(f"Mediana de puntos reducidos por grupo: {informe['mediana_puntos_reducidos_por_superpunto']:.1f}")
        print(f"Grupos de un solo punto reducido: {informe['superpuntos_con_un_punto_reducido']:,}")
        print(f"Salida: {args.salida.resolve()}")
        print("Abre superpuntos.ply en CloudCompare para revisar los grupos por colores.")
        print("Los colores identifican grupos; no son clases como viga, tablero o pilar.")

    if args.comando == "geometria":
        try:
            from M1.SPG.geometria import procesar_geometria
            informe = procesar_geometria(args.archivo, args.salida, args.vecinos, args.bloque, args.elevacion)
        except ModuleNotFoundError as error:
            parser.exit(1, f"Falta una dependencia: {error.name}. Instala requirements.txt.\n")
        except (OSError, ValueError, KeyError) as error:
            parser.exit(1, f"No se pudo calcular la geometría: {error}\n")
        print(f"Vecindarios degenerados: {informe['vecindarios_degenerados']:,}")
        radios = informe["radio_percentiles_5_50_95_max"]
        print(f"Radio de vecindario: mediana {radios[1]:.4f}; percentil 95 {radios[2]:.4f}; máximo {radios[3]:.4f} unidades.")
        print(f"Salida: {args.salida.resolve()}")
        print("geometria.npz: atributos sin ponderar, radios y parámetros.")
        print("geometria.ply: campos escalares para revisar en CloudCompare.")
        print("Todavía no se han generado superpuntos.")

    if args.comando == "reducir":
        try:
            from M1.limpieza_y_preparacion.reduccion import reducir_nube
            informe = reducir_nube(args.archivo, args.salida, args.celda)
        except ModuleNotFoundError as error:
            parser.exit(1, f"Falta una dependencia: {error.name}. Instala requirements.txt.\n")
        except (OSError, ValueError) as error:
            parser.exit(1, f"No se pudo reducir la nube: {error}\n")
        print(f"Puntos originales: {informe['puntos_originales']:,}")
        print(f"Puntos reducidos: {informe['puntos_reducidos']:,}")
        print(f"Porcentaje conservado: {informe['porcentaje_conservado']:.2f}%")
        print(f"Salida: {args.salida.resolve()}")
        print("nube_reducida.ply: vista en CloudCompare.")
        print("reduccion.npz: coordenadas, colores, conteos, correspondencia y parámetros.")
        print("La nube original permanece intacta. Las celdas todavía no son superpuntos.")

    if args.comando == "revisar":
        try:
            from M1.limpieza_y_preparacion.nube import revisar_nube
            informe = revisar_nube(args.archivo)
        except ModuleNotFoundError as error:
            parser.exit(1, f"Falta una dependencia: {error.name}. Instala requirements.txt.\n")
        except (OSError, ValueError) as error:
            parser.exit(1, f"No se pudo revisar la nube: {error}\n")
        print(f"Archivo: {args.archivo.name}")
        print(f"Puntos: {informe['puntos']:,}")
        print(f"Coordenadas no finitas: {informe['no_finitos']:,}")
        print(f"Colores RGB: {'sí' if informe['rgb'] else 'no'}")
        for eje, minimo, maximo in zip("XYZ", informe["minimos"], informe["maximos"]):
            print(f"{eje}: {minimo:.4f} a {maximo:.4f} (extensión {maximo - minimo:.4f})")
        print("Las unidades dependen de la escala aplicada previamente; PLY no acredita metros.")


if __name__ == "__main__":
    main()
