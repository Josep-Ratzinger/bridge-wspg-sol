"""Punto de entrada de M3."""
import argparse
from importlib import import_module
import sys


COMANDOS = {
    "generar-lotes": "aumento_sintetico.generar_lotes",
    "entrenar-comparar": "entrenamiento.entrenar_comparar",
    "visualizar-lurin": "evaluacion.visualizar_lurin",
}


def main():
    parser = argparse.ArgumentParser(description="M3: aumento sintético")
    parser.add_argument("comando", choices=COMANDOS)
    parser.add_argument("opciones", nargs=argparse.REMAINDER)
    if len(sys.argv) == 1:
        parser.print_help()
        return
    args = parser.parse_args()
    sys.argv[1:] = args.opciones
    import_module(f"M3.{COMANDOS[args.comando]}").main()


if __name__ == "__main__":
    main()
