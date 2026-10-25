"""Entrada única para la versión final de M2."""
import argparse
import sys


def main():
    args=sys.argv[1:]
    # Compatibility with the application command already used in this project.
    if args and args[0]=='--solo-aplicar':args=['aplicar']+args[1:]
    parser=argparse.ArgumentParser(description='M2 final: WSPG y WSPG + SOL estructural.')
    parser.add_argument('comando',choices=['preparar','entrenar','aplicar','evaluar','visualizar'],nargs='?')
    if not args:parser.print_help();return
    command,rest=args[0],args[1:]
    if command=='preparar':from .corpus import main as run
    elif command=='entrenar':from .entrenamiento import main as run
    elif command=='aplicar':from .inferencia import main as run
    elif command=='evaluar':from .evaluacion import main as run
    elif command=='visualizar':from .visualizacion import main as run
    else:parser.parse_args([command]);return
    sys.argv=[sys.argv[0]]+rest;run()


if __name__=='__main__':main()
