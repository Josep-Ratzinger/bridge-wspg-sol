"""Ejecutor del modulo AdaPoinTr."""
import argparse,json

def main(argv=None):
    import sys
    argv=sys.argv[1:] if argv is None else argv
    if argv and argv[0]=='completar':
        from .src.completado_automatico import main as complete
        complete(argv[1:])
        return 0
    parser=argparse.ArgumentParser(description='AdaPoinTr para completado supervisado de componentes de puentes.')
    parser.add_argument('comando',choices=['diagnosticar','verificar-modelo','completar'])
    parser.add_argument('--config',default=None)
    args,remaining=parser.parse_known_args(argv)
    if args.comando=='completar':
        from .src.completado_automatico import main as complete
        complete(remaining)
        return 0
    if remaining:parser.error('Argumentos no reconocidos: '+' '.join(remaining))
    if args.comando=='verificar-modelo':
        from .scripts.verificar_modelo import main as verify
        verify()
        return 0
    from .src.diagnostico import diagnose
    try:report=diagnose(args.config)
    except (OSError,ValueError,KeyError,ImportError) as error:parser.exit(1,str(error)+'\n')
    print(json.dumps(report,indent=2,ensure_ascii=False))
    return 0

if __name__=='__main__':raise SystemExit(main())
