"""Propuesta de clases y diagnóstico preliminar de Lurín."""
import argparse,json
from .anotacion import MODULE,PROJECT,generar,sha256


def main(argv=None):
    p=argparse.ArgumentParser(description='Anotación asistida específica de Lurín; no clasificador WSPG entrenado.')
    p.add_argument('accion',nargs='?',choices=['ejecutar','proponer','evaluar'],default='ejecutar')
    p.add_argument('--actualizar',action='store_true',help='Regenerar las anotaciones y retirar solo la evaluación desactualizada');a=p.parse_args(argv)
    if a.actualizar and a.accion=='evaluar':p.error('Actualizar requiere ejecutar o proponer')
    if a.accion in ('ejecutar','proponer'):
        report=MODULE/'resultados/anotacion.json'
        if a.actualizar:annotation=generar(reemplazar=True)
        elif report.exists():
            annotation=json.loads(report.read_text(encoding='utf8'))
            if sha256(PROJECT/'M1/SPG/resultados/entrada/nube_completada.npz')!=annotation['sha256_entrada'] or sha256(MODULE/'configs/lurin.json')!=annotation['sha256_config']:raise ValueError('La entrada o configuración cambió; no reutilizar anotaciones anteriores')
            for name,expected in annotation.get('archivos_sha256',{}).items():
                if sha256(MODULE/name)!=expected:raise ValueError('Anotación modificada: '+name)
        else:annotation=generar()
        print(f"Propuestas: {MODULE/'resultados/puente_clases.ply'}")
        print(f"Referencia asistida preliminar: {annotation['puntos_referencia']:,} puntos observados ({annotation['cobertura_referencia_observados']:.2%}).")
    if a.accion in ('ejecutar','evaluar'):
        from .evaluacion import evaluar
        file=MODULE/'resultados/pureza_preliminar/pureza.json'
        if file.exists():
            r=json.loads(file.read_text(encoding='utf8'))
            if r['sha256_etiquetas']!=sha256(MODULE/'datos/referencia_preliminar.ply') or r['sha256_particion']!=sha256(PROJECT/'M1/SPG/resultados/superpuntos/superpuntos.npz'):raise ValueError('La referencia o SPG cambió')
        else:r=evaluar()
        print(f"Pureza PRELIMINAR: {r['pureza_ponderada_puntos']:.3%}; cobertura observada: {r['cobertura_puntos']:.2%}.")
        print('Anotación asistida sin revisión experta independiente; no valida fronteras ambiguas ni toda la nube.')


if __name__=='__main__':main()
