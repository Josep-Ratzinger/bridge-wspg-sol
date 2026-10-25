"""Recorrido SPG completo desde el completado aprobado."""
import argparse,json,os
from pathlib import Path
os.environ['OMP_NUM_THREADS']='1'
from .particion import sha256

MODULE=Path(__file__).resolve().parent
PROJECT=MODULE.parents[1]


def ejecutar(entrada,salida,reanudar=False):
    from .entrada import preparar_completado
    from .geometria import procesar_geometria
    from .particion import generar_superpuntos
    from .grafo import generar_grafo
    from .preparacion import preparar_entrada
    entrada,salida=Path(entrada).resolve(),Path(salida).resolve()
    estado=salida/'proceso.json';firma=sha256(entrada)
    params={'entrada':str(entrada),'sha256_entrada':firma,'reduccion_adicional':False,'vecinos_geometria':45,
            'elevacion':'z-minmax','vecinos_particion':10,'regularizacion':.03,'peso_verticalidad':1.,'semilla_muestreo':42}
    if salida.exists():
        if not reanudar or not estado.exists():raise ValueError('La salida ya existe; para continuar una ejecución usar --reanudar')
        record=json.loads(estado.read_text(encoding='utf8'))
        if record['parametros']!=params:raise ValueError('La entrada o parámetros cambiaron; usar otra salida')
    else:
        salida.mkdir(parents=True);record={'parametros':params,'etapas':{},'estado':'en curso','pureza_semantica':'pendiente de etiquetas de referencia'}
    def guardar():
        tmp=estado.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2,ensure_ascii=False),encoding='utf8');tmp.replace(estado)
    nube=salida/'entrada/nube_completada.npz';geo=salida/'geometria/geometria.npz';part=salida/'superpuntos/superpuntos.npz'
    stages=[('entrada',[nube],lambda:preparar_completado(entrada,nube)),
            ('geometria',[geo,geo.with_suffix('.ply')],lambda:procesar_geometria(nube,geo.parent,elevacion='z-minmax')),
            ('superpuntos',[part,part.with_suffix('.ply')],lambda:generar_superpuntos(nube,geo,part.parent)),
            ('grafo',[salida/'grafo/grafo.npz',salida/'grafo/informe.json'],lambda:generar_grafo(nube,part,salida/'grafo')),
            ('entrada_red',[salida/'entrada_red/entrada_superpuntos.npz'],lambda:preparar_entrada(nube,geo,part,salida/'entrada_red'))]
    for name,outputs,action in stages:
        if name in record['etapas']:
            for f in outputs:
                if not f.is_file() or sha256(f)!=record['etapas'][name]['archivos'][str(f.relative_to(salida))]:raise ValueError('Etapa guardada modificada: '+name)
            print('Etapa ya verificada: '+name,flush=True);continue
        record['etapa_actual']=name;guardar();print('Etapa SPG: '+name,flush=True)
        report=action();record['etapas'][name]={'resultado':report,'archivos':{str(f.relative_to(salida)):sha256(f) for f in outputs}};guardar()
    if sha256(entrada)!=firma:raise ValueError('El completado original cambió durante SPG')
    record['estado']='SPG y entrada de red generados; clasificación y evaluación semántica pendientes';record.pop('etapa_actual',None);guardar()
    print('SPG terminado. Salida: '+str(salida),flush=True)
    return record


def main():
    p=argparse.ArgumentParser(description='SPG desde la nube completada; conserva origen y no vuelve a reducir densidad.')
    p.add_argument('--entrada',type=Path,default=PROJECT/'M1/AdaPoinTr/resultados/completado/puente_completado.ply')
    p.add_argument('--salida',type=Path,default=MODULE/'resultados')
    p.add_argument('--reanudar',action='store_true');a=p.parse_args()
    ejecutar(a.entrada,a.salida,a.reanudar)


if __name__=='__main__':main()
