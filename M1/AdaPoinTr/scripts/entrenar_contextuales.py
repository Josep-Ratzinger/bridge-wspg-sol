"""Ajuste acotado del AdaPoinTr oficial; selección solo con validación."""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from ..src.diagnostico import MODULE, sha256
from ..src.modelo import load_official
from ..src.geometrias_contextuales import pair
from ..src.entrenamiento import calibrate_bn, evaluate
from ..src.entrenamiento import step_model
from ..src.entrenamiento import save_state, write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pasos',type=int,default=1000)
    parser.add_argument('--salida',type=Path,default=MODULE/'modelos/completado_contextual');args=parser.parse_args()
    if args.pasos<200 or args.pasos%200:parser.error('Pasos debe ser múltiplo de 200, mínimo 200')
    out=args.salida.resolve()
    if out.exists():raise ValueError('Entrenamiento existente; no sobrescribir')
    initial_path=MODULE/'modelos/inicializacion/mejor.pt'
    protocol={'seed':241142,'steps':args.pasos,'batch':4,'selection':'mejor F-score combinado en 18 casos val cada 200 pasos, incluye inicial',
              'train':'recetas train independientes 0..29999, variante elegida al azar','val':'18 recetas val 0..17','test':'18 recetas test 0..17 solo tras selección',
              'generator_sha256':sha256(MODULE/'src/geometrias_contextuales.py'),'initial_checkpoint_sha256':sha256(initial_path),
              'architecture':'AdaPoinTr oficial sin cambios','loss':'get_loss oficial, ponderación por extensión de referencia como ensayo previo',
              'lurin_used':False,'units':'proporciones propias, no cotas normativas','normalization':'centro y extensión solo de entrada observada'}
    torch.set_num_threads(4);torch.manual_seed(protocol['seed']);rng=np.random.default_rng(protocol['seed'])
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,cfg=load_official('cuda');initial=torch.load(initial_path,weights_only=False,map_location='cpu');model.load_state_dict(initial['state_dict'],strict=True)
    for layer in model.modules():
        if isinstance(layer,(torch.nn.BatchNorm1d,torch.nn.BatchNorm2d)):layer.momentum=.9
    optimizer=torch.optim.AdamW(model.parameters(),lr=.00005,weight_decay=.0005)
    val=[pair('val',i,(i//3)%3) for i in range(18)];calibration=[pair('train',i*31,i%3) for i in range(32)]
    out.mkdir(parents=True);write_json(out/'protocolo.json',protocol);baseline,_=evaluate(model,val);write_json(out/'validacion_inicial.json',baseline)
    best=baseline['summary']['combined']['fscore'];best_step=0;history=[];start=time.time();used=set()
    base={'architecture':'AdaPoinTr','repository_commit':initial['repository_commit'],'model_config':cfg['model'],'protocol':protocol,'trained_on_lurin':False}
    save_state(out/'mejor.pt',{**base,'state_dict':model.state_dict(),'step':0})
    for step in range(1,args.pasos+1):
        indices=rng.integers(0,30000,4);batch=[pair('train',int(i),int(rng.integers(3000))) for i in indices];used.update(int(i) for i in indices)
        loss=step_model(model,optimizer,batch)
        if step%50==0:
            write_json(out/'estado.json',{'step':step,'total':args.pasos,'loss':loss,'seconds':time.time()-start});print('Contextuales',step,'/',args.pasos,'loss',round(loss,6),flush=True)
        if step%200==0:
            calibrate_bn(model,calibration,4);result,_=evaluate(model,val);score=result['summary']['combined']['fscore'];history.append({'step':step,'summary':result['summary']});write_json(out/'avance.json',history)
            if score>best:
                best=score;best_step=step;save_state(out/'mejor.pt',{**base,'state_dict':model.state_dict(),'step':step})
            print('Validación F-score',round(score,4),'mejor paso',best_step,flush=True)
    test=[pair('test',i,(i//3)%3) for i in range(18)]
    model.load_state_dict(initial['state_dict'],strict=True);before,_=evaluate(model,test)
    selected=torch.load(out/'mejor.pt',weights_only=False,map_location='cpu');model.load_state_dict(selected['state_dict'],strict=True);after,proposals=evaluate(model,test)
    report={'protocol':protocol,'best_step':best_step,'baseline_test':before,'selected_test':after,'training_recipe_indices':sorted(used),'checkpoint_sha256':sha256(out/'mejor.pt'),'seconds':time.time()-start,'test_never_used_for_selection':True}
    write_json(out/'resultado.json',report);write_json(out/'estado.json',{'completed':True,'step':args.pasos,'selected_step':best_step,'seconds':time.time()-start})
    print('FINAL',json.dumps({'best_step':best_step,'test_before':before['summary'],'test_after':after['summary']}),flush=True)


if __name__=='__main__':main()
