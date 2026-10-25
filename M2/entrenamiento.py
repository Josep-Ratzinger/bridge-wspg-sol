"""Reproducción del entrenamiento de las dos ramas finales de M2."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from .datos import BASE,CORPUS,MODELOS,CLASSES,firma,guardar
from .corpus import cargar
from .utilidades import tensors,allowed_matrix,partial_loss,resumen
from .WSPG.modelo import WSPG
from .WSPG.relacional import WSPGRelacional
from .SOL.estructural import preparar as preparar_sol,perdida as loss_sol
from .estructura import NOMBRES


def planificar(data,rng):
    train=[d for d in data if d['division']=='train'];real=[d for d in train if d['tipo']=='real'];synth=[d for d in train if d['tipo']=='sintetico']
    if not real or len(synth)<4:raise ValueError('Faltan puentes reales o sintéticos train')
    plan=[d for d in real for _ in range(4 if d['histogramas'][:,0].sum()>0 else 2)]
    return plan+[synth[i] for i in rng.choice(len(synth),4,replace=False)]


def preparar_tensores(data,device,inicializacion):
    ck=torch.load(inicializacion,map_location='cpu',weights_only=True)
    base=WSPG(clases=10).to(device);base.load_state_dict(ck['state_dict']);base.eval()
    sc={k:v.numpy() for k,v in ck['scaler'].items()}
    f=np.concatenate([d['atributos_nodo'] for d in data if d['division']=='train' and d['tipo']=='real'])
    sc['nm']=f.mean(0);sc['ns']=f.std(0).clip(.05)
    for d in data:
        x,h=tensors(d,sc,device)
        with torch.no_grad():d['z']=base(*x[:5])
        d['features']=x[5];d['raw']=x[5]*torch.as_tensor(sc['ns'],device=device)+torch.as_tensor(sc['nm'],device=device)
        d['edge']=x[3];d['hist']=h;d['sol']=preparar_sol(d['atributos_nodo'],d['aristas'],device)
    return base,sc


@torch.no_grad()
def evaluar_ajuste(ajuste,data):
    ajuste.eval();cm=np.zeros((10,10),dtype='i8')
    for d in data:
        if d['division']!='val' or d['tipo']!='real':continue
        pred=(d['z']+ajuste(d['z'],d['features'],d['edge'],d['raw'])).argmax(1).cpu().numpy()
        for k in range(10):np.add.at(cm[k],pred,d['histogramas'][:,k])
    return resumen(cm)


def entrenar_rama(nombre,alpha,data,base,sc,device,epochs,out,manifest_sha,inicializacion):
    folder=Path(out)/nombre;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'mejor.pt').exists():
        print(nombre+': pesos finales conservados. Para repetir entrenamiento use otra --salida.');return
    torch.manual_seed(614);np.random.seed(614);torch.cuda.manual_seed_all(614)
    model=WSPGRelacional().to(device);model.base.load_state_dict(base.state_dict())
    model.nm.copy_(torch.as_tensor(sc['nm'],device=device));model.ns.copy_(torch.as_tensor(sc['ns'],device=device))
    train=[d for d in data if d['division']=='train'];count=np.zeros(10);nsynth=sum(d['tipo']=='sintetico' for d in train)
    for d in train:
        repeat=(4 if d['histogramas'][:,0].sum()>0 else 2) if d['tipo']=='real' else 4/nsynth
        count+=d['histogramas'][:,:10].sum(0)*repeat
    weights=torch.tensor(np.maximum(count,1000)**-.5,dtype=torch.float32,device=device);weights/=weights.mean()
    allowed=allowed_matrix(device);opt=torch.optim.Adam(model.ajuste.parameters(),lr=.001,weight_decay=.0001)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs,eta_min=.00003);rng=np.random.default_rng(614)
    best=-1.;best_epoch=0;state=None;history=[];started=time.time()
    for epoch in range(1,epochs+1):
        plan=planificar(data,rng);model.ajuste.train();losses=[]
        for idx in rng.permutation(len(plan)):
            d=plan[idx];opt.zero_grad(set_to_none=True)
            scale=0. if rng.random()<.25 else float(rng.uniform(.25,1.));prior=d['z']*scale
            correction=model.ajuste(prior,d['features'],d['edge'],d['raw']);logits=prior+correction
            h=d['hist'].clone();h[:,10:]*=.35
            loss=partial_loss(logits,h,weights,allowed)+.35*partial_loss(correction,h,weights,allowed)
            loss+=alpha*min(1.,max(0.,(epoch-10)/20))*loss_sol(logits,d['sol'])
            if not torch.isfinite(loss):raise ValueError('Pérdida no finita')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.ajuste.parameters(),1.);opt.step();losses.append(float(loss.detach()))
        sched.step();score=evaluar_ajuste(model.ajuste,data)['mIoU']
        history.append(dict(epoca=epoch,loss=float(np.mean(losses)),mIoU_val_real=score))
        if score>best:best=score;best_epoch=epoch;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if epoch==1 or epoch%20==0:print(nombre,epoch,'mIoU val',round(score*100,2),flush=True)
        if epoch>=100 and epoch-best_epoch>=50:break
    cfg=dict(nombre=nombre,estructura=True,atributos_nodo=20,nombres_atributos=NOMBRES,perfil='estructural_final',strength=1.,base_scale=1.,alpha=alpha,
        arquitectura_version=2,semilla=614,epoca=best_epoch,epocas_ejecutadas=epoch,mIoU_val_real=best,
        manifiesto_sha256=manifest_sha,inicializacion_sha256=firma(inicializacion),historial=history,segundos=time.time()-started,
        no_entrenado_con_lurin=True,seleccion_pesos='validación externa; configuración final fija, sin búsqueda de variantes en Lurín')
    torch.save(dict(state_dict=state,scaler={k:torch.as_tensor(v) for k,v in sc.items()},clases=10,config=cfg),folder/'mejor.pt')
    guardar(folder/'entrenamiento.json',cfg)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--datos',type=Path,default=CORPUS)
    p.add_argument('--salida',type=Path,default=MODELOS);p.add_argument('--epocas',type=int,default=220)
    p.add_argument('--inicializacion',type=Path,default=MODELOS/'inicializacion_wspg.pt');a=p.parse_args()
    if a.epocas<1:p.error('epocas debe ser positivo')
    if all((a.salida/n/'mejor.pt').exists() for n in ['WSPG','WSPG_SOL']):
        print('Las dos ramas finales ya están guardadas; no se sobrescriben.');return
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu';_,data=cargar(a.datos)
    base,sc=preparar_tensores(data,device,a.inicializacion)
    for name,alpha in [('WSPG',0.),('WSPG_SOL',.5)]:entrenar_rama(name,alpha,data,base,sc,device,a.epocas,a.salida,firma(a.datos/'manifiesto.json'),a.inicializacion)


if __name__=='__main__':main()
