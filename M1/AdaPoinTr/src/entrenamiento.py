"""Pérdida oficial, métricas y calibración reproducible del entrenamiento final."""
import json
import numpy as np
import torch
from scipy.spatial import cKDTree

def cloud_metrics(points,reference,tolerance=.01):
    d1=cKDTree(reference).query(points)[0];d2=cKDTree(points).query(reference)[0]
    precision=float((d1<=tolerance).mean());recall=float((d2<=tolerance).mean())
    return {'cd_l1':float((d1.mean()+d2.mean())/2),'cd2':float((d1*d1).mean()+(d2*d2).mean()),'precision':precision,'recall':recall,'fscore':2*precision*recall/max(precision+recall,1e-15)}

def evaluate(model,cases):
    model.eval();rows=[];proposals=[]
    with torch.inference_mode():
        for case in cases:
            prediction=model(torch.from_numpy(case['partial'][None]).cuda())[1][0].cpu().numpy()
            observed_tree=cKDTree(case['partial']);reference_tree=cKDTree(case['complete'])
            new=prediction[observed_tree.query(prediction)[0]>.003]
            if len(new):
                _,ids=np.unique(np.floor(new/.003).astype('i8'),axis=0,return_index=True);new=new[np.sort(ids)]
            combined=np.vstack([case['partial'],new]);hidden=case['complete'][observed_tree.query(case['complete'])[0]>.01]
            recovery=float((cKDTree(new).query(hidden)[0]<=.01).mean()) if len(new) and len(hidden) else 0.
            row={'base_id':case['base_id'],'shape':case['shape'],'variant':case['variant'],'actual_occlusion_fraction':case['occlusion_fraction'],'partial':cloud_metrics(case['partial'],case['complete']),'prediction':cloud_metrics(prediction,case['complete']),'combined':cloud_metrics(combined,case['complete']),'hidden_reference_points':len(hidden),'missing_region_recovery':recovery,'added_points':len(new),'supported_added_fraction':float((reference_tree.query(new)[0]<=.01).mean()) if len(new) else None}
            rows.append(row);proposals.append(new)
    summary={}
    for key in ('partial','prediction','combined'):
        summary[key]={metric:float(np.mean([r[key][metric] for r in rows])) for metric in rows[0][key]}
    summary.update(cases=len(rows),physical_base_geometries=len({r['base_id'] for r in rows}),missing_region_recovery=float(np.mean([r['missing_region_recovery'] for r in rows])),supported_added_fraction=float(np.mean([r['supported_added_fraction'] for r in rows if r['supported_added_fraction'] is not None])) if any(r['supported_added_fraction'] is not None for r in rows) else None)
    return {'summary':summary,'cases':rows},proposals

def calibrate_bn(model,training_cases,batch_size=4):
    """Calibra la rama limpia con entradas train; nunca usa referencia o val."""
    model.eval()
    layers=[m for m in model.modules() if isinstance(m,(torch.nn.BatchNorm1d,torch.nn.BatchNorm2d))]
    momentums=[m.momentum for m in layers]
    for m in layers:
        m.reset_running_stats();m.momentum=None;m.train()
    with torch.no_grad():
        for start in range(0,len(training_cases),batch_size):
            x=np.stack([c['partial'] for c in training_cases[start:start+batch_size]])
            model(torch.from_numpy(x).to(next(model.parameters()).device))
    for m,momentum in zip(layers,momentums):m.momentum=momentum
    model.eval()

def step_model(model,optimizer,cases):
    x=torch.from_numpy(np.stack([c['partial'] for c in cases])).cuda();y=torch.from_numpy(np.stack([c['complete'] for c in cases])).cuda()
    model.train();optimizer.zero_grad(set_to_none=True);ret=model(x);losses=[]
    for i in range(len(cases)):
        denoise,recon=model.get_loss(tuple(t[i:i+1] for t in ret),y[i:i+1])
        extent=float(np.ptp(cases[i]['complete'],axis=0).max())
        losses.append((denoise+recon)/max(1.,extent))
    loss=torch.stack(losses).mean()
    if not torch.isfinite(loss):raise ValueError('Perdida no finita')
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True);optimizer.step();return float(loss.detach())

def write_json(path,value):
    temporary=path.with_suffix('.json.tmp');temporary.write_text(json.dumps(value,indent=2),encoding='utf8');temporary.replace(path)

def save_state(path,state):
    temporary=path.with_suffix('.pt.tmp');torch.save(state,temporary);temporary.replace(path)
