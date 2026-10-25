"""Prueba tecnica CUDA: operadores, forward, perdida oficial y optimizador."""
import json,time,sys
from pathlib import Path
import torch
from M1.AdaPoinTr.src.modelo import load_official
from M1.AdaPoinTr.src.diagnostico import MODULE,sha256

def main():
    if not torch.cuda.is_available():raise RuntimeError('Se requiere CUDA para esta prueba')
    torch.set_num_threads(4);torch.manual_seed(241110);torch.cuda.reset_peak_memory_stats()
    start=time.time();model,cfg=load_official('cuda')
    from pointnet2_ops import pointnet2_utils
    # FPS sobre linea conocida y gather con referencia PyTorch y gradiente.
    line=torch.tensor([[[0.,0,0],[1,0,0],[2,0,0],[3,0,0]]],device='cuda')
    idx=pointnet2_utils.furthest_point_sample(line,2)
    if set(idx[0].tolist())!={0,3}:raise AssertionError('FPS incorrecto')
    features=torch.randn(1,3,4,device='cuda',requires_grad=True)
    actual=pointnet2_utils.gather_operation(features,idx);expected=features.gather(2,idx.long()[:,None].expand(-1,3,-1))
    torch.testing.assert_close(actual,expected);actual.sum().backward()
    expected_grad=torch.zeros_like(features).scatter_add_(2,idx.long()[:,None].expand(-1,3,-1),torch.ones_like(actual))
    torch.testing.assert_close(features.grad,expected_grad)
    a=torch.randn(1,97,3,device='cuda',requires_grad=True);b=torch.randn(1,83,3,device='cuda')
    native=model.loss_func(a,b);dist=torch.cdist(a,b)
    reference=(dist.amin(2).mean()+dist.amin(1).mean())/2
    torch.testing.assert_close(native,reference,rtol=1e-4,atol=1e-5)
    native.backward()
    if not torch.isfinite(a.grad).all():raise AssertionError('Gradiente Chamfer invalido')
    print('Operadores CUDA comprobados contra referencias conocidas.',flush=True)
    # Configuracion oficial sin reducir capas, dimensiones ni salida.
    partial=torch.randn(1,2048,3,device='cuda')*.2
    target=torch.randn(1,16384,3,device='cuda')*.2
    model.eval()
    with torch.inference_mode():coarse,fine=model(partial)
    if coarse.shape!=(1,512,3) or fine.shape!=(1,16384,3) or not torch.isfinite(fine).all():raise AssertionError('Salida oficial inesperada')
    print('Inferencia oficial correcta:',list(coarse.shape),list(fine.shape),flush=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.0005)
    first=next(model.parameters());before=first.detach().clone();losses=[]
    for step in range(2):
        model.train();optimizer.zero_grad(set_to_none=True);ret=model(partial)
        denoise,reconstruction=model.get_loss(ret,target)
        loss=denoise+reconstruction
        if not torch.isfinite(loss):raise AssertionError('Perdida no finita')
        loss.backward()
        gradients=[p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or any(not torch.isfinite(g).all() for g in gradients):raise AssertionError('Gradientes invalidos')
        optimizer.step();torch.cuda.synchronize();losses.append({'step':step+1,'denoising':float(denoise.detach()),'reconstruction':float(reconstruction.detach()),'total':float(loss.detach())})
        print('Paso de prueba:',json.dumps(losses[-1]),flush=True)
    changed=not torch.equal(before,first.detach())
    if not changed:raise AssertionError('El optimizador no actualizo pesos')
    import chamfer,pointnet2_ops._ext
    report={'model':'AdaPoinTr','python':sys.executable,'repository':'https://github.com/yuxumin/PoinTr','commit':json.loads((MODULE/'upstream.json').read_text())['commit'],'torch':str(torch.__version__),'cuda_runtime':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'official_model_config':cfg['model'],'parameters':sum(p.numel() for p in model.parameters()),'input_shape':list(partial.shape),'coarse_shape':list(coarse.shape),'dense_shape':list(fine.shape),'native_operators_verified':True,'official_get_loss':True,'gradients_finite':True,'optimizer_updated_parameters':changed,'optimizer_steps':2,'losses':losses,'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3,'extension_sha256':{'chamfer':sha256(Path(chamfer.__file__)),'pointnet2':sha256(Path(pointnet2_ops._ext.__file__))},'extension_files':{'chamfer':str(chamfer.__file__),'pointnet2':str(pointnet2_ops._ext.__file__)},'selective_import':'clase AdaPoinTr y dependencias originales; no importa otros modelos de comparacion','network_architecture_modified':False,'operators_substituted':False,'bridge_training_started':False,'weights_saved':False,'lurin_modified':False,'seconds':time.time()-start}
    (MODULE/'instalacion/verificacion_modelo.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print('CORRECTO. Prueba tecnica completada. Memoria reservada GiB:',report['peak_reserved_gib'],flush=True)

if __name__=='__main__':main()
