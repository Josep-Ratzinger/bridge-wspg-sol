"""Estado e integridad del método de completado aprobado, sin ensayos previos."""
import hashlib,importlib.util,json,sys
from pathlib import Path

MODULE=Path(__file__).resolve().parents[1]
PROJECT=MODULE.parents[1]


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def diagnose(config=None):
    cfg=json.loads((Path(config) if config else MODULE/'configs/puentes.json').read_text(encoding='utf8'))
    manifest=json.loads((MODULE/'upstream.json').read_text(encoding='utf8'));upstream=MODULE/'third_party/PoinTr'
    if cfg['model']!='AdaPoinTr' or cfg['commit']!=manifest['commit']:raise ValueError('Modelo diferente del fijado')
    for name,expected in manifest['files_sha256'].items():
        source=(upstream/name).resolve()
        if not source.is_relative_to(upstream.resolve()) or not source.is_file() or sha256(source)!=expected:raise ValueError('Fuente oficial modificada: '+name)
    deps={name:importlib.util.find_spec(name) is not None for name in ('torch','timm','einops','matplotlib','easydict','yaml','pointnet2_ops','chamfer')}
    hardware={}
    if deps['torch']:
        import torch
        hardware={'torch':str(torch.__version__),'cuda_runtime':torch.version.cuda,'cuda_available':torch.cuda.is_available(),
                  'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    technical_path=MODULE/'instalacion/verificacion_modelo.json';native_verified=False
    if technical_path.is_file():
        t=json.loads(technical_path.read_text(encoding='utf8'))
        native_verified=(t.get('commit')==manifest['commit'] and Path(t.get('python','')).resolve()==Path(sys.executable).resolve()
                         and t.get('torch')==hardware.get('torch') and t.get('native_operators_verified') is True
                         and t.get('gradients_finite') is True and t.get('optimizer_updated_parameters') is True)
        for name,p in t.get('extension_files',{}).items():
            if not Path(p).is_file() or sha256(Path(p))!=t['extension_sha256'].get(name):native_verified=False
    checkpoint=MODULE/cfg['checkpoint'];training=checkpoint.parent/'resultado.json';weights_verified=False
    if checkpoint.is_file() and training.is_file():weights_verified=sha256(checkpoint)==json.loads(training.read_text())['checkpoint_sha256']
    automatic=cfg['automatic_completion'];output=MODULE/automatic['verified_output'];execution=MODULE/automatic['execution_record'];completion=None
    if execution.is_file():
        r=json.loads(execution.read_text(encoding='utf8'))
        completion={'candidate':str(output),'file_verified':output.is_file() and sha256(output)==r['output_sha256'],
                    'method':r['method'],'originals_verified':r['original_coordinates_colors_ids_verified'],
                    'estimated_points_added':r['estimated_points_added'],'manual_bridge_context_used':r['manual_bridge_context_used'],
                    'visual_review_accepted_by_user':automatic.get('visual_review_accepted_by_user',False),'scope':r['scope']}
    return {'model':'AdaPoinTr','python':sys.executable,'upstream_commit':manifest['commit'],'upstream_files_verified':len(manifest['files_sha256']),
            'dependencies_detected':deps,'hardware':hardware,'native_model_execution_verified':native_verified,
            'input_ply_exists':(PROJECT/cfg['input_ply']).is_file(),'input_npz_exists':(PROJECT/cfg['input_npz']).is_file(),
            'checkpoint':str(checkpoint),'checkpoint_verified':weights_verified,'synthetic_training_record':str(training),
            'completion_result':completion,'evaluation_directory':str(MODULE/'evaluacion'),
            'completion_status':'resultado visual aceptado; conservar origen estimado al pasar a SPG',
            'real_missing_surface_reference_available':False,'validation_on_other_real_bridges_pending':True,
            'next_step':'preparar SPG desde el resultado aprobado, conservando la distincion observado/estimado'}
