"""Carga selectiva del AdaPoinTr oficial sin importar modelos ajenos."""
import importlib,importlib.machinery,importlib.util,json,sys
from pathlib import Path
from .diagnostico import MODULE,sha256

def load_official(device='cuda'):
    import yaml
    from easydict import EasyDict
    source=MODULE/'third_party/PoinTr'
    manifest=json.loads((MODULE/'upstream.json').read_text(encoding='utf8'))
    for name,expected in manifest['files_sha256'].items():
        if sha256(source/name)!=expected:raise ValueError('Fuente oficial modificada: '+name)
    if str(source) not in sys.path:sys.path.insert(0,str(source))
    for package in ('models','utils','extensions'):
        if package in sys.modules:
            paths=[Path(p).resolve() for p in getattr(sys.modules[package],'__path__',[])]
            if (source/package).resolve() not in paths:raise RuntimeError('Conflicto con paquete '+package)
        else:
            # El __init__ de upstream importa GRNet, Snowflake y PoinTr, con
            # extensiones propias. Registrar solo su paquete permite importar
            # AdaPoinTr intacto sin instalar operadores de los otros modelos.
            spec=importlib.machinery.ModuleSpec(package,loader=None,is_package=True)
            module=importlib.util.module_from_spec(spec);module.__path__=[str(source/package)]
            sys.modules[package]=module
    importlib.import_module('pointnet2_ops._ext')
    importlib.import_module('chamfer')
    model_module=importlib.import_module('models.AdaPoinTr')
    if Path(model_module.__file__).resolve()!=(source/'models/AdaPoinTr.py').resolve():raise RuntimeError('Clase de otro repositorio')
    config=yaml.safe_load((source/'cfgs/PCN_models/AdaPoinTr.yaml').read_text())
    model=model_module.AdaPoinTr(EasyDict(config['model'])).to(device)
    return model,config
