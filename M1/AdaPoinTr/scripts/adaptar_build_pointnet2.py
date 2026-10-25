"""Ajustes de compilacion para Windows y Ada Lovelace; kernels intactos."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'instalacion'
path=root/'fuentes/Pointnet2_PyTorch/pointnet2_ops_lib/setup.py'
before=hashlib.sha256(path.read_bytes()).hexdigest();source=path.read_text()
old='os.environ["TORCH_CUDA_ARCH_LIST"] = "3.7+PTX;5.0;6.0;6.1;6.2;7.0;7.5"'
if old not in source:raise ValueError('setup.py difiere de la fuente esperada')
source=source.replace(old,'os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")')
source=source.replace('"cxx": ["-O3"]','"cxx": ["/O2"] if os.name == "nt" else ["-O3"]')
path.write_text(source)
(root/'pointnet2_build_adaptacion.json').write_text(json.dumps({'file':str(path),'before_sha256':before,'after_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'changes':['objetivo CUDA 8.9 RTX 4060','flag MSVC /O2'],'kernels_modified':False},indent=2),encoding='utf8')
