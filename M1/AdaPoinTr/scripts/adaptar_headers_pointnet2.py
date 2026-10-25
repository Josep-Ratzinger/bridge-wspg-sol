"""Compatibilidad de cabeceras con PyTorch actual y MSVC; sin tocar kernels."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'instalacion'
include=root/'fuentes/Pointnet2_PyTorch/pointnet2_ops_lib/pointnet2_ops/_ext-src/include'
changes=[]
for name in ('utils.h','cuda_utils.h'):
    path=include/name;source=path.read_text();before=hashlib.sha256(path.read_bytes()).hexdigest()
    if '#include <ATen/cuda/CUDAContext.h>' not in source:raise ValueError('Cabecera inesperada')
    source=source.replace('#include <ATen/cuda/CUDAContext.h>','#include <c10/cuda/CUDAStream.h>')
    if name=='cuda_utils.h':source=source.replace('#include <vector>','#include <vector>\n#ifdef _MSC_VER\n#define __PRETTY_FUNCTION__ __FUNCTION__\n#endif')
    path.write_text(source);changes.append({'file':name,'before_sha256':before,'after_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
(root/'pointnet2_headers_adaptacion.json').write_text(json.dumps({'changes':changes,'reason':'usar cabecera especifica del stream CUDA; macro de diagnostico MSVC','cuda_kernels_modified':False,'algorithms_modified':False},indent=2),encoding='utf8')
