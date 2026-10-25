"""Descarga verificable de la dependencia PointNet++ oficial."""
import hashlib,io,json,urllib.request,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/'instalacion'

def main():
    repository='erikwijmans/Pointnet2_PyTorch'
    request=urllib.request.Request('https://api.github.com/repos/'+repository+'/commits/master',headers={'User-Agent':'WSPG-SOL'})
    commit=json.load(urllib.request.urlopen(request,timeout=30))['sha']
    with urllib.request.urlopen('https://codeload.github.com/'+repository+'/zip/'+commit,timeout=60) as response:data=response.read(20*1024**2+1)
    if len(data)>20*1024**2:raise ValueError('Dependencia excede limite')
    destination=ROOT/'fuentes/Pointnet2_PyTorch'
    if destination.exists():raise ValueError('Fuentes ya existentes')
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        if archive.testzip():raise ValueError('ZIP corrupto')
        for item in archive.infolist():
            parts=Path(item.filename).parts[1:]
            if not parts or item.is_dir():continue
            target=destination.joinpath(*parts).resolve()
            if not target.is_relative_to(destination.resolve()):raise ValueError('Ruta insegura')
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(archive.read(item))
    report={'repository':'https://github.com/'+repository,'commit':commit,'archive_sha256':hashlib.sha256(data).hexdigest(),'source_files_sha256':{p.relative_to(destination).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.rglob('*') if p.is_file()}}
    (ROOT/'pointnet2_fuente.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps({'commit':commit,'source':str(destination)}))
if __name__=='__main__':main()
