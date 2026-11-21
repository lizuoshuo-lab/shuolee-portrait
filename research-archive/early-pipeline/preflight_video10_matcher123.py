"""Local imports, immutable hashes and landscape coordinate convention only."""
import sys,hashlib,json,subprocess,zipfile
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,torch
sys.path[:0]=['/opt/gs-slam-lab/shuolee-roma116/python','/opt/gs-slam-lab/shuolee-roma116/RoMa']
from romatch import roma_outdoor
from roma116_coordinates import raw_centers_to_normalized,normalized_to_raw_centers,sample_direction
from PIL import Image
Q=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video10-correspondence-contrast')
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8388608),b''):h.update(chunk)
    return h.hexdigest()
def local(s):return Path('/mnt/'+s[0].lower()+s[2:].replace('\\','/')) if len(s)>1 and s[1]==':' else Path(s)
assert not (Q/'preflight.json').exists() and not (Q/'run').exists()
p=read(Q/'plan.json'); verified=[]
for s,h in p['input_sha256'].items():
    path=local(s);assert sha(path)==h,s
    if path.suffix=='.png':assert Image.open(path).size==(2704,1520)
    verified.append(s)
assert sha(local(p['coordinate_adapter']))==p['coordinate_adapter_sha256']
assert sha(Q/'review-allocation.json')==p['review_allocation_sha256']
weights=[]
for w in p['weights']:
    f=local(p['weight_root'])/w['filename'];assert f.stat().st_size==w['bytes'] and sha(f)==w['sha256']
    weights.append(dict(name=f.name,bytes=f.stat().st_size,sha256=sha(f)))
native_hw=(1520,2704)
points=torch.tensor([[0.,0.],[2703.,1519.],[1351.5,759.5],[230.,1200.]],dtype=torch.float64)
rt=normalized_to_raw_centers(raw_centers_to_normalized(points,native_hw),native_hw)
assert torch.max(torch.abs(rt-points)).item()<1e-9
# Synthetic nonsquare symmetric map: identical camera pixel grids are identity in both directions.
hh,ww=28,49
yy,xx=torch.meshgrid((torch.arange(hh,dtype=torch.float64)+.5)*2/hh-1,(torch.arange(ww,dtype=torch.float64)+.5)*2/ww-1,indexing='ij')
grid=torch.stack((xx,yy),-1);warp=torch.cat([torch.cat((grid,grid),-1),torch.cat((grid,grid),-1)],1)
certainty=torch.ones((hh,ww*2),dtype=torch.float64)
interior=torch.tensor([[800.,400.],[1300.,700.],[2100.,1100.]],dtype=torch.float64);errors=[]
for direction in ['AtoB','BtoA']:
    predicted,confidence,valid=sample_direction(warp,certainty,interior,native_hw,native_hw,direction)
    error=float(torch.max(torch.abs(predicted-interior)));assert error<1e-8 and valid.all() and (confidence>.999999).all();errors.append(error)
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],capture_output=True,text=True,timeout=15)
historical=Q.parent.parent/'portrait-alignment-open-source-116/roma'
source_record=read(historical/'preflight.json')
commit=source_record['commit']
archive=historical/'RoMa-source-77f8d688.zip'
assert sha(archive)==source_record['archive_sha256']
repo=Path('/opt/gs-slam-lab/shuolee-roma116/RoMa')
source_files=0
with zipfile.ZipFile(archive) as z:
    prefix='RoMa-'+commit+'/'
    for item in z.infolist():
        assert item.filename.startswith(prefix)
        if item.is_dir():continue
        relative=item.filename[len(prefix):]
        installed=repo/relative
        assert installed.resolve().is_relative_to(repo.resolve())
        assert installed.read_bytes()==z.read(item),relative
        source_files+=1
result=dict(updated=datetime.now(timezone.utc).isoformat(),state='preflight_passed_no_inference',plan_sha256=sha(Q/'plan.json'),review_rows=len(read(Q/'review-allocation.json')['rows']),
    input_files_verified=len(verified),weights=weights,repository_commit=commit,coordinate_checks=dict(hw=list(native_hw),roundtrip_max=float(torch.max(torch.abs(rt-points))),symmetric_identity_max=errors),
    torch_version=torch.__version__,cuda_available=torch.cuda.is_available(),gpu_process_snapshot=gpu.stdout.strip(),gpu_query_exit=gpu.returncode,
    source_transport=source_record['source_transport'],source_archive_sha256=sha(archive),source_files_compared=source_files,
    preflight_repair='Source was installed from a pinned official archive, not a git checkout. Compare every archived file instead of git rev-parse; prior terminal failure retained.',
    model_loaded=False,inference_started=False,training_started=False,new_weights=0)
(Q/'preflight.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
