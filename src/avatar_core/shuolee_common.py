"""Local from-scratch adapter paths. No previous reconstruction is loaded."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,hashlib,os,sys
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
BASE=LAB/'outputs/mini-splatting2-mip-depth-001/shuolee'
HEAD=BASE/'current/novel-view-face/head-structure'
PRIVATE=HEAD/'private-runtime'
ROOT=BASE/'current/gaussianavatars-smplx-from-scratch-001'
SOURCE=HEAD/'flame-photometric-registration/whole-body-gaussianavatar/all-views-background/manifest.json'
MODEL=PRIVATE/'models/smplx/SMPLX_NEUTRAL.npz'
def stamp():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(temp,p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def loc(p):
    s=str(p).replace('\\','/')
    return Path('/mnt/'+s[0].lower()+s[2:]) if len(s)>1 and s[1]==':' else Path(s)
def runtime():
    sys.dont_write_bytecode=True
    sys.path[:0]=[str(PRIVATE/'python-deps/smplx-official'),str(PRIVATE/'python-deps'),
        '/opt/gs-slam-lab/shuolee-ga-raster/target','/opt/gs-slam-lab/mini-splatting2-native-001/deps']
    import torch
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    return torch
