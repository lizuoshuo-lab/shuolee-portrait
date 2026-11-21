from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,subprocess,shutil
P=Path('D:/GS-SLAM-Data/shuolee-portrait-new-capture-118')
E=Path('D:/Projects/Shuolee-validation-20261029/free-clothing-transport')
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
D=T/'shuolee/current/local-clothed-surface-123/smplifyx-reference-control/framewise-canonical-route/free-clothing-transport'
N=D/'company00-return'
def lp(p):return Path('\\\\?\\'+str(Path(p).resolve()))
def sha(p):return hashlib.sha256(lp(p).read_bytes()).hexdigest()
def read(p):return json.loads(lp(p).read_text(encoding='utf-8-sig'))
def save(p,v):
    with lp(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
receipt=read(E/'deployment-receipt.json');patch=E/'free-clothing-followup.patch'
assert sha(E/'deployment-receipt.json')=='854de6158797fac004e259084220b2a3ea9dc30a68f9a265fa8d1e345bfcca5e'
assert sha(patch)==receipt['followup_patch']['sha256']=='3624239612ccbcc8180a077b1c06bffda9e8bd114331002bc3351696a2dcc743'
assert receipt['ci']['conclusion']=='success' and receipt['ci']['tests_run']==164
files=['src/shuolee_portrait/free_clothing_transport.py','tests/test_free_clothing_transport.py']
for name in files:assert sha(P/name)==receipt['file_hashes'][name]['original_sha256']
lp(N).mkdir(exist_ok=False)
for name in files:shutil.copy2(lp(P/name),lp(N/Path(name).name))
base=['git','-c','core.autocrlf=false','-c','core.eol=lf','-C',str(P),'apply']
subprocess.run(base+['--check',str(patch)],check=True)
subprocess.run(base+[str(patch)],check=True)
for name in files:assert sha(P/name)==receipt['file_hashes'][name]['final_sha256']
save(N/'merge.json',dict(updated=datetime.now(timezone.utc).isoformat(),passed=True,
    commit=receipt['commit'],branch=receipt['branch'],receipt_sha256=sha(E/'deployment-receipt.json'),
    patch_sha256=sha(patch),files={n:sha(P/n) for n in files},
    review='Only relative covariance validation, NPZ CRC/duplicate validation, explicit CPU factories and local autocast disable. Existing equations and identity policy preserved.',
    regression_pending=True,previous_native_result_scope='Native14 was executed on original b0e271a3 source; no retroactive claim for the merged source.',
    no_running_local_source_when_changed=True))
print(json.dumps(dict(merged=True,files={n:sha(P/n) for n in files})))
