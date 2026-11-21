"""Bounded, public-example conversion audit; never consumes subject fit outputs."""
from pathlib import Path
import datetime, hashlib, json, zipfile
import requests

LAB = Path(r'C:/Users/Administrator/Desktop/GS-SLAM-Lab')
BASE = LAB / 'outputs/mini-splatting2-mip-depth-001/shuolee'
RUN = BASE / 'current/mhr-smplx-official-conversion-027'
MODELS = Path(r'D:/GS-SLAM-Data/models/mhr/v1.0.1')
REPO = LAB / 'repos/MHR-shuolee'
NOW = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()

def save(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2), encoding='utf-8')

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

RUN.mkdir(parents=True, exist_ok=True)
if (RUN/'protocol.json').exists():
    raise RuntimeError('Phase already prepared; inspect before retrying.')
save(RUN/'protocol.json', {
    'owner':'shuolee', 'created':NOW, 'phase':27,
    'purpose':'Verify official MHR to SMPL-X conversion on four public SAM3DBody output examples before subject inference.',
    'repo_commit':'877cba5ee61a5f77d1457bf4de93a66695132bd9',
    'repo_url':'https://github.com/facebookresearch/MHR',
    'source_license':'Apache-2.0; existing user-provided SMPL-X template used locally',
    'input_examples':'tools/mhr_smpl_conversion/data/sam3d_body_outputs/*.npz',
    'initializer':'Official default conversion initialization; no learned subject state or Gaussian checkpoint.',
    'runtime':'Isolated WSL Python 3.12, PyMomentum CPU and PyTorch CPU; existing GPU environments unchanged.',
    'verification':['MHR topology and units','official SAM3D prediction translation contract','SMPL-X fitted parameter forward replay','mapped vertex errors and front/side inspection'],
    'limitations':['Public example conversion only, not user-subject reconstruction.','No camera image ground truth supplied for these public NPZ examples.','A successful conversion is not evidence of accurate unseen-view body geometry.'],
    'gaussian_training':False, 'remote_deployment':False,
})
status = json.loads((BASE/'work-status.json').read_text(encoding='utf-8-sig'))
status.update(updated=NOW, current_phase='mhr_smplx_official_conversion_027', local_training_running=False)
status['body_conversion_027']={'state':'preparing_public_example_runtime', 'protocol':str(RUN/'protocol.json'), 'subject_inference_started':False}
save(BASE/'work-status.json',status)

MODELS.mkdir(parents=True,exist_ok=True)
archive=MODELS/'assets.zip'
url='https://github.com/facebookresearch/MHR/releases/download/v1.0.1/assets.zip'
if archive.exists():
    raise RuntimeError('Archive exists: inspect rather than overwrite.')
part=MODELS/'assets.zip.part'
with requests.get(url, stream=True, timeout=(20,60)) as r:
    r.raise_for_status()
    with part.open('wb') as f:
        for b in r.iter_content(4*1024*1024): f.write(b)
assert part.stat().st_size == 198943157, part.stat().st_size
part.replace(archive)
files=[]
needed={'assets/compact_v6_1.model','assets/corrective_activation.npz','assets/corrective_blendshapes_lod1.npz','assets/lod1.fbx'}
with zipfile.ZipFile(archive) as z:
    save(RUN/'release-archive-members.json',[{'name':i.filename,'bytes':i.file_size} for i in z.infolist()])
    assert needed.issubset(set(z.namelist())), sorted(z.namelist())
    for name in sorted(needed):
        p=MODELS/name
        assert p.resolve().is_relative_to(MODELS.resolve())
        p.parent.mkdir(parents=True,exist_ok=True)
        with z.open(name) as src, p.open('wb') as dst:
            for b in iter(lambda:src.read(4*1024*1024),b''): dst.write(b)
        files.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)})
save(RUN/'model-download-receipt.json',{'url':url,'release':'v1.0.1','archive':str(archive),'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive),'extracted_files':files,'note':'SHA records downloaded bytes; no independent official checksum yet verified.'})
print(json.dumps({'phase':str(RUN),'downloaded':archive.stat().st_size,'extracted_files':files},indent=2),flush=True)
