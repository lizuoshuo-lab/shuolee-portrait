"""Register the missing facial-articulation control before any optimization."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=T/'shuolee/current/local-clothed-surface-123';O=REC/'grayT-facial-articulation-review'
ROOT='/opt/gs-slam-lab/shuolee-local-surface-123/grayT'
MODEL='/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets/smplx/SMPLX_NEUTRAL.npz'
def win(p):return Path(p.replace('/mnt/d/','D:/')) if p.startswith('/mnt/d/') else Path('//wsl.localhost/Ubuntu-20.04'+p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert not O.exists(),'This registered control already exists; do not replace it.'
files=[ROOT+'/fit/fitted.npz',ROOT+'/fit/observation-weights.npz',ROOT+'/data/observations.npz',ROOT+'/data/inputs.json',MODEL,ROOT.rsplit('/',1)[0]+'/runtime/src/shuolee_portrait/fitting.py']
source=H/'probe_grayT123_facial_articulation.py'
stamp=datetime.now(timezone.utc).isoformat()
plan=dict(created=stamp,hypothesis='The existing fitter constrains 9 nose and 20 mouth semantics while both jaw_pose and expression remain zero and outside its optimizer. Testing one shared facial articulation isolates this representational limitation; it cannot by itself establish camera/motion truth.',
    single_changed_factor='One static 10-coefficient expression and 3-axis jaw; initialized at zero. K/E, shared body pose/root/scale/identity, all 95 original observations and weights remain frozen.',
    known_confounds='Facial semantic detections, model-derived visibility and all-frame staticity are not physical ground truth. Do not tune/remove points by resulting residual.',
    sources=[dict(path=p,sha256=sha(win(p))) for p in files],diagnostic_source_sha256=sha(source),
    parameter_scales=[3.]*10+[.35]*3,parameterization='scale*tanh(raw), one bounded static vector for every frame',
    objective='Exactly the saved weighted soft_l1 face term, coordinate scale6*box_height/1000; no additional regularizer or weight changes. This is not a full new body fit.',
    optimizer='one LBFGS max_iter100 max_eval150, strong_wolfe, zero initial parameters',cpu_seconds=150,device='CPU',threads=4,
    parity_tolerance_px=.01,review_frames=[40,48,56,64,72,80],
    controls=['Zero-param SMPL-X output versus saved world vertices and native projected_face51','All13 objective gradients versus central finite differences','All original source SHA256 unchanged'],
    negative_controls=['K/E, body pose and translation/scale are fixed','Prior independent M0 static3D/material validation is unchanged and never enters optimization'],
    stop=['Any parity/gradient/depth/nonfinite failure stops; no rerun or threshold sweep','Budget is final; a near-bound solution is reported, not relaxed','No fresh Gaussian training or candidate camera/model promotion from semantic loss reduction'],
    acceptance='Quantify omission effect only; physical_alignment_accepted remains false without independent physical validation')
O.mkdir();save(O/'plan.json',plan)
q=read(REC/'queue-state.json');q.update(updated=stamp,current_stage='facial-articulation-control-registered',next_plan=str(O/'plan.json'),
    next_action='One CPU-only frozen-camera/body control of omitted shared jaw/expression; no production changes.',training_allowed=False,training_started=False)
save(REC/'queue-state.json',q)
print(json.dumps(dict(plan=str(O/'plan.json'),sources=len(files),image_key_example=read(win(files[3]))['records'][39].get('image'),source_sha256=plan['diagnostic_source_sha256']),ensure_ascii=False))
