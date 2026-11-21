"""Record an agent's actual image review, never infer approval from a metric."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,hashlib
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=T/'shuolee/current/local-clothed-surface-123'
ROOT=Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/shuolee-local-surface-123')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--subject',required=True);p.add_argument('--phase',choices=['fit','surface','train'],required=True)
p.add_argument('--step',type=int,default=0);p.add_argument('--observation',required=True);p.add_argument('--stop',action='store_true');a=p.parse_args()
root=ROOT/a.subject;folder=REC/f'{a.subject}-{a.phase}-{a.step:06d}-review';idx=read(folder/'index.json')
images=[x for x in idx['images'] if x['region']=='body'];assert len(images)==6
for item in idx['images']:assert sha(folder/item['file'])==item['sha256']
d=dict(updated=datetime.now(timezone.utc).isoformat(),actual_visual_review=True,
       reviewed_frames=[x['frame'] for x in images],index_sha256=sha(folder/'index.json'),summary=a.observation,
       quality_accepted=False,physical_calibration_accepted=False,formal_training_admitted=False)
if a.phase=='fit':
    gate=read(root/'fit/geometry-gate.json');assert gate['coarse_initialization_gate_passed'] or a.stop
    d.update(fit_sha256=sha(root/'fit/fitted.npz'),allow_surface_initialization=not a.stop,coarse_gate=gate)
    name='fit'
elif a.phase=='surface':
    d.update(points_sha256=sha(root/'clothing-init/points.npz'),allow_finite_research_training=not a.stop,
             actual_center_distribution_review=True);name='surface'
else:
    d.update(snapshot_sha256=sha(root/f'gaussian-run/step-{a.step:06d}/world.ply'),continue_same_run=not a.stop,
             actual_center_distribution_review=True);name=f'gaussian-step-{a.step:06d}'
target=root/'reviews'/f'{name}-decision.json'
if target.exists():raise RuntimeError('Consumed decisions cannot be replaced.')
payload=json.dumps(d,ensure_ascii=False,indent=2)+'\n'
tmp=target.with_suffix('.tmp');tmp.write_text(payload,encoding='utf-8');tmp.replace(target)
(folder/'actual-review.json').write_text(payload,encoding='utf-8')
print(target)
