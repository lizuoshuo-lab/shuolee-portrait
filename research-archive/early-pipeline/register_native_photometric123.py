"""Register one real native-patch measurement contrast after the synthetic implementation control."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
H=Path(__file__).parent;R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122');Q=R/'video10-correspondence-contrast';D=Q/'head-roi-contrast';O=D/'native-photometric-contrast'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not O.exists();c=read(D/'native-patch-control/result.json');assert c['passed'] and c['total']==24
p=read(D/'plan.json');files=[D/'review-allocation.json',D/'native-patch-control/result.json',D/'plan.json',H/'native_patch_control123.py',R/'video10-semantic-role/secondary-observations.json']
files += [Path(v['source']) for v in p['crops'].values()]
files += [Q/'run'/f'map_{a:06d}_{b:06d}.npz' for a,b in p['pairs']]
plan=dict(registered=datetime.now(timezone.utc).isoformat(),state='registered_not_executed',
    hypothesis='The full-frame matcher may assign coarse valid materials but insufficient subpixel positions. Test a native photometric translation refinement at exactly the24 frozen queries, independent of cameras. This cannot fix wrong object identity, specularity or motion.',
    source_sha256={str(z):sha(z) for z in files},queries=24,frames=p['frames'],
    synthetic_control=dict(passed=True,max_translation_error_px=c['errors']['maximum'],real_physical_accuracy_verified=False),
    operation='Each query follows A→B→C→A with an independent return match on each edge (6patch fits per row). Initial targets are sampled from saved full-frame RoMa maps at the current source pixel. No new model inference. Source and target native97x97 context,31x31 active pixels, cubic sampling,dx/dy plus affine brightness. Same optimizer as synthetic control, max30evaluations, translation bounds±3px, gain0.25..4,bias±0.5. No multi-start or parameter scan.',
    evaluation='Keep original native query/family/split. Baseline full-frame cycles from the prior saved comparison. New gates use the same0.9 bidirectional map confidence,1pxFB and1pxcycle thresholds at refined chain positions; require finite successful non-boundary patch optimization. Record every failure, displacement, before/after photometricRMS,FB,cycle and identity conflict. PhotometricRMS is training residual, not independent accuracy.',
    lineage='ExistingFIT/HELD/quarantine immutable. All24 remain in output.8px near-family check on B/C; any opposite-split/quarantine approach is flagged. No refined points enter any solver or training.',
    budget=dict(CPU_seconds=120,max_patch_fits=144,max_nfev_per_fit=30,model_inference=0,camera_fits=0,training=0),
    physical_review='All24 comparisons on unscaled original192px crops, including failed and regressed rows. No image text, no portrait warping. Patch resampling exists only inside the matching objective, never in exported training images.',
    stop='One real contrast only. No patch-size/threshold/brightness-model sweep. Failure does not prove camera error or motion. Success cannot certify independently measured face accuracy. All physical-face/camera/SMPL-X training gates remain closed.')
O.mkdir();(O/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(plan=str(O/'plan.json'),patch_budget=144,CPU_seconds=120),ensure_ascii=False))
