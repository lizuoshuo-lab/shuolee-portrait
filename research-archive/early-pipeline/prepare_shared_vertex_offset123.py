from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

T = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
O = T/'shuolee/current/local-clothed-surface-123/smplifyx-reference-control/official-first-cycle'
H = Path(__file__).parent
P = Path('D:/GS-SLAM-Data/shuolee-portrait-new-capture-118')
C = H.parent/'shuolee-coordination'
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name(p.name+'.tmp')
    q.write_text(json.dumps(d, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    q.replace(p)
def linux(p):
    value = p.as_posix()
    return '/mnt/'+value[0].lower()+value[2:]

target = O/'shared-vertex-offset-1000'
assert not target.exists()
source = O/'full94-shared-shape'
plan = dict(created=datetime.now(timezone.utc).isoformat(),
    phase='shared_vertex_offset_original_mechanism', source_root=linux(source), target_root=linux(target),
    user_direction=['共享vertexoffset', '这部分源码应该就有，先沿用原版'],
    mechanism='Original FLAME static_offset before LBS, adapted to frozen official SMPL-X forward with one overridden v_template buffer.',
    original_source='src/avatar_core/scene/flame_gaussian_model.py:209-212; flame_model/flame.py static_offset before lbs',
    upstream_optimizer_state='static_offset group is commented out; this experiment explicitly enables the same group and lr1e-6',
    hypothesis='A single canonical vertex displacement shared by94 poses lets the rigged mesh and its Gaussians learn surface deviations.',
    changes=['Enable one10475x3 static_offset parameter before official LBS', 'Save learned offset and bound Gaussian state for verified pose export'],
    unchanged=['94 official SMPLify-X fitted poses and shared beta', 'cameras/full K/native images/masks',
               'Gaussian triangle binding and per-triangle coverage', 'L1+SSIM foreground/silhouette/position/scale losses',
               'initial Gaussian count/seed/optimizer/schedule'],
    config=dict(initial_points=200000, iterations=18000, training_height=1520, seed=6060,
                image_cache_frames=94, export_frame_mode='scene_up_keep_axis_order', shared_static_offset=True),
    actual_step_budget=1000, fit_reused_not_refitted=True, reference_frame_id='frame_000051',
    reference_choice='Previously reviewed open-eye pose51/source250; same selected-reference control, metadata only.',
    training_from_scratch=True, end_to_end_benchmark=False, no_per_frame_vertex_offset=True,
    risks=['Original lr1e-6 is small:1000steps may not expand to loose clothing.',
           'No additional offset regularizer is introduced; inspect folding and joint deformation.',
           'Shared offsets can absorb errors in existing poses/cameras; no calibration claim.'],
    stop_conditions=['zero-offset official vertex error >2e-6m or native pixel error >0.01px',
                     'nonfinite/missing shared offset gradient', 'failed native PLY parity',
                     'actual1000step face/body/centres/novel-view review before any further training'],
    input_hashes={name:sha(source/name) for name in ('fit/fitted.npz','fit/official-fit-result.json','data/training-inputs.json')},
    source_hashes={name:sha(P/name) for name in ('src/shuolee_portrait/training.py','src/shuolee_portrait/export.py',
        'src/avatar_core/scene/smplx_gaussian_model.py','src/avatar_core/scene/gaussian_model.py')},
    existing_launcher_sha256=sha(H/'run_official_gaussians123.py'), quality_accepted=False)
# Resolve actual immutable frame spelling instead of inventing a filename.
import numpy as np
with np.load(source/'fit/fitted.npz', allow_pickle=False) as f:
    plan['reference_frame_id'] = str(f['frame_ids'][50])
    assert int(f['source_frames'][50]) == 250
save(O/'shared-vertex-offset-plan.json', plan)
proposal = O/'limb-and-reference-review/surface-change-proposal.json'
d = read(proposal)
d.update(state='superseded_by_user_shared_vertex_offset_original_source', question_pending=False,
         user_resolution=plan['user_direction'], superseding_plan=str(O/'shared-vertex-offset-plan.json'))
save(proposal,d)
for p in (T/'shuolee/current/robust-portrait-122/queue-state.json',
          T/'shuolee/current/local-clothed-surface-123/queue-state.json', C/'local-research-status.json'):
    d=read(p); d.update(state='shared_vertex_offset_original_bridge_preflight', updated_at=plan['created'],
        training_started=False, next='Verify94 zero-offset meshes and gradients, then fresh1000step shared-offset comparison.',
        latest_shared_vertex_offset=dict(plan=str(O/'shared-vertex-offset-plan.json'),
            user_authorized=True, representation='original shared canonical static_offset before LBS',
            per_frame_offset=False, quality_accepted=False))
    save(p,d)
print(json.dumps(dict(plan=str(O/'shared-vertex-offset-plan.json'),reference=plan['reference_frame_id'], source_hashes=plan['source_hashes'])))
