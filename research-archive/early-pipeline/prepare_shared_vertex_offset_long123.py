"""Record the user's duration-only control before starting it."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

T = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
O = T/'shuolee/current/local-clothed-surface-123/smplifyx-reference-control/official-first-cycle'
H = Path(__file__).parent
C = H.parent/'shuolee-coordination'
P = Path('D:/GS-SLAM-Data/shuolee-portrait-new-capture-118')
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, d):
    q = p.with_name(p.name+'.tmp')
    q.write_text(json.dumps(d, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    q.replace(p)
def linux(p):
    s = p.as_posix()
    return '/mnt/'+s[0].lower()+s[2:]

source = O/'shared-vertex-offset-1000'
target = O/'shared-vertex-offset-18000'
plan_path = O/'shared-vertex-offset-long-plan.json'
assert not target.exists() and not plan_path.exists()
old = read(O/'shared-vertex-offset-plan.json')
for name, digest in old['source_hashes'].items():
    assert sha(P/name) == digest, name
assert read(source/'gaussian-run/result.json')['steps'] == 1000
assert read(source/'export/quality-review.json')['quality_accepted'] is False
plan = dict(created=datetime.now(timezone.utc).isoformat(),
    user_direction='做长训练试一下', phase='original_shared_vertex_offset_duration_control',
    source_root=linux(source), target_root=linux(target),
    changed_variable='executed Gaussian steps: 1000 -> 18000',
    config=old['config'], actual_step_budget=18000,
    source_hashes=old['source_hashes'], existing_launcher_sha256=sha(H/'run_official_gaussians123.py'),
    long_launcher_sha256=sha(H/'run_shared_vertex_offset_long123.py'),
    input_hashes={name: sha(source/name) for name in
        ('fit/fitted.npz', 'fit/official-fit-result.json', 'data/training-inputs.json')},
    previous_preflight=dict(path=str(O/'shared-vertex-offset-preflight.json'),
                            sha256=sha(O/'shared-vertex-offset-preflight.json'), sources_unchanged=True),
    training_from_scratch=True, old_optimizer_state_loaded=False,
    fresh_reason='The earlier terminal 1000-step export contains no complete optimizer/RNG/densification state.',
    fit_reused_not_refitted=True, reference_frame_id='frame_000051',
    unchanged=['94 official poses and shared shape', 'native images, masks, cameras and full K',
               'static_offset lr1e-6 and original optimizer/schedule',
               'initial200000 Gaussians, seed6060, original losses, triangle binding and densification'],
    snapshots=[1000,6000,12000,18000],
    review='Actual same-view face/body/feet comparison at1000,6000,12000,18000; final36 novel views and Gaussian/points UI.',
    stop_conditions=['nonfinite loss/gradients', 'native PLY parity failure', 'existing1200000 point limit or coverage constraint'],
    risks=['Longer training can sharpen source views without fixing erroneous poses/cameras.',
           'Original late densification may increase point count, memory and runtime.'],
    end_to_end_benchmark=False, quality_accepted=False)
save(plan_path, plan)
for p in (T/'shuolee/current/robust-portrait-122/queue-state.json',
          T/'shuolee/current/local-clothed-surface-123/queue-state.json', C/'local-research-status.json'):
    d=read(p)
    d.update(updated=plan['created'], updated_at=plan['created'],
        state='shared_vertex_offset_18000_prepared', current_stage='duration_only_long_training',
        question_pending=False, human_choice_pending=None, goal_active=True, goal_complete=False,
        training_running=False, fitting_running=False,
        latest_shared_vertex_offset_long=dict(plan=str(plan_path), root=str(target), user_authorized=True,
                                             quality_accepted=False),
        next='Start one fresh18000step duration control with unchanged shared-offset sources; actual1000step review first.')
    save(p,d)
print(json.dumps(dict(plan=str(plan_path),target=str(target),source_hashes=plan['source_hashes'])))
