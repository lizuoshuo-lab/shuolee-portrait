"""Record the latest selected-frame scope without changing any fitted array."""
from datetime import datetime, timezone
import audit_gaussianavatars_params123 as a

root = a.REC/'cross-subject-sh0-124'
out = root/'quality-first-eight124'
stopped = root/'grass-temporal-handidentity'
stamp = datetime.now(timezone.utc).isoformat()
assert a.read(stopped/'conclusion.json')['state'] == 'stopped_at13_user_depth_rejection'
assert a.read(stopped/'process-exit.json')['exit_code'] == 1
policy = dict(updated=stamp, target_frames=8, all_frames_must_pass=False,
    latest_user_instructions=['选8帧左右拟合比较好的帧，最好均匀分布；不需要所有帧都好',
                              '截图中手应靠近胸前，伸到胸外的深度不正确'],
    fitting_action='No new fitting. Reuse one coherent existing fit per person; never stitch winners from different fits.',
    selection='Quality first, observed-view balance second. Preserve missing coverage instead of forcing a bad frame into each bin.',
    depth='Low 2D error and temporal smoothness cannot certify wrist-to-chest depth. Review state is explicit and never inferred from rank.',
    geometry_policy='No subject-specific pull-to-chest loss, no image warping, no new sequence solver.',
    gaussian_gate='Only the selected frames need pose acceptance; rejected and unselected frames remain evidence/evaluation, not RGB training inputs.',
    limitation='Eight selected views do not establish full-video accuracy, unseen-side recovery, or broad generalization.',
    old_all94_admission_superseded=True, old_candidates_restart_allowed=False,
    input_identity_guard_default_adopted=False, gaussian_training_started=False)
a.save(out/'scope.json', policy)
summary = '用户指出胸前手臂深度仍错误；草地输入对照已在13帧停止。改为从已有同一套拟合中筛选约8帧，质量优先并兼顾已拍视角；不再要求全序列通过，不强凑坏帧。'
next_action = ('Implement and review quality-first-eight124 using the existing quality rank and cached fits. '
    'No more all94 fitting, no restart of stopped candidates, no per-frame mixture of experiments. '
    'Selected ~8 good poses suffice; preserve coverage gaps and record actual native/raw/side review. '
    'The raised hand must not pass solely because it is in front of the body: its depth must agree with available image evidence. '
    'Training remains off until selected poses are accepted; then reuse the existing SH0 masked Gaussian adapter and1000-step review. '
    'Do not promote the experimental identity guard or claim it was evaluated by the stopped13 run.')
for path in [a.R/'protocol.json', a.R/'queue-state.json', a.R/'evidence-ledger.json', a.C/'local-research-status.json']:
    d = a.read(path)
    d.update(updated=stamp, current_stage='quality_first_sparse_selection124',
        summary=summary, next_action=next_action, next=next_action,
        selected_frame_policy124=policy, current_active_experiment=None,
        active_experiment=None, running=[], worker_pid=None, active_pid=None,
        current_worker_pid_wsl=None, worker_command=None, fitting_running=False,
        fitter_running=False, own_gpu_task_running=False, local_GPU_task_active=False,
        GPU_task_running=False, training_running=False, gaussian_training_running=False,
        training_allowed=False, training_started=False, quality_accepted=False,
        goal_complete=False, last_goal_turn=dict(classification='scope_correction', evidence=summary, goal_complete=False))
    if 'hand_identity124' in d:
        d['hand_identity124'].update(frames=13, pid=None, state='stopped_at13_user_depth_rejection',
            guard_fit_effect_tested=False, conclusion=str(stopped/'conclusion.json'))
    if path == a.C/'local-research-status.json':
        d.update(state=d['current_stage'], updated_at=stamp, as_of_date='2026-11-03')
    a.save(path, d)
path = a.R/'long-term-plan.md'
content = a.lp(path).read_text(encoding='utf-8')
heading = '## 2026-11-03 Latest selected-frame scope (supersedes all-frame admission)'
if heading not in content:
    content += '\n\n'+heading+'\n\n'+summary+'\n\n'+next_action+'\n'
a.lp(path).write_text(content, encoding='utf-8')
print('SPARSE_SCOPE_RECORDED_NO_TRAINING')
