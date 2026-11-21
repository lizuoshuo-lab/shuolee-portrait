"""Record the CPU integration handoff; never starts a fit."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

P = Path('D:/GS-SLAM-Data/shuolee-portrait-new-capture-118')
D = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/smplifyx-reference-control/framewise-canonical-route/free-clothing-transport')
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path, obj): Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
reports = [D / ('sequence-streaming-' + name + '-wsl-final.json') for name in ('cpu', 'optimizer', 'root', 'unit')]
assert all(read(p).get('state') == 'passed' or read(p).get('passed') is True for p in reports)
sources = [P / 'src/shuolee_portrait' / name for name in
           ('sequence_fitting.py', 'sequence_optimizer.py', 'sequence_input_adapter.py')]
sources += [P / 'tests/test_sequence_streaming_cpu.py', D / 'full-sequence-inputs.json']
receipt = dict(schema='shared-sequence-streaming-handoff/1', updated=datetime.now(timezone.utc).isoformat(),
    state='CPU_integration_and_real94_input_readback_complete',
    sources={str(p): sha(p) for p in sources},
    tests=[dict(path=str(p), sha256=sha(p), count=read(p)['tests_run']) for p in reports],
    CPU_checks=sum(read(p)['tests_run'] for p in reports),
    environment='WSL Ubuntu-20.04 dynamic-human venv torch2.7.1+cu128, CUDA_VISIBLE_DEVICES empty; no CUDA initialized',
    first_pass_evidence_preserved=['sequence-streaming-cpu-first.json', 'sequence-streaming-unit-regression.json'],
    previous_environment_failure='Windows older Torch lacks weights_only torch.load and Module._apply(recurse); two supplemental unit-test API errors retained. Same unchanged sources pass all25 in actual WSL runtime.',
    APIs=dict(forward='forward_state(state=None, frame_indices=None): state remains full roster; selection is explicit.',
              projection='project(out, frame_indices=None): full K at selected original frame indices.',
              world='world_vertices(out, frame_indices=None): normalized source E applied once.',
              solver="optimize_sequence(..., root_policy='provided_initialization', initialization_source=..., evaluation_mode='streamed')"),
    exactness=['All five registered schedules and weights unchanged.',
       'All per-frame residual terms retain denominator sqrt(total N). Shared beta prior appears once.',
       'Objective and derivatives accumulated over every frame. LBFGS strong_wolfe operates on the single full parameter vector.',
       'Column-squared norms and J-transpose-r accumulated from eight-row VJP blocks, with original tolerance1e-8.',
       'Complete K/skew, fixed cameras, same detector q90 across94, observation eligibility and holdout identities retained.',
       'Synthetic factor1/2 gives exactly equal camera-domain fitting loss and gradient; world export differs as declared. No physical scale claim.'],
    memory=dict(total_parameters94=4804, local_parameters=61, peak_J_rows=8,
                J_block_FP64_bytes=8*61*8, full_J_materialized=False,
                live_body_graph_frames=1, measured_real_GPU_peak=None),
    budgets=dict(full_evaluation_reservation_at_sweep_entry=True, wall_check_before_each_frame_and_J_block=True,
       partial_sweeps_never_accepted_as_best=True, final_diagnose_uses_same_budget=True,
       work_counts=['sweeps_started','sweeps_completed','frame_forwards_completed',
                    'objective_backward_calls_completed','jacobian_vjp_calls_completed','jacobian_vjp_seed_rows_completed']),
    readback=dict(path=str(D / 'full-sequence-inputs-readback.json'),sha256=sha(D / 'full-sequence-inputs-readback.json'),
       frames=94, images_and_masks_fully_decoded=188, source_review_frames=3, source_review_records=153,
       active_face_FIT=27,active_face_HELD=7, other91_face_records=4641, other91_state='unknown/inactive',
       old_optimized_fit_loaded=False, camera_intrinsics_reestimated=False, old_detector_repeat=False),
    runnable_next_action='Root-owned fit_full_sequence123.py --plan <registered once-only plan>. Parent owns GPU lock, one real run and visual acceptance; no new runner duplicated here.',
    this_handoff=dict(real_human_model_forwards=0, GPU_jobs=0, real_fits=0, Gaussian_steps=0,
       physical_calibration_accepted=False, real_quality_accepted=False))
save(D / 'sequence-streaming-handoff.json', receipt)
plan_path = D / 'full-sequence-fit-plan.json'
plan = read(plan_path)
plan['updated'] = receipt['updated']
plan['status'] = 'exact_streamed_solver_CPU_verified_full94_inputs_decoded_ready_for_root_registered_real_run'
plan['ownership']['not_edited'] = []
plan['ownership']['implemented_by_frame_contract_deck'] = ['src/shuolee_portrait/sequence_fitting.py', 'src/shuolee_portrait/sequence_optimizer.py', 'tests/test_sequence_streaming_cpu.py']
plan['ownership']['actual_GPU_runner'] = 'root owns H/fit_full_sequence123.py; no model forward or GPU run by this agent'
plan['adapter']['frozen_manifest'] = str(D / 'full-sequence-inputs.json')
plan['adapter']['frozen_manifest_sha256'] = sha(D / 'full-sequence-inputs.json')
plan['adapter']['draft_runnable'] = False
plan['adapter']['missing_declaration'] = None
plan['adapter']['common_unit_declaration'] = read(D / 'full-sequence-inputs.json')['camera_length_contract']
plan['acceptance'].update(actual94_loader_run=True, streamed_solver_implemented=True, CPU_check_count=55,
    full94_shared_fit=False, real_quality_accepted=False)
plan['implementation_receipt'] = str(D / 'sequence-streaming-handoff.json')
plan['shortest_next_action'] = [
    'Register/freeze one root-owned full94 run plan with existing same-video priors, input manifest and final source SHA.',
    'Run fit_full_sequence123.py --plan once under one GPU lock, declared budget and fresh optimizer. No existing fitted checkpoint is loaded.',
    'Export all94 per-frame camera/world vertices and projections without an all-frame graph; retain source hashes and normalized E provenance.',
    'Review original-image overlays and held face observations. CPU equivalence, file readback and a completed schedule do not establish real fit convergence or Gaussian training admission.']
save(plan_path, plan)
print(json.dumps(dict(receipt=str(D / 'sequence-streaming-handoff.json'), sha256=sha(D / 'sequence-streaming-handoff.json'), checks=receipt['CPU_checks'])))
