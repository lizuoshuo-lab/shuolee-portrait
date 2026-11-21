"""Export the completed bounded fixed-K run; never starts or resumes training."""
from pathlib import Path
import os, sys

PROJECT = Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118')
ROOT = Path('/opt/gs-slam-lab/shuolee-new-capture-118-native-fixedk')
RECORD = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/new-capture-118')
sys.path.insert(0, str(PROJECT / 'src'))
from shuolee_portrait import runtime as rt
from shuolee_portrait.cli import gpu_lock

result = rt.read(ROOT / 'gaussian-run/result.json')
assert result['steps'] == 18000 and result['from_scratch']
assert not result['formal_training_admitted']
assert not (ROOT / 'export').exists(), 'Inspect existing export instead of overwriting.'
with gpu_lock():
    state = dict(updated=rt.now(), pid=os.getpid(), run_root=str(ROOT),
                 state='fixedK_research_exporting', stage='export', training_running=False,
                 steps=18000, step_budget=18000, formal_training_admitted=False,
                 quality_accepted=False)
    rt.save(RECORD / 'queue-state.json', state)
    try:
        control=rt.read(ROOT/'export-numerical-control/result.json')
        assert control['passed'] and control['position_inverse_max']==0
        cfg=rt.read(ROOT/'config.json');cfg['export_frame_mode']='nearest_scene_axes'
        rt.save(ROOT/'export-plan.json',dict(mode=cfg['export_frame_mode'],training_changed=False,
                reason='Exact proper scene-axis frame preserves source positions without recentering; arbitrary body-root rotation exceeded the unchanged strict per-pixel parity threshold.',
                control=control,original_tilt_preserved=True))
        rt.configure('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets',ROOT)
        from shuolee_portrait.export import export
        export(ROOT,cfg)
        validation = rt.read(ROOT / 'export/validation.json')
        assert validation['format_passed']
        rt.save(RECORD / 'fixedk-export-validation.json', validation)
        state.update(updated=rt.now(), state='fixedK_exported_pending_actual_review',
                     stage='quality_and_viewer_review', validation_sha256=validation['sha256'])
        rt.save(RECORD / 'queue-state.json', state)
        print('EXPORT_FORMAT_PASSED_QUALITY_AND_UI_PENDING', flush=True)
    except BaseException as exc:
        state.update(updated=rt.now(), state='fixedK_export_failed', error=repr(exc))
        rt.save(RECORD / 'queue-state.json', state)
        raise
