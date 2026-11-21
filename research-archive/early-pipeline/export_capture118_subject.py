"""Export only a finished fresh subject; never starts or resumes training."""
import argparse, os
from capture118_all_subjects import ROOT, REC, ASSETS, rt, state
from shuolee_portrait.cli import gpu_lock

def main(subject,diagnostic_stop=False):
    root=ROOT/subject
    result=rt.read(root/'gaussian-run/result.json')
    assert result['from_scratch']
    if diagnostic_stop:
        decision=rt.read(root/'reviews/gaussian-step-001000-decision.json')
        assert result['state']=='stopped_after_actual_review' and result['steps']==1000
        assert decision['continue_same_run'] is False and decision['actual_same_camera_review']
        assert decision['snapshot_sha256']==rt.sha(root/'gaussian-run/step-001000/world.ply')
    else:
        assert result['steps']==18000
    assert not result['formal_training_admitted']
    assert not (root/'export').exists(), 'Existing export must be inspected, not overwritten.'
    with gpu_lock():
        state(subject,state='exporting',training_running=False)
        try:
            cfg=rt.read(root/'config.json');cfg['export_frame_mode']='scene_up_keep_axis_order'
            rt.save(root/'export-plan.json',dict(mode=cfg['export_frame_mode'],training_changed=False,
                proper_rigid_rotation=True,original_tilt_preserved=True,
                reason='Exact scene-axis permutation avoids export-rounding displacement; this does not estimate gravity or improve geometry.',
                physical_calibration_accepted=False))
            rt.configure(ASSETS,root)
            from shuolee_portrait.export import export
            export(root,cfg)
            validation=rt.read(root/'export/validation.json')
            assert validation['format_passed']
            rt.save(REC/f'{subject}-export-validation.json',validation)
            state(subject,state='exported_pending_actual_review',training_running=False,
                source_sha256=validation['sha256'])
            print('EXPORT_FORMAT_PASSED_QUALITY_AND_UI_PENDING',flush=True)
        except BaseException as exc:
            state(subject,state='export_failed',training_running=False,error=repr(exc))
            raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--subject',choices=['grass','sofa','portrait','indoor118'],required=True);p.add_argument('--diagnostic-stop',action='store_true')
    a=p.parse_args();main(a.subject,a.diagnostic_stop)
