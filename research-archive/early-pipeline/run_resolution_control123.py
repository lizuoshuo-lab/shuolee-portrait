"""Config-only 1000-step control, reusing the verified launcher and official fit."""
import argparse
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import run_official_gaussians123 as existing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    path = Path(args.plan)
    plan = existing.read(path)
    source, target = Path(plan['source_root']), Path(plan['target_root'])
    assert plan['config'] == dict(initial_points=200000,iterations=18000,training_height=1520,
        seed=6060,image_cache_frames=94,export_frame_mode='scene_up_keep_axis_order')
    assert plan['actual_step_budget']==1000 and plan['fit_reused_not_refitted'] is True
    assert existing.sha(existing.__file__)==plan['existing_launcher_sha256']
    for file, digest in plan['input_hashes'].items(): assert existing.sha(source/file)==digest,file
    assert not target.exists(), 'No resume or overwrite.'
    sys.path.insert(0,str(existing.PROJECT/'src'))
    from shuolee_portrait.training import verify_official_fit
    provenance=verify_official_fit(source)
    if args.check:
        print('PREPARED_INPUTS_VERIFIED_NO_GPU_NO_TRAINING',flush=True)
        return
    approval=existing.read(path.with_name('native-resolution-1000-approval.json'))
    assert approval['plan_sha256']==existing.sha(path) and approval['human_approved'] is True
    assert approval['scope']=='native_resolution_fresh_1000_steps'
    target.mkdir(exist_ok=False)
    for file in ['fit/fitted.npz','fit/official-fit-result.json','data/training-inputs.json','progress.json']:
        dest=target/file;dest.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(source/file,dest)
    existing.save(target/'control-input.json',dict(plan=plan,plan_sha256=existing.sha(path),
        approval=approval,invoker_sha256=existing.sha(__file__),official_fit=provenance,
        fit_reused_not_refitted=True,Gaussian_from_scratch=True,end_to_end_benchmark=False))
    check_decision=existing.review_decision
    def stop_only(decision_path,request):
        result=check_decision(decision_path,request)
        if result['decision']!='stop':raise ValueError('This control is capped at1000 steps after actual review.')
        return result
    existing.review_decision=stop_only
    existing.CFG=plan['config']
    existing.run(SimpleNamespace(root=str(target),project_root=str(existing.PROJECT),assets_root=str(existing.ASSETS)))


if __name__=='__main__': main()
