"""One capped fresh Gaussian probe of the already reviewed new-video118 fit."""
from pathlib import Path
import argparse, json, os, signal, subprocess, sys, time
import numpy as np

PROJECT=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118')
ROOT=Path('/opt/gs-slam-lab/shuolee-new-capture-118-support')
BASE=Path('/opt/gs-slam-lab/shuolee-new-capture-118')
RECORD=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/new-capture-118')
ASSETS='/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets'
sys.path.insert(0,str(PROJECT/'src'))
from shuolee_portrait import runtime as rt
from shuolee_portrait.cli import gpu_lock

def worker(stage):
    rt.configure(ASSETS,ROOT)
    cfg=rt.read(ROOT/'config.json')
    if stage=='probe':
        from shuolee_portrait.training import diagnostic_probe
        diagnostic_probe(ROOT,cfg)
    else:
        from shuolee_portrait.export import export
        export(ROOT,cfg)

def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',choices=['probe','export']);args=p.parse_args()
    if args.worker:return worker(args.worker)
    with gpu_lock():
        assert not (ROOT/'gaussian-run').exists(),'Never resume or replace an earlier Gaussian experiment.'
        assert rt.read(RECORD/'queue-state.json')['state']=='support_fit_pending_visual_review'
        a=dict(np.load(BASE/'fit/fitted.npz'));b=dict(np.load(ROOT/'fit/fitted.npz'))
        wa=dict(np.load(BASE/'fit/observation-weights.npz'));wb=dict(np.load(ROOT/'fit/observation-weights.npz'))
        obs=dict(np.load(ROOT/'data/observations.npz'))
        assert np.array_equal(wa['body_target'],wb['body_target'])
        common=(wa['body']*wa['visibility']>.25)&(wb['body']*wb['visibility']>.25)
        h=obs['boxes'][:,3]-obs['boxes'][:,1]
        scores={}
        for key,fit in [('baseline',a),('image_supported',b)]:
            e=np.linalg.norm(fit['projected_coco17']-wa['body_target'],axis=-1)/h[:,None]
            scores[key]=dict(observations=int(common.sum()),median=float(np.median(e[common])),p90=float(np.quantile(e[common],.9)))
        scores['scope']='Same target pixels and frozen common observation subset; in-sample body landmark fit only, not physical accuracy.'
        rt.save(RECORD/'common-fit-observations.json',scores)
        review=dict(updated=rt.now(),fit_sha256=rt.sha(ROOT/'fit/fitted.npz'),
            reviewed_frames=['frame_000001','frame_000018','frame_000036','frame_000053','frame_000071','frame_000089'],
            method='Actual native-ratio original and mesh overlay images inspected for both fresh fits.',
            findings=['Head, torso and arms broadly overlap the observed person in all six sampled views.',
                'Image-support weighting removes observed off-person foot targets; it does not establish anatomical identity.',
                'Lower legs and feet remain poorly constrained by cropping and furniture occlusion.',
                'Mesh overlay has no furniture depth test; overlay on furniture is not a physical depth verdict.',
                'No clear large improvement from the second fit; aggregate weighted gates are not a controlled quality comparison.'],
            common_observation_errors=scores,formal_geometry_accepted=False,
            next_experiment='Only a capped fresh Gaussian probe to expose appearance, contour and geometry failures before full training.',
            quality_accepted=False)
        rt.save(ROOT/'fit/visual-review.json',review);rt.save(RECORD/'body-visual-review.json',review)
        cfg=rt.read(ROOT/'config.json');rows=rt.read(ROOT/'data/training-inputs.json')['records']
        cfg['training_height']=min(r['wh'][1] for r in rows)
        rt.save(ROOT/'config.json',cfg)
        plan=dict(created=rt.now(),kind='bounded_diagnostic_probe',max_steps=1000,frames=len(rows),
            initial_points=200000,schedule_steps=18000,training_height=cfg['training_height'],
            hypothesis='With real whole-image cameras and a fresh image-supported static body, a short appearance fit will expose whether cross-view ghosting and clothing/boundary failures persist before a full 18k run.',
            variables='One bounded integration probe; not a single-factor causal comparison to any old video or training.',
            camera_sha256=rt.sha(ROOT/'data/training-inputs.json'),fit_sha256=rt.sha(ROOT/'fit/fitted.npz'),
            run_sha256=rt.sha(ROOT/'run.json'),
            source_sha256={n:rt.sha(PROJECT/'src/shuolee_portrait'/n) for n in ['training.py','export.py','admission.py']},
            from_scratch=True,old_gaussians_loaded=False,empty_optimizer_required=True,
            same_experiment_fresh_fit_used=True,formal_training_admitted=False,
            release_gate_unchanged=True,quality_accepted=False,
            stop_conditions=['1000 optimizer steps or 1200 seconds; no automatic extra training.',
                'Stop on nonfinite gradient/loss, changed input hashes, projection mismatch, invalid static mesh, point budget or export-parity failure.',
                'Actual same-camera and orbit review required before any larger run; cropped/occluded feet stay uncertain.'],
            known_limitations=['Factory FISHEYE_8/export mapping remains unknown; SfM and held source tracks are empirical evidence only.',
                'No scene-occluder depth model; silhouette conflicts may damage hidden legs.',
                'Body prior is bare; Gaussian displacement can learn clothes, but this probe does not certify clothing geometry.',
                'Training uses native rectified height, not the prior planned upsample to 1536.'],
            deliverable='Research-only 1000-step world/+Y Gaussian PLY and unwarped source/render/orbit images after format checks.')
        rt.save(ROOT/'gaussian-probe-plan.json',plan);rt.save(RECORD/'gaussian-probe-plan.json',plan)
        # The stable train entry must still reject this unvalidated input.
        from shuolee_portrait.admission import require_alignment,AlignmentNotAdmitted
        try:require_alignment(ROOT)
        except AlignmentNotAdmitted:pass
        else:raise RuntimeError('Formal training unexpectedly admitted this research input.')
        for stage in ['probe','export']:
            cmd=[sys.executable,str(Path(__file__).resolve()),'--worker',stage]
            env=dict(os.environ,PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
            env.pop('PYTHONPATH',None);env.pop('SHUOLEE_EXTRA_PYTHONPATH',None)
            logpath=ROOT/'logs'/f'gaussian-{stage}.log'
            with logpath.open('x') as log:
                proc=subprocess.Popen(cmd,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                rt.save(RECORD/'queue-state.json',dict(updated=rt.now(),state='gaussian_probe_running' if stage=='probe' else 'gaussian_export_running',
                    stage=stage,pid=proc.pid,supervisor_pid=os.getpid(),command=cmd,run_root=str(ROOT),
                    training_started=stage=='export',probe_allocation_pending=stage=='probe',quality_accepted=False,
                    formal_training_admitted=False,log=str(logpath)))
                print(json.dumps(dict(stage=stage,pid=proc.pid,log=str(logpath))),flush=True)
                started=time.monotonic()
                while proc.poll() is None:
                    time.sleep(10)
                    if stage=='probe' and (ROOT/'progress.json').is_file():
                        progress=rt.read(ROOT/'progress.json')
                        if progress.get('execution_kind')=='bounded_diagnostic_probe':
                            state=rt.read(RECORD/'queue-state.json');state.update(updated=rt.now(),training_started=True,probe_allocation_pending=False,progress=progress)
                            rt.save(RECORD/'queue-state.json',state)
                    if time.monotonic()-started>1200:
                        os.killpg(proc.pid,signal.SIGTERM)
                        try:proc.wait(timeout=15)
                        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL)
                        raise RuntimeError('Bounded probe wall-time budget exhausted.')
                code=proc.returncode
            if code:
                rt.save(RECORD/'queue-state.json',dict(updated=rt.now(),state='gaussian_probe_failed',stage=stage,pid=None,run_root=str(ROOT),exit_code=code,training_started=(ROOT/'gaussian-run/preflight.json').is_file(),quality_accepted=False,log=str(logpath)))
                raise SystemExit(code)
        result=rt.read(ROOT/'result.json')
        rt.save(RECORD/'gaussian-probe-result.json',result)
        rt.save(RECORD/'queue-state.json',dict(updated=rt.now(),state='gaussian_probe_pending_review',stage='export',pid=None,run_root=str(ROOT),training_started=True,formal_training_admitted=False,quality_accepted=False,steps=result['steps']))
        print('1000-step probe and format export complete; actual quality/UI review remains.',flush=True)

if __name__=='__main__':main()
