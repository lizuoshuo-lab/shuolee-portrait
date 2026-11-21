"""One new video, one fresh static reconstruction directory."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from . import runtime as rt
from .admission import AlignmentNotAdmitted, inspect_alignment, require_alignment

STAGES=('extract','sfm','observations','material-tracks','alignment-diagnostics','prior','fit','fit-gate','alignment-gate','train','export')


def validate_config(cfg):
    for key in ['frame_stride','max_image_size','min_frames','max_frames','fit_steps','initial_points','iterations','training_height','image_cache_frames']:
        if type(cfg.get(key)) is not int or cfg[key]<=0:raise ValueError(f'{key} must be a positive integer.')
    if cfg['profile']!='static_single_person' or cfg['camera_model']!='SIMPLE_RADIAL' or cfg['matching']!='exhaustive':
        raise ValueError('This release supports one static person, one shared SIMPLE_RADIAL camera and exhaustive image matching.')
    if cfg['min_frames']<3 or cfg['max_frames']<cfg['min_frames'] or cfg['initial_points']<20908:
        raise ValueError('Invalid camera budget or too few points to cover the full mesh.')
    if cfg.get('quality_accepted') is not False:raise ValueError('Quality cannot be pre-approved in an input config.')
    from .sfm_holdout import validate_settings
    validate_settings(cfg.get('sfm_holdout'))
    from .alignment_diagnostics import validate_settings as diagnostic_settings
    diagnostic_settings(cfg.get('alignment_diagnostics'))


@contextmanager
def gpu_lock():
    from .queue119 import worker_lock
    path=rt.PROJECT/'.gpu-task.lock'
    with worker_lock(path) as stream:
        stream.seek(0);stream.truncate()
        stream.write(json.dumps(dict(pid=os.getpid(),host=__import__('socket').gethostname(),command=sys.argv)).encode())
        stream.flush()
        yield


def doctor(assets):
    rt.configure(assets);checks={}
    for executable in ['ffprobe','ffmpeg']:
        if not shutil.which(executable):raise RuntimeError(f'Missing {executable}; install FFmpeg on the server.')
        checks[executable]=shutil.which(executable)
    checks['assets']=rt.verify_assets()
    torch=rt.c.runtime()
    import pycolmap,smplx,roma,nvdiffrast.torch,pytorch3d,simple_knn._C,diff_gaussian_rasterization
    import ultralytics,rtmlib,onnxruntime,hydra,hydra_zen,pytorch_lightning,timm,yacs,sam2
    # This upstream import reads a generic joint-map asset at import time. Keep
    # it in the asset-aware doctor, not in weight-free installation checks.
    import hmr4d
    hmr4d.PROJ_ROOT=rt.ASSETS/'gvhmr'
    import hmr4d.utils.preproc.vitpose_pytorch,hmr4d.model.gvhmr.utils.endecoder
    from .prior import inference_settings
    inference_settings()
    if pycolmap.__version__!='3.10.0':raise RuntimeError('Requires pycolmap==3.10.0.')
    checks.update(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),pycolmap=pycolmap.__version__,sfm_gpu_available=pycolmap.has_cuda,asset_dependent_modules_imported=True,prior_configuration_composed=True,training_started=False)
    print(json.dumps(checks,indent=2),flush=True);return checks


def stage(name,root,assets):
    if name=='alignment-diagnostics':
        from .alignment_diagnostics import diagnose
        cfg=rt.read(root/'config.json')
        report=diagnose(root,root/'data/alignment-diagnostics',cfg.get('alignment_diagnostics'))
        if not report['measured_tracks']:
            raise AlignmentNotAdmitted('No observable pre-SfM reserved tracks. See data/alignment-diagnostics/summary.json. No body fitting or Gaussian training was started by this stage.')
        return
    if name=='material-tracks':
        from .material_tracks import extract_material_tracks
        cfg=rt.read(root/'config.json')
        report=extract_material_tracks(root,root/'data/material-tracks',cfg.get('material_tracks'))
        if not report['track_count']:
            raise AlignmentNotAdmitted('No cycle-consistent material candidates were found. See data/material-tracks/summary.json. No body fitting or Gaussian training was started by this stage.')
        return
    # Reject before asset configuration or CUDA imports. The training function
    # also checks this boundary when invoked directly by another caller.
    if name in ('alignment-gate','train'):
        require_alignment(root)
        if name=='alignment-gate':return
    cfg=rt.read(root/'config.json');rt.configure(assets,root)
    if name=='extract':
        from .video_sfm import extract
        extract(Path(rt.read(root/'run.json')['video']),root,cfg)
    elif name=='sfm':
        from .video_sfm import sfm
        sfm(root,cfg)
    elif name=='observations':
        from .observations import observe
        observe(root,cfg)
    elif name=='prior':
        from .prior import infer
        infer(root,cfg)
    elif name=='fit':
        from .fitting import main
        main(root,cfg['fit_steps'])
    elif name=='fit-gate':
        from .fit_gate import evaluate
        evaluate(root,cfg)
    elif name=='train':
        from .training import train
        train(root,cfg)
    elif name=='export':
        from .export import export
        export(root,cfg)


def main():
    parser=argparse.ArgumentParser(description=__doc__,epilog='Development release: reconstruct stops at alignment-gate with exit code 3 until independent alignment validation is implemented. It cannot yet complete formal Gaussian training.')
    parser.add_argument('command',choices=['doctor','reconstruct','alignment-status','_stage'])
    parser.add_argument('--assets',type=Path,default=rt.PROJECT/'assets')
    parser.add_argument('--config',type=Path,default=rt.PROJECT/'configs/server.json')
    parser.add_argument('--video',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--stop-after',choices=STAGES,default='export',help='Bounded diagnosis; never resumes a previous run.')
    parser.add_argument('--stage',choices=STAGES,help=argparse.SUPPRESS)
    args=parser.parse_args();assets=args.assets.resolve()
    if args.command=='doctor':doctor(assets);return
    if args.command=='alignment-status':
        if not args.output:parser.error('alignment-status requires --output with an existing run directory')
        if not args.output.is_dir():raise FileNotFoundError(args.output)
        result=inspect_alignment(args.output.resolve())
        print(json.dumps(result,indent=2),flush=True)
        raise SystemExit(3 if not result['training_admitted'] else 0)
    if args.command=='_stage':
        if not args.stage or not args.output:parser.error('_stage requires --stage and --output')
        try:stage(args.stage,args.output.resolve(),assets)
        except AlignmentNotAdmitted as exc:
            print(str(exc),file=sys.stderr,flush=True)
            raise SystemExit(3)
        return
    if not args.video or not args.output:parser.error('reconstruct requires --video and --output')
    video=args.video.resolve();root=args.output.resolve();cfg=rt.read(args.config);validate_config(cfg)
    if not video.is_file():raise FileNotFoundError(video)
    if root.exists():raise FileExistsError(f'Fresh run only. Output already exists: {root}')
    # Early input-only diagnostics do not need body weights or a GPU.
    if args.stop_after not in ['extract','sfm']:doctor(assets)
    for exe in ['ffprobe','ffmpeg']:
        if not shutil.which(exe):raise RuntimeError(f'Missing executable: {exe}')
    with gpu_lock():
        root.mkdir(parents=True);(root/'logs').mkdir();(root/'data').mkdir()
        rt.save(root/'config.json',cfg)
        rt.save(root/'run.json',dict(video=str(video),video_sha256=rt.sha(video),assets=str(assets),created=rt.now(),from_scratch=True,old_subject_loaded=False,quality_accepted=False,release='0.1.0-rc1'))
        completed=[]
        try:
            for name in STAGES:
                rt.save(root/'status.json',dict(state='running',stage=name,completed=completed,updated=rt.now(),quality_accepted=False))
                print(f'Starting {name}; log: {root/"logs"/(name+".log")}',flush=True)
                with (root/'logs'/f'{name}.log').open('x') as log:
                    subprocess.run([sys.executable,str(rt.PROJECT/'run.py'),'_stage','--stage',name,'--output',str(root),'--assets',str(assets)],stdout=log,stderr=subprocess.STDOUT,check=True)
                completed.append(name)
                if name==args.stop_after:break
        except Exception as exc:
            not_admitted=name in ('material-tracks','alignment-diagnostics','alignment-gate','train') and isinstance(exc,subprocess.CalledProcessError) and exc.returncode==3
            rejection_report={'material-tracks':'data/material-tracks/summary.json',
                'alignment-diagnostics':'data/alignment-diagnostics/summary.json'}.get(name,'alignment/training-readiness.json')
            rt.save(root/'status.json',dict(state='alignment_not_admitted' if not_admitted else 'failed',stage=name,completed=completed,updated=rt.now(),quality_accepted=False,error=traceback.format_exc(),
                training_started=False if not_admitted else None,
                alignment_report=rejection_report if not_admitted else None))
            if not_admitted:
                print(f'Stopped before Gaussian training. Alignment evidence is not admitted; see {root/rejection_report}.',file=sys.stderr,flush=True)
                raise SystemExit(3)
            raise
        rt.save(root/'status.json',dict(state='completed_pending_quality_review' if completed[-1]=='export' else 'diagnostic_stage_complete',completed=completed,updated=rt.now(),quality_accepted=False))
        print(f'Completed {completed[-1]}. Outputs: {root}. Quality acceptance remains pending.',flush=True)


if __name__=='__main__':main()
