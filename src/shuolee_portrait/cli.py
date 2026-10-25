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

STAGES=('extract','sfm','observations','prior','fit','fit-gate','train','export')


def validate_config(cfg):
    for key in ['frame_stride','max_image_size','min_frames','max_frames','fit_steps','initial_points','iterations','training_height','image_cache_frames']:
        if type(cfg.get(key)) is not int or cfg[key]<=0:raise ValueError(f'{key} must be a positive integer.')
    if cfg['profile']!='static_single_person' or cfg['camera_model']!='SIMPLE_RADIAL' or cfg['matching']!='exhaustive':
        raise ValueError('This release supports one static person, one shared SIMPLE_RADIAL camera and exhaustive image matching.')
    if cfg['min_frames']<3 or cfg['max_frames']<cfg['min_frames'] or cfg['initial_points']<20908:
        raise ValueError('Invalid camera budget or too few points to cover the full mesh.')
    if cfg.get('quality_accepted') is not False:raise ValueError('Quality cannot be pre-approved in an input config.')


@contextmanager
def gpu_lock():
    path=rt.PROJECT/'.gpu-task.lock'
    try:fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL)
    except FileExistsError:raise RuntimeError(f'Existing task lock: {path}. Inspect the recorded PID; do not start a duplicate.')
    try:
        os.write(fd,json.dumps(dict(pid=os.getpid(),host=__import__('socket').gethostname(),command=sys.argv)).encode());os.close(fd)
        yield
    finally:path.unlink()


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
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['doctor','reconstruct','_stage'])
    parser.add_argument('--assets',type=Path,default=rt.PROJECT/'assets')
    parser.add_argument('--config',type=Path,default=rt.PROJECT/'configs/server.json')
    parser.add_argument('--video',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--stop-after',choices=STAGES,default='export',help='Bounded diagnosis; never resumes a previous run.')
    parser.add_argument('--stage',choices=STAGES,help=argparse.SUPPRESS)
    args=parser.parse_args();assets=args.assets.resolve()
    if args.command=='doctor':doctor(assets);return
    if args.command=='_stage':
        if not args.stage or not args.output:parser.error('_stage requires --stage and --output')
        stage(args.stage,args.output.resolve(),assets);return
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
        except Exception:
            rt.save(root/'status.json',dict(state='failed',stage=name,completed=completed,updated=rt.now(),quality_accepted=False,error=traceback.format_exc()))
            raise
        rt.save(root/'status.json',dict(state='completed_pending_quality_review' if completed[-1]=='export' else 'diagnostic_stage_complete',completed=completed,updated=rt.now(),quality_accepted=False))
        print(f'Completed {completed[-1]}. Outputs: {root}. Quality acceptance remains pending.',flush=True)


if __name__=='__main__':main()
