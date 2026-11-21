"""Prepare a joint-video diffusion trial; inference requires an authorized local SVD snapshot."""
import argparse
import functools
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

S = Path(__file__).parent
LAB = S.parents[1]
ROOT = LAB / 'outputs/mini-splatting2-mip-depth-001'
OUT = ROOT / 'viewextrapolator-pilot-001'
REPO = LAB / 'repos/ViewExtrapolator'
DEPENDENCIES = '/opt/gs-slam-lab/mini-splatting2-refine/video-diffusion-deps'


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def prepare():
    import numpy as np
    import torch
    from local_refinement import read_model, render
    from run_difix_distill_pilot import camera
    OUT.mkdir(exist_ok=False)
    # Rebuild from retained 3D parameters, without depending on a retired Difix run.
    cameras = json.loads((ROOT / 'shuolee/camera-paths.json').read_text())
    baseline = json.loads((ROOT / 'native-resolution-cache-full-18000-s0/manifest.json').read_text())
    source_model = Path(baseline['baked_ply'])
    assert digest(source_model) == baseline['baked_ply_sha256']
    model = read_model(source_model)
    clips = {}
    for label, start in [('pitch_up', 0), ('pitch_down', 30)]:
        selected = [next(c for c in cameras if c['name'] == f'elevation-{i:03d}') for i in range(start, start + 25)]
        rgb_frames, alpha_frames = [], []
        with torch.no_grad():
            for row in selected:
                rgb_frame, alpha, _ = render(model, camera(row), mode='mip')
                rgb_frames.append((rgb_frame.clamp(0,1)*255).round().byte().cpu().numpy())
                alpha_frames.append(alpha.detach().cpu().numpy().squeeze())
        rgb = np.stack(rgb_frames)
        opacity = np.stack(alpha_frames)
        assert rgb.shape == (25, 1024, 576, 3) and opacity.shape == (25, 1024, 576)
        assert np.isfinite(opacity).all()
        # A high-opacity region is a guidance heuristic, not a correctness/confidence label.
        np.savez_compressed(OUT / (label + '.npz'), rgb=rgb, mask=np.clip(opacity, 0, 1).astype(np.float32))
        clips[label] = dict(cameras=selected, frames=25, file=label + '.npz', sha256=digest(OUT / (label + '.npz')),
            alpha_mean=float(opacity.mean()), pitch_delta=[float(c['delta_deg']) for c in selected])
    commit = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    protocol = dict(status='prepared_waiting_for_authorized_weights', created=time.time(), clips=clips,
        repo='https://github.com/Kunhao-Liu/ViewExtrapolator', commit=commit,
        source_files={name:digest(REPO / name) for name in ['svd_pipeline.py', 'scheduler.py', 'refine_video.py']},
        weight_source='https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1',
        revision='043843887ccd51926e3efed36270444a838e7861', anonymous_weight_access_http_status=401,
        fp16_weights_bytes=4509184986, resolution=[576,1024], frames=25, fps=6, seed=42,
        steps=25, resample_steps=3, guide_until=15, cpu_offload=True, encode_chunk=2, decode_chunk=2,
        mask='Continuous rendered alpha, identical across treatment; not a correctness/visibility GT',
        changes_from_official='Explicit portrait resolution; chunked VAE encode/decode; feed lossless arrays instead of compressed video',
        unchanged='Official denoising and scheduler implementation; first rendered frame as image condition',
        planned_comparisons=['same raw 3D frames', 'joint video diffusion frames'],
        evaluation=['human A/B detail clarity', 'appearance preservation', 'structure and occlusion',
                    'temporal continuity', 'overall preference', 'reprojection diagnostics as secondary evidence'],
        inference_run=False, geometry_updated=False, accepted=False,
        limitations=['No novel-view GT', 'Not a dynamic human reconstruction acceptance test',
                     'Opacity does not guarantee correct geometry', 'Hardware fit not yet measured for SVD'])
    dump(OUT / 'protocol.json', protocol)
    dump(OUT / 'status.json', dict(status=protocol['status'], clips=2, frames=50, inference_run=False))
    print(json.dumps(dict(status=protocol['status'], clips=2, frames=50)), flush=True)


def imports():
    sys.path.insert(0, DEPENDENCIES)
    sys.path.insert(0, str(REPO))
    import torch, diffusers, transformers, accelerate
    from svd_pipeline import StableVideoDiffusionPipeline
    from scheduler import EulerDiscreteScheduler
    return torch, StableVideoDiffusionPipeline, EulerDiscreteScheduler, dict(
        torch=torch.__version__, diffusers=diffusers.__version__, transformers=transformers.__version__,
        accelerate=accelerate.__version__, cuda=torch.cuda.is_available(), gpu=torch.cuda.get_device_name(0))


def preflight():
    *_, versions = imports()
    dump(OUT / 'environment-check.json', dict(status='official_pipeline_import_passed', versions=versions,
        dependency_path=DEPENDENCIES, base_training_packages_replaced=False, inference_run=False))
    print(json.dumps(versions), flush=True)


def infer(weights):
    import numpy as np
    from PIL import Image
    torch, Pipeline, Scheduler, versions = imports()
    protocol = json.loads((OUT / 'protocol.json').read_text())
    for name, expected in protocol['source_files'].items():
        assert digest(REPO / name) == expected, 'Official source changed after preparation'
    weights = Path(weights)
    required = ['model_index.json','unet/diffusion_pytorch_model.fp16.safetensors',
                'image_encoder/model.fp16.safetensors','vae/diffusion_pytorch_model.fp16.safetensors']
    assert all((weights / name).is_file() for name in required), 'Provide a complete official diffusers-format fp16 snapshot'
    dump(OUT / 'weight-receipt.json', {name:dict(bytes=(weights/name).stat().st_size, sha256=digest(weights/name)) for name in required})
    pipe = Pipeline.from_pretrained(str(weights), torch_dtype=torch.float16, variant='fp16', local_files_only=True)
    pipe.scheduler = Scheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.unet.enable_forward_chunking(chunk_size=1, dim=1)
    pipe.encode_video_latents = functools.partial(pipe.encode_video_latents, encode_chunk_size=2)
    torch.set_num_threads(4)
    for label, clip in protocol['clips'].items():
        dest = OUT / label
        dest.mkdir(exist_ok=False)
        assert digest(OUT / clip['file']) == clip['sha256']
        data = np.load(OUT / clip['file']); start=time.time(); torch.cuda.reset_peak_memory_stats()
        dump(OUT / 'status.json', dict(status='inference_running', clip=label, inference_run=True))
        with torch.inference_mode():
            result = pipe(data['rgb'].astype(np.float32)[None] / 255., data['mask'],
                height=1024, width=576, num_frames=25, decode_chunk_size=2, fps=6,
                generator=torch.manual_seed(42), num_inference_steps=25, resample_steps=3, guide_util=15).frames[0]
        assert len(result) == 25
        for i, frame in enumerate(result):
            if not isinstance(frame, Image.Image):
                arr = np.asarray(frame); assert np.isfinite(arr).all()
                frame = Image.fromarray(np.clip(arr * 255, 0, 255).round().astype(np.uint8))
            frame.save(dest / (clip['cameras'][i]['name'] + '.png'))
        torch.cuda.synchronize()
        dump(dest / 'summary.json', dict(status='generated_pending_visual_review', frames=25,
            elapsed_seconds=time.time()-start, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            versions=versions, geometry_updated=False, accepted=False))
    dump(OUT / 'status.json', dict(status='generated_pending_visual_review', clips=2, frames=50, inference_run=True, accepted=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action', choices=['prepare','preflight','infer']);parser.add_argument('--weights')
    args=parser.parse_args()
    if args.action=='infer':
        assert args.weights, 'Provide --weights for an authorized local snapshot'
        infer(args.weights)
    else: globals()[args.action]()
