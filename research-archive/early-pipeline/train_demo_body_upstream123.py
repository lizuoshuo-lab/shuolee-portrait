"""Whole-body input adapter calling the unmodified upstream training function.

OptimizationParams are read verbatim from the pinned public repository.
This is a local SMPL-X input adapter, not a complete official body release.
"""
import argparse
import ast
import fcntl
import hashlib
import importlib.util
import json
import os
import random
import shutil
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from run_official_gaussians123 import PROJECT, ASSETS, read, save, sha, now

DEMO = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/demo-vhap-initialization')
SOURCE = DEMO / 'smplify-vhap-pose-prior-long'
OUT = DEMO / 'body-gaussian-upstream-defaults'
UP = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/repos/GaussianAvatars_SMPLX')
COMMIT = 'e44c1a6c5fbb7f76089f14982eba98d94e553214'
SNAPSHOTS = [1000, 10000, 30000, 60000, 120000, 300000, 600000]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def prepare():
    from export_vhap_smplx_gaussian123 import run
    run(SimpleNamespace(source_run=str(SOURCE), checkpoint_epoch=120,
        inputs='/opt/gs-slam-lab/shuolee-local-surface-123/demo-vhap-fps3/data/inputs.json',
        output_root=str(OUT), reference_frame='frame_000033', max_seconds=180.))
    params_source = UP / 'arguments/__init__.py'
    tree = ast.parse(params_source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'OptimizationParams')
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    defaults = {n.targets[0].attr: ast.literal_eval(n.value) for n in init.body
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Attribute)}
    paths = [Path(__file__), UP/'train.py', params_source, UP/'scene/gaussian_model.py',
             UP/'scene/flame_gaussian_model.py', PROJECT/'src/avatar_core/scene/smplx_gaussian_model.py',
             PROJECT/'src/shuolee_portrait/training.py', PROJECT/'src/avatar_core/gaussian_renderer/__init__.py']
    save(OUT/'plan.json', dict(updated=now(), state='prepared', upstream_commit=COMMIT,
        user_authorization=['开始高斯训练吧', '全部用原版参数'], supersedes='高斯训练的vertexoffset调高一点',
        options=defaults, source_hashes={str(p): sha(p) for p in paths}, frames=59,
        training_function='Pinned upstream train.py::training, AST compiled without modifications',
        initial_points=20908, from_scratch=True, optimizer_state_reused=False,
        static_vertex_offset='Load VHAP fitted offset exactly once; frozen as upstream default; no offset optimizer group.',
        mesh_pose='Frozen59-frame VHAP fitted meshes. Official FLAME pose/expression fine-tuning has no complete public SMPL-X input entry and is not invented here.',
        image_loss='Original full-image0.8 L1 +0.2 DSSIM on RGB composited with the original mask over black; no added alpha or foreground-normalized loss.',
        input_adapters=['Native full-K/half-pixel camera and named per-frame SMPL-X meshes',
            'Use existing corrected face-local split units; no new strategy',
            'CPU uint8 cache and zero loader workers; shuffle and optimizer sequence retained',
            'Nonzero scene extent from fitted reference body diagonal in camera-relative gauge'],
        point_target='Original adaptive density; no forced200k cap or padded points.',
        snapshots=SNAPSHOTS, review_step=1000, max_points_safety_stop=1200000,
        quality_accepted=False, upstream_complete_reproduction=False,
        fit_quality_accepted=False, admission='User explicitly requests exploratory Gaussian training despite known fitted limb errors.',
        no_training_image_warp=True, no_old_gaussians=True, no_automatic_new_experiment=True))
    save(OUT/'progress.json', dict(updated=now(), state='prepared', pid=None, step=0, quality_accepted=False))
    print('BODY_UPSTREAM_PREPARED', flush=True)


def train():
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm
    lock = open('/tmp/shuolee-native122-gpu.lock', 'a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = read(OUT/'plan.json')
    for path, digest in plan['source_hashes'].items():
        assert sha(path) == digest, path
    receipt = read(OUT/'vhap-gaussian-interface-receipt.json')
    assert receipt['frames'] == 59 and receipt['replay_max_abs_m'] < 2e-6
    assert sha(OUT/'fit/fitted.npz') == receipt['fit_sha256']
    assert sha(OUT/'data/training-inputs.json') == receipt['training_inputs_sha256']
    run = OUT/'gaussian-run'
    run.mkdir(exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(4)
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    sys.path.insert(0, str(PROJECT/'src'))
    from shuolee_portrait import runtime as rt
    rt.configure(ASSETS)
    from shuolee_portrait.training import camera, named_pose_layout, image
    from scene.smplx_gaussian_model import SMPLXGaussianModel
    from gaussian_renderer import render
    from utils.loss_utils import l1_loss, ssim
    official = module(UP/'scene/gaussian_model.py', 'body_original_gaussian123')
    arguments = module(UP/'arguments/__init__.py', 'body_original_arguments123')
    parser = argparse.ArgumentParser()
    opt = arguments.OptimizationParams(parser).extract(parser.parse_args([]))
    assert vars(opt) == plan['options']
    with np.load(OUT/'fit/fitted.npz', allow_pickle=False) as archive:
        data = dict(archive)
    rows = read(OUT/'data/training-inputs.json')['records']
    layout = named_pose_layout(data, rows)
    assert layout['reference_frame_id'] == layout['initialization_frame_id'] == 'frame_000033'
    class Body(official.GaussianModel):
        def __init__(self, degree, *unused):
            super().__init__(degree)
            self.static_offset = None
            self.vertices_sequence = torch.as_tensor(data['vertices_world'], device='cuda', dtype=torch.float32)
            self.faces = torch.as_tensor(data['faces'], device='cuda', dtype=torch.long)
            self.num_timesteps = 59
            self.binding = torch.arange(len(self.faces), device='cuda')
            self.binding_counter = torch.ones(len(self.faces), device='cuda', dtype=torch.int32)
            self.select_mesh_by_timestep(0)
        select_mesh_by_timestep = SMPLXGaussianModel.select_mesh_by_timestep
        densify_and_split = SMPLXGaussianModel.densify_and_split
        get_covariance = SMPLXGaussianModel.get_covariance
        export_world_ply = SMPLXGaussianModel.export_world_ply
    cams, records, pixels, masks = [], [], [], []
    for i, row in enumerate(rows):
        assert sha(row['image']) == row['source_sha256'] and sha(row['mask']) == row['mask_sha256']
        cam, rec = camera(row, i, pose_index=layout['pose_indices'][i])
        cams.append(cam); records.append(rec)
        pixels.append(torch.from_numpy(np.array(Image.open(row['image']).convert('RGB'))).permute(2, 0, 1))
        masks.append(torch.from_numpy(np.array(Image.open(row['mask']).convert('L')))[None])
    counts = np.zeros(59, dtype=int)
    class Images:
        def __len__(self): return len(rows)
        def __getitem__(self, i):
            # Official dataset semantics: mask-composited image, black background.
            cams[i].original_image = pixels[i].float().div(255) * masks[i].float().div(255)
            counts[i] += 1
            return cams[i]
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    background = torch.zeros(3, device='cuda')
    model = Body(3)
    extent = float(np.linalg.norm(np.ptp(data['vertices_world'][0], axis=0)))
    assert 0 < extent < 10
    saved = []
    def status(state, step, **fields):
        save(OUT/'progress.json', dict(updated=now(), state=state, pid=os.getpid(), step=step,
            budget=opt.iterations, seconds=time.monotonic()-started, points=len(model._xyz),
            quality_accepted=False, **fields))
    class Scene:
        def __init__(self, dataset, gaussians):
            assert gaussians is model
            self.gaussians = model; self.model_path = str(run); self.cameras_extent = extent
            model.create_from_pcd(None, 1.)
        def getTrainCameras(self): return Images()
        def save(self, step):
            folder = run/f'step-{step:06d}'; folder.mkdir()
            old = model.timestep
            model.export_world_ply(folder/'world.ply', 0)
            model.save_ply(str(folder/'bound-local.ply'))
            torch.save(dict(step=step, capture=model.capture()), folder/'checkpoint.pt')
            np.savez_compressed(folder/'binding.npz', triangle_ids=model.binding.cpu().numpy(),
                frame_ids=data['frame_ids'], static_offset=data['static_offset'])
            plain = official.GaussianModel(3); plain.load_ply(str(folder/'world.ply'))
            plain.active_sh_degree = model.active_sh_degree
            model.select_mesh_by_timestep(0)
            delta = (render(cams[0], model, pipe, background)['render']-
                     render(cams[0], plain, pipe, background)['render']).abs()
            parity = dict(mean_abs=float(delta.mean()), max_abs=float(delta.max()))
            assert parity['mean_abs'] < 2e-6 and parity['max_abs'] < .005
            for i, row in enumerate(rows):
                model.select_mesh_by_timestep(layout['pose_indices'][i])
                rgb = render(cams[i], model, pipe, background)['render']
                image(rgb).save(folder/(row['frame']+'_gaussian.png'))
            model.select_mesh_by_timestep(old)
            save(folder/'result.json', dict(updated=now(), step=step, points=len(model._xyz),
                reference_frame='frame_000033', reference_slot=0, active_sh_degree=model.active_sh_degree,
                world_ply_sha256=sha(folder/'world.ply'), native_ply_parity=parity, frames_rendered=59,
                vertex_offset='Frozen final VHAP shared offset', quality_accepted=False))
            saved.append(step)
            for path, digest in plan['source_hashes'].items(): assert sha(path) == digest, path
            if step == 1000:
                decision = OUT/'review-1000-decision.json'
                assert not decision.exists()
                save(OUT/'review-1000-request.json', dict(updated=now(), step=1000,
                    snapshot=str(folder), world_ply_sha256=sha(folder/'world.ply')))
                status('review_pending', step, optimizer_waiting=True)
                while not decision.exists(): time.sleep(2)
                d = read(decision)
                assert d['world_ply_sha256'] == sha(folder/'world.ply') and d['decision'] in ('continue', 'stop')
                if d['decision'] == 'stop': raise RuntimeError('STOPPED_AFTER_ACTUAL_1000_REVIEW')
                status('training', step, optimizer_waiting=False)
    # Execute the pinned original function, with dataset/logging interfaces only.
    source = (UP/'train.py').read_text()
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'training')
    fn_source = ast.get_source_segment(source, fn)
    (run/'upstream-training-function.py').write_text(fn_source+'\n')
    save(run/'training-cameras.json', dict(records=records,
        note='These cameras require the named articulated mesh poses; identity extrinsics are not a recovered scene trajectory.'))
    save(run/'effective-optimization.json', vars(opt))
    def report(writer, step, losses, elapsed, testing, scene, render_fn, render_args):
        assert torch.isfinite(losses['total'])
        assert len(model._xyz) < plan['max_points_safety_stop']
        if step == 1:
            assert not model.optimizer.state and all(g['name'] != 'static_offset' for g in model.optimizer.param_groups)
            assert all(p.grad is not None and torch.isfinite(p.grad).all()
                       for g in model.optimizer.param_groups for p in g['params'])
            save(run/'preflight.json', dict(updated=now(), finite_backward=True,
                initial_points=len(model._xyz), optimizer_updates=0, source_geometry_parity=receipt['replay_max_abs_m'],
                max_camera_projection_error_px=max(r['projection_error_px'] for r in records),
                original_optimization_defaults_equal=True, static_offset_optimizer_present=False,
                loss_keys=list(losses), pose_finetuning=False,
                optimizer_groups=[dict(name=g['name'], lr=g['lr']) for g in model.optimizer.param_groups]))
        if step == 1 or step % 100 == 0:
            values={k: float(v) for k,v in losses.items()}
            status('training', step, loss=values, per_step_gpu_ms=elapsed, views_seen=int((counts>0).sum()))
            with (run/'loss.jsonl').open('a') as log:
                log.write(json.dumps(dict(step=step, seconds=time.monotonic()-started, **values))+'\n')
            print(json.dumps(dict(step=step, points=len(model._xyz), loss=values['total'])), flush=True)
        if step % 1000 == 0:
            assert shutil.disk_usage(run).free > 15*2**30, 'Insufficient checkpoint disk space'
    def loader(dataset, **kwargs):
        kwargs.update(num_workers=0, persistent_workers=False, pin_memory=False)
        return torch.utils.data.DataLoader(dataset, **kwargs)
    namespace=dict(torch=torch, F=F, DataLoader=loader, prepare_output_and_logger=lambda d:None,
        FlameGaussianModel=lambda *args:model, GaussianModel=official.GaussianModel, Scene=Scene,
        NVDiffRenderer=lambda:None, network_gui=SimpleNamespace(conn=None,try_connect=lambda:None),
        render=render, l1_loss=l1_loss, ssim=ssim, training_report=report,
        tqdm=lambda *args,**kwargs:tqdm(*args,**kwargs,disable=True))
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(UP/'train.py'),'exec'), namespace)
    save(run/'provenance.json', dict(updated=now(), upstream_commit=COMMIT,
        training_function_sha256=hashlib.sha256(fn_source.encode()).hexdigest(),
        unmodified_training_function=True, unchanged_optimization_defaults=True,
        from_scratch=True, input_mesh_sha256=receipt['fit_sha256'], frozen_static_offset=True,
        source_hashes=plan['source_hashes'], local_full_body_adapter=True,
        full_official_reproduction=False, old_checkpoint_loaded=False))
    dataset=SimpleNamespace(bind_to_mesh=True, sh_degree=3, disable_flame_static_offset=False,
        not_finetune_flame_params=True, white_background=False, model_path=str(run))
    status('training_starting', 0)
    namespace['training'](dataset,opt,pipe,[],SNAPSHOTS,[],None,-1)
    save(run/'result.json', dict(updated=now(), state='completed_pending_quality_review',
        steps=opt.iterations, points=len(model._xyz), seconds=time.monotonic()-started,
        per_frame_counts=counts.tolist(), snapshots=saved, quality_accepted=False))
    save(OUT/'progress.json', dict(updated=now(), state='completed_pending_quality_review',pid=None,
        step=opt.iterations, points=len(model._xyz), quality_accepted=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['prepare','train'])
    args=parser.parse_args()
    try: prepare() if args.mode=='prepare' else train()
    except BaseException as error:
        if OUT.exists():
            save(OUT/(args.mode+'-failure.json'), dict(updated=now(), error=repr(error),
                traceback=traceback.format_exc(), auto_retry=False, quality_accepted=False))
            save(OUT/'progress.json',dict(updated=now(),state='failed_preserved',pid=None,error=repr(error),quality_accepted=False))
        raise
