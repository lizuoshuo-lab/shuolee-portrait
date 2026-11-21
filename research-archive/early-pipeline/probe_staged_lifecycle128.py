"""Bounded synthetic native-CUDA checkpoint lifecycle control; no person fitting.

The pinned original training function is executed unchanged on a tetrahedron.
Its total schedule remains18000; the registered test stop is24, with a staged
boundary at8. This is an engineering control, never a reconstruction score.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np

H = Path(__file__).resolve().parent
LAB = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
OUT = LAB/'outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/checkpoint-lifecycle128'
UP = LAB/'repos/GaussianAvatars_SMPLX'
TRAIN_SHA = 'db308103af3b7cd7a6718fb17e168bea43725bdf4cb0be07899964afc3f9230a'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8')


def run(mode):
    import fcntl
    import signal
    lock = open('/tmp/shuolee-native122-gpu.lock', 'a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Synthetic180s limit')))
    signal.alarm(180)
    import torch
    import torch.nn.functional as F
    from types import SimpleNamespace
    from tqdm import tqdm
    from run_official_gaussians123 import ASSETS, PROJECT
    from train_demo_body_upstream123 import module
    from staged_training_lifecycle128 import StagedTrainingLifecycle, ReviewStop
    import sys
    sys.path.insert(0, str(PROJECT/'src'))
    from shuolee_portrait import runtime
    runtime.configure(ASSETS)
    from shuolee_portrait.training import camera
    from gaussian_renderer import render
    from utils.loss_utils import l1_loss, ssim
    from utils.graphics_utils import compute_face_orientation
    from roma import rotmat_to_unitquat, quat_xyzw_to_wxyz
    from scene.smplx_gaussian_model import SMPLXGaussianModel
    from native_target_cache124 import NativeTargetCache
    from paired_training_background124 import PairedTrainingBackground

    assert sha(UP/'train.py') == TRAIN_SHA
    official = module(UP/'scene/gaussian_model.py', 'lifecycle128_original_gaussians')
    args = module(UP/'arguments/__init__.py', 'lifecycle128_original_arguments')
    parser = argparse.ArgumentParser()
    opt = args.OptimizationParams(parser).extract(parser.parse_args([]))
    # These compressed timings belong only to this synthetic control.
    opt.iterations = 18000
    opt.position_lr_max_steps = 18000
    opt.densify_from_iter = 4
    opt.densification_interval = 4
    opt.densify_until_iter = 25
    opt.opacity_reset_interval = 12
    opt.densify_grad_threshold = 0.0
    opt.lambda_dynamic_offset = 0
    opt.lambda_dynamic_offset_std = 0
    opt.lambda_laplacian = 0.01

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(128); np.random.seed(128); torch.manual_seed(128); torch.cuda.manual_seed_all(128)
    branch = 'continuous' if mode == 'continuous' else 'staged'
    dest = OUT/branch
    resume = mode == 'resume'
    if not resume:
        dest.mkdir(parents=True, exist_ok=False)
    else:
        assert dest.is_dir() and (dest/'chkpnt8.pth').is_file()
    started = time.monotonic()
    frames = [f'synthetic_{i}' for i in range(5)]
    cams = []
    for i, frame in enumerate(frames):
        E = np.eye(4); E[0, 3] = (i-2)*0.05
        row = dict(frame=frame, wh=[64, 64], K=[[70.,0.,31.25],[0.,73.,33.5],[0.,0.,1.]], w2c=E.tolist())
        cam, _ = camera(row, i, pose_index=i)
        cam.supervision_frame = frame
        cams.append(cam)
    yy, xx = np.mgrid[:64,:64]
    masks = [torch.from_numpy((((xx-32)**2+(yy-32)**2)<24**2).astype(np.uint8)*255)[None] for _ in frames]
    pixels = [torch.from_numpy(np.stack([(xx*3+i*13)%256,(yy*3+i*9)%256,((xx+yy)*2+i*7)%256]).astype(np.uint8)) for i in range(5)]
    cache = NativeTargetCache(pixels, masks, frames)
    counts = [0]*len(frames)
    traces = []
    class Images:
        def __len__(self): return len(frames)
        def __getitem__(self, i):
            counts[i] += 1
            cams[i].original_image = cache.get(i, frames[i])
            return cams[i]

    class Body(official.GaussianModel):
        def __init__(self, degree, *unused):
            super().__init__(degree)
            self.vertices = torch.tensor([[-.45,-.35,2.6],[.45,-.35,2.6],[0.,.5,2.6],[0.,0.,3.1]], device='cuda')
            self.faces = torch.tensor([[0,1,2],[0,3,1],[1,3,2],[2,3,0]], device='cuda')
            self.static_offset = torch.nn.Parameter(torch.zeros_like(self.vertices))
            self.binding = torch.arange(4, device='cuda')
            self.binding_counter = torch.ones(4, dtype=torch.int32, device='cuda')
            self.num_timesteps = len(frames)
            self.select_mesh_by_timestep(0)
        def select_mesh_by_timestep(self, timestep, original=False):
            self.timestep = int(timestep)
            verts = self.vertices+self.static_offset
            self.face_center = verts[self.faces].mean(1)
            self.face_orien_mat, scale = compute_face_orientation(verts, self.faces, return_scale=True)
            self.face_scaling = scale.clamp_min(1e-8)
            self.face_orien_quat = quat_xyzw_to_wxyz(rotmat_to_unitquat(self.face_orien_mat))
        def training_setup(self, settings):
            super().training_setup(settings)
            self.optimizer.add_param_group(dict(params=[self.static_offset], lr=1e-6, name='static_offset'))
        def compute_laplacian_loss(self): return self.static_offset.square().mean()
        densify_and_split = SMPLXGaussianModel.densify_and_split
        get_covariance = SMPLXGaussianModel.get_covariance

    model = Body(0)
    paired = PairedTrainingBackground(render, masks, frames, cache)
    def counters_getter(): return dict(counts=list(counts), traces=list(traces))
    def counters_setter(state):
        counts[:] = state['counts']; traces[:] = state['traces']
    stop = 8 if mode == 'stage' else 24
    identity = dict(schema='synthetic-tetra-lifecycle128', frames=frames, camera_K=[70.,73.,31.25,33.5],
        total_iterations=18000, options=vars(opt), person_data_used=False)
    life = StagedTrainingLifecycle(model=model, paired_background=paired, camera_ids=frames,
        identity=identity, source_hashes={str(UP/'train.py'):TRAIN_SHA,str(H/'staged_training_lifecycle128.py'):sha(H/'staged_training_lifecycle128.py'),str(Path(__file__)):sha(__file__)},
        total_iterations=18000, stop_iterations=[stop], optimization=opt,
        counters_getter=counters_getter, counters_setter=counters_setter)
    class Scene:
        def __init__(self, dataset, gaussians):
            assert gaussians is model
            self.model_path = str(dest); self.cameras_extent = 1.5; self.gaussians = model
            model.create_from_pcd(None, 1.)
        def getTrainCameras(self): return Images()
        def save(self, iteration): raise AssertionError('Synthetic control has no pre-update preview save')
    def report(writer, step, losses, *unused):
        assert torch.isfinite(losses['total']) and len(model._xyz) < 1024
        traces.append(dict(step=step, frame=cams[model.timestep].supervision_frame,
            loss=float(losses['total']), python_random=random.random(), numpy_random=float(np.random.random()),
            cpu_random=float(torch.rand(())), cuda_random=float(torch.rand((),device='cuda')),
            background_state_sha=hashlib.sha256(paired.generator.get_state().cpu().numpy().tobytes()).hexdigest(),points=len(model._xyz)))
    source = (UP/'train.py').read_text()
    tree = ast.parse(source); fn = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='training')
    namespace = dict(torch=life.torch_proxy(torch), F=F, DataLoader=life.camera_loader,
        prepare_output_and_logger=lambda _:None, FlameGaussianModel=lambda *args:model,
        GaussianModel=official.GaussianModel, Scene=Scene, NVDiffRenderer=lambda:None,
        network_gui=SimpleNamespace(conn=None,try_connect=lambda:None), render=paired,
        l1_loss=l1_loss, ssim=ssim, training_report=report,
        tqdm=lambda *a,**kw:tqdm(*a,**kw,disable=True))
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(UP/'train.py'),'exec'),namespace)
    dataset = SimpleNamespace(bind_to_mesh=True,sh_degree=0,disable_flame_static_offset=False,
        not_finetune_flame_params=True,white_background=False,model_path=str(dest))
    pipe = SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
    stopped = False
    try:
        checkpoint_steps = [8,24] if mode == 'continuous' else [stop]
        namespace['training'](dataset,opt,pipe,[],[],checkpoint_steps,str(dest/'chkpnt8.pth') if resume else None,-1)
    except ReviewStop:
        stopped = True
    assert stopped and traces[-1]['step'] == stop and sum(counts) == stop
    assert model.static_offset.grad is None
    def cpu_tree(value):
        if torch.is_tensor(value): return value.detach().cpu().clone()
        if isinstance(value,dict): return {k:cpu_tree(v) for k,v in value.items()}
        if isinstance(value,tuple): return tuple(cpu_tree(v) for v in value)
        if isinstance(value,list): return [cpu_tree(v) for v in value]
        return value
    torch.save(cpu_tree(dict(capture=model.capture(),static_offset=model.static_offset,counters=counters_getter(),
        python_rng=random.getstate(),numpy_rng=np.random.get_state(),cpu_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state(),paired_rng=paired.generator.get_state(),paired_calls=paired.calls)),dest/f'actual-state-{stop:02d}.pt')
    with torch.no_grad():
        model.select_mesh_by_timestep(0)
        rendered = render(cams[0], model, pipe, torch.zeros(3,device='cuda'))['render'].cpu()
    torch.save(rendered,dest/f'actual-render-{stop:02d}.pt')
    save(dest/f'result-{stop:02d}.json',dict(mode=mode,step=stop,points=len(model._xyz),counts=counts,seconds=time.monotonic()-started,
        actual_original_training_function=True,source_sha=TRAIN_SHA,quality_accepted=False,synthetic_only=True,training_person=False,
        actual_resume_receipt=life.last_resume_receipt,SH0_rest_tensor_elements=int(model._features_rest.numel()),
        empty_SH0_rest_group_not_a_nonzero_gradient_claim=True,static_offset_max_abs=float(model.static_offset.detach().abs().max()),
        final_opacity_max=float(model.get_opacity.detach().max())))
    signal.alarm(0)
    print('SYNTHETIC_LIFECYCLE_BOUNDARY_SAVED',mode,stop,len(model._xyz),flush=True)


def compare():
    import torch
    a=torch.load(OUT/'continuous/actual-state-24.pt',map_location='cpu',weights_only=False)
    b=torch.load(OUT/'staged/actual-state-24.pt',map_location='cpu',weights_only=False)
    comparisons=[]
    def exact(x,y,key):
        if torch.is_tensor(x):
            assert torch.is_tensor(y) and x.shape==y.shape and x.dtype==y.dtype,key
            assert torch.equal(x,y),key
        elif isinstance(x,np.ndarray): assert isinstance(y,np.ndarray) and np.array_equal(x,y),key
        elif isinstance(x,dict):
            assert x.keys()==y.keys(),key
            for k in x: exact(x[k],y[k],key+'.'+str(k))
        elif isinstance(x,(tuple,list)):
            assert type(x)==type(y) and len(x)==len(y),key
            for i,(v,w) in enumerate(zip(x,y)): exact(v,w,key+f'[{i}]')
        else: assert x==y,(key,x,y)
        comparisons.append(key)
    exact(a,b,'final_state')
    ra=torch.load(OUT/'continuous/actual-render-24.pt',map_location='cpu',weights_only=True)
    rb=torch.load(OUT/'staged/actual-render-24.pt',map_location='cpu',weights_only=True)
    assert torch.equal(ra,rb)
    assert len(a['counters']['traces'])==24 and len(set(a['counters']['counts']))>=1
    points=[x['points'] for x in a['counters']['traces']]
    assert len(set(points))>1,'Must actually exercise original densification'
    assert a['static_offset'].abs().max()>0,'Shared offset must actually update'
    assert torch.sigmoid(a['capture'][6]).max()<=.0100001,'Final original reset must be exercised'
    assert a['capture'][3].numel()==0,'SH0 rest group is intentionally empty'
    result=dict(passed=True,synthetic_only=True,original_training_function=True,stage_step=8,end_step=24,
        original_total_clock=18000,all_parameters_optimizer_RNG_camera_background_exact=True,
        native_render_max_abs=float((ra-rb).abs().max()),recursive_exact_comparisons=len(comparisons),
        points_per_step=points,completed_original_updates=24,quality_accepted=False,person_model_trained=False)
    save(OUT/'result.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['continuous','stage','resume','compare'])
    mode=p.parse_args().mode
    compare() if mode=='compare' else run(mode)
