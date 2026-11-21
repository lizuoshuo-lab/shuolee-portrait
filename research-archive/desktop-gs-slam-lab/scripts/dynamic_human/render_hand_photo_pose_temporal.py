"""Render unchanged native temporal windows for the pose-only comparison."""
from pathlib import Path
import json, hashlib, shutil
import numpy as np, torch
from PIL import Image
from enerf_second_temporal_graph import GraphMotion
from enerf_hand_joint_model import HandJointRenderer
from enerf_mano_gaussian_surface import ManoGaussianSurface, render_hand_surface

L = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
S = L/'outputs/dynamic-human-enerf-second-interval-079'
U = L/'outputs/dynamic-human-enerf-hand-time-scope-088'
Q = L/'outputs/dynamic-human-enerf-contour-surface-103'
O = L/'outputs/dynamic-human-hand-photo-pose-105'
T = O/'temporal'; T.mkdir(exist_ok=True)
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
save = lambda p,x: p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
tt = lambda x: torch.as_tensor(x,dtype=torch.float32,device='cuda')
u8 = lambda x: np.rint(np.clip(x,0,1)*255).astype(np.uint8)
assert not (T/'completion.json').exists()
assert len(read(O/'native-evaluation/metrics.json')['records']) == 134
plan = read(U/'temporal/protocol.json'); save(T/'protocol.json',plan)
audit = {(r['camera'],r['frame']):r for r in read(S/'input-audit.json')['records']}
photo = T/'source'; photo.mkdir(exist_ok=True)
inputs = {}; checks = []; torch.set_num_threads(6)
for c in plan['cameras']:
    for f in plan['frames']:
        stem = f'camera{c}-frame{f:06d}'; path = S/'native-input'/f'{stem}.npz'
        assert sha(path) == audit[c,f]['input_sha256']
        z = np.load(path)
        inputs[c,f] = dict(W=tt(z['w2c']),K=tt(z['K']),bbox=plan['boxes'][c],input_sha256=sha(path))
        for suffix in ['npz','png']:
            shutil.copy2(U/'temporal/source'/f'{stem}.{suffix}',photo/f'{stem}.{suffix}')
for b, previous in [('parent','parent'),('fixed_pose','contour_prior')]:
    dest = T/b; dest.mkdir(exist_ok=True)
    for c,f in inputs:
        stem = f'camera{c}-frame{f:06d}'
        for suffix in ['npz','png']:
            src = Q/'temporal'/previous/f'{stem}.{suffix}'
            shutil.copy2(src,dest/src.name); assert sha(src) == sha(dest/src.name)
        checks.append(dict(branch=b,camera=c,frame=f,reused103=True))
path = O/'resume.pt'; before = sha(path)
cp = torch.load(path,map_location='cpu',weights_only=False)
assert cp['renderer'] == 'mano_photo_pose105'
p = {k:v.cuda() for k,v in cp['params'].items()}
g = GraphMotion(S/'graph.npz',cp['scene_scale']); g.load_state_dict(cp['graph']); g.requires_grad_(False)
active = torch.from_numpy(np.load(Q/'context-reset-mask.npy')).cuda()
old = HandJointRenderer(S/'heldout/capture-affines-estimated.npz',g,g,active)
qp = read(Q/'protocol.json')
hand = ManoGaussianSurface(L/qp['priors']['contour_prior']['path'],S/'heldout/capture-affines-estimated.npz',cp['scene_scale'],qp['initial_color'],True)
hand.load_state_dict(cp['hand_surface']); hand.requires_grad_(False)
dest = T/'photo_pose'; dest.mkdir(exist_ok=True)
with torch.no_grad():
    for (c,f),o in inputs.items():
        raw,alpha,_ = render_hand_surface(p,old,hand,c,f,o['W'],o['K'],1920,1080,emit_surface_weight=True)
        x0,y0,x1,y1 = o['bbox']; r = raw[0,y0:y1,x0:x1,:3].cpu().numpy(); a = alpha[0,y0:y1,x0:x1,0].cpu().numpy()
        assert np.isfinite(r).all() and np.isfinite(a).all()
        stem = f'camera{c}-frame{f:06d}'
        np.savez_compressed(dest/f'{stem}.npz',rgb=r,alpha=a)
        Image.fromarray(u8(np.clip(r+.5*(1-a[...,None]),0,1))).save(dest/f'{stem}.png')
        error = None
        if f == 731:
            existing = np.load(O/'native-evaluation'/f'{stem}-prediction.npz')
            error = max(float(abs(r-existing['rgb'][y0:y1,x0:x1]).max()),float(abs(a-existing['alpha'][y0:y1,x0:x1]).max()))
            assert error < 2e-6, error
        checks.append(dict(branch='photo_pose',camera=c,frame=f,input_sha256=o['input_sha256'],bbox=o['bbox'],checkpoint_sha256=before,evaluation_reproduction_max_error=error))
        print('POSE105_TEMPORAL',c,f,flush=True)
assert sha(path) == before and len(checks) == 78
save(T/'completion.json',dict(status='26new_native_ROIs_plus52_exact_reused103',records=checks,frames_per_camera=13,cameras=plan['cameras'],goal_complete=False))
shutil.copy2(__file__,T/Path(__file__).name)
