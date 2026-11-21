"""Run the actual CUDA renderer on a face-only observation and a full-body control."""
from pathlib import Path
import json
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS

import numpy as np
from PIL import Image
import torch

PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-face-supplement-20261111')
sys.path.insert(0, str(PROJECT/'src'))
from shuolee_portrait import runtime as rt
from shuolee_portrait.face_supplement_training import masked_face_loss, prepare_targets, save_replay, train_face_view
from shuolee_portrait.render_support import replacement_evaluator, source_weights
from shuolee_portrait.training import camera
from shuolee_portrait.world_scale import scale_config
from scene.smplx_gaussian_model import SMPLXGaussianModel
from diff_gaussian_rasterization import _C
from gaussian_renderer import render
from probe_cuda_size_optimization_round17 import build_model, make_constraints

PROOF = Path('/mnt/c/Users/Administrator/Documents/New project/outputs/smpl-delivery-round18/input-audit/cuda-face-supplement-proof.json')
FILES = ('face_supplement.py','face_supplement_training.py','render_support.py','training.py','world_scale.py')

def masked_loss_probe():
    source = torch.zeros(3, 9, 9, device='cuda')
    pred = torch.full_like(source, .25, requires_grad=True)
    alpha = torch.full((9, 9), .5, device='cuda', requires_grad=True)
    valid = torch.zeros((9, 9), dtype=torch.bool, device='cuda')
    valid[2:5, 3:6] = True
    target = torch.ones_like(alpha)
    exact = masked_face_loss(pred, alpha, source, target, valid)
    other_rgb, other_alpha = source.clone(), target.clone()
    other_rgb[:, ~valid] = torch.nan
    other_alpha[~valid] = torch.nan
    altered = masked_face_loss(pred, alpha, other_rgb, other_alpha, valid)
    assert all(torch.equal(a,b) for a,b in zip(exact, altered, strict=True))
    sum(exact).backward()
    assert (pred.grad[:, ~valid] == 0).all() and (alpha.grad[~valid] == 0).all()
    assert (pred.grad[:, valid] != 0).all() and (alpha.grad[valid] != 0).all()
    try:
        masked_face_loss(pred, alpha, source, target, torch.zeros_like(valid))
    except ValueError as error:
        assert 'zero verified' in str(error)
    else:
        raise AssertionError('Empty admitted ROI entered optimization.')
    return dict(unknown_nan_loss_unchanged=True, unknown_gradients_zero=True,
                black_foreground_rgb_alpha_gradients_nonzero=True, empty_roi_rejected=True)

def bound_mesh():
    initial = build_model()
    model = SMPLXGaussianModel(0)
    vars(model).update(vars(initial))
    points = np.array([[-.32,-.20,2.],[.32,-.20,2.],[0,.37,2.],
                       [.9,-.20,2.],[1.54,-.20,2.],[1.22,.37,2.]],np.float32)
    vertices = np.repeat(points[None],25,axis=0)
    vertices[24,:,1] += .075
    model.vertices_sequence = torch.from_numpy(vertices).cuda()
    model.supplement_vertices = model.vertices_sequence[24:]
    model.faces = torch.tensor([[0,1,2],[3,4,5]], dtype=torch.long,device='cuda')
    model.num_timesteps = 24
    with torch.no_grad():
        model._opacity[0].fill_(2.)
    model.select_mesh_by_timestep(0)
    return model

def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    loss_proof = masked_loss_probe()
    with TemporaryDirectory(prefix='face-supplement-cuda-') as tmp:
        directory = Path(tmp)
        model = bound_mesh()
        constraints = make_constraints(model, directory)
        model.select_mesh_by_timestep(24)
        row = dict(constraints.rows[0], frame='fixture-face', image=str(directory/'face-source.png'),
                   source_frame_index=154, mask_sha256='1'*64)
        row['w2c'] = np.asarray(row['w2c']).copy()
        row['w2c'][0,3] += .048
        cam, _ = camera(row,24,pose_index=24)
        y0,y1,x0,x1 = 38,76,48,83
        box = x0,y0,x1,y1
        with torch.no_grad():
            seen = render(cam,model,constraints.pipe,constraints.black,
                          override_color=torch.ones_like(model.get_xyz))['render'][0,y0:y1,x0:x1]
            valid = (seen > .65).cpu().numpy()
        assert valid.sum() > 4
        Image.fromarray(np.zeros((112,128,3),np.uint8)).save(row['image'])
        row['source_sha256'] = rt.sha(Path(row['image']))
        item = NS(row=row, box=box, valid=valid, alpha=valid.astype(np.float32))
        [target] = prepare_targets([item],24,camera,directory)
        assert target.camera_record['pose_index']==24 and target.camera_record['projection_error_px']<.001
        assert torch.count_nonzero(target.guard_view.outside) == 0
        model.select_mesh_by_timestep(0)
        primary_xyz = model.get_xyz.detach().clone()
        model.select_mesh_by_timestep(24)
        assert not torch.equal(primary_xyz,model.get_xyz)
        model.select_mesh_by_timestep(0)
        head_faces = torch.tensor([True,False],device='cuda')
        train_face_view(target, model, head_faces, render, constraints.pipe, constraints.black, 2401, density=True)
        assert model.timestep==0 and target.record['updates']==1 and target.record['density_observations']==1
        assert target.record['max_density_visible_rows']>0
        for name in ('_xyz','_scaling','_rotation','_opacity','_features_dc'):
            grad = getattr(model,name).grad
            assert grad is not None and torch.isfinite(grad).all() and torch.count_nonzero(grad[1:])==0, name
        assert torch.count_nonzero(model._features_dc.grad[0])>0
        model.optimizer.zero_grad(set_to_none=True)
        constraints.face_supplement = [target]
        parent = torch.tensor([0],device='cuda')
        weights = source_weights(model,constraints,parent)
        with replacement_evaluator(model,constraints,parent,None,None,None,weights,scale_config()['render_support']) as trial:
            result=trial([0])
        [face_trial] = result['face_supplement_frames']
        assert len(result['frames'])==24 and not result['accepted'] and face_trial['failures']
        assert face_trial['background_gain_px']==0 and face_trial['frame']=='fixture-face'
        reloaded = SMPLXGaussianModel(0)
        for name in ('_xyz','_scaling','_rotation','_opacity','_features_dc','_features_rest','binding','faces'):
            setattr(reloaded,name,getattr(model,name))
        reloaded.vertices_sequence = model.vertices_sequence[:24]
        reloaded.select_mesh_by_timestep(0)
        with torch.no_grad():
            save_replay(directory,[target],model,reloaded,dict(bound_ply_sha256='3'*64),render,constraints.pipe,constraints.black)
        [replay] = rt.read(directory/'replay.json')['records']
        assert replay['valid_rgb_max_abs'] < .005 and replay['geometry_parity']['center_max_abs_m']==0
        assert model.timestep==reloaded.timestep==0 and len(reloaded.vertices_sequence)==24
        data = dict(state='passed', gpu=torch.cuda.get_device_name(),
            runtime=dict(torch=torch.__version__, cuda=torch.version.cuda, raster_extension_sha256=rt.sha(Path(_C.__file__))),
            source_sha256={f'src/shuolee_portrait/{name}':rt.sha(PROJECT/'src/shuolee_portrait'/name) for name in FILES},
            fixture_sha256=rt.sha(Path(__file__)), pixel_loss=loss_proof,
            bound_scene=dict(body_gradient_rows=0, visible_head_gradient_rows=target.record['max_density_visible_rows'],
                same_optimizer_update=True, full_view_count=len(result['frames']), face_guard=face_trial,
                pose_index=replay['pose_index'], exact_pose_geometry=replay['geometry_parity'],
                native_replay_rgb_max_abs=replay['valid_rgb_max_abs']))
        rt.save(PROOF,data)
        print(json.dumps(dict(state='passed', path=str(PROOF), sha256=rt.sha(PROOF)),ensure_ascii=False),flush=True)

if __name__=='__main__':
    main()

