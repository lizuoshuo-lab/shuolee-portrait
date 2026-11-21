"""Check the existing full-body Gaussian consumer without optimizer updates.

Calls the production pose selector, cameras, mask loss and CUDA renderer on
the completed named fits. Does not run the training loop or admit fit quality.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import fcntl
import os
import sys
import time
import traceback
import numpy as np
import run_official_gaussians123 as io


def run(plan_path):
    plan = io.read(plan_path)
    root, out = Path(plan['source_root']), Path(plan['output_root'])
    out.mkdir(exist_ok=False)
    began = time.monotonic()

    def status(state, **fields):
        io.save(out/'progress.json', dict(state=state, updated=io.now(),
            pid=None if state in ('complete', 'failed') else os.getpid(),
            training_started=False, elapsed_seconds=time.monotonic()-began, **fields))

    def inputs_unchanged():
        for path, digest in plan['source_hashes'].items():
            assert io.sha(path) == digest, path
        for path, digest in plan['input_hashes'].items():
            assert io.sha(root/path) == digest, path
        assert io.sha(__file__) == plan['probe_sha256']

    def budget():
        if time.monotonic()-began > plan['max_seconds']:
            raise TimeoutError('Read-only Gaussian input check exceeded registered time')

    status('checking_inputs')
    try:
        inputs_unchanged()
        with open('/tmp/shuolee-native122-gpu.lock', 'a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            import torch
            torch.set_num_threads(4)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.manual_seed(6060)
            torch.cuda.manual_seed_all(6060)
            np.random.seed(6060)
            sys.path.insert(0, str(io.PROJECT/'src'))
            from shuolee_portrait import runtime as rt
            rt.configure(io.ASSETS)
            from shuolee_portrait.training import (verify_official_fit, named_pose_layout,
                Views, render_pose, foreground_loss)
            from scene.smplx_gaussian_model import SMPLXGaussianModel
            from gaussian_renderer import render
            provenance = verify_official_fit(root)
            with np.load(root/'fit/fitted.npz', allow_pickle=False) as archive:
                fit = dict(archive)
            rows = io.read(root/'data/training-inputs.json')['records']
            assert len(rows) == plan['expected_frames'] == 59
            layout = named_pose_layout(fit, rows)
            assert layout['initialization_frame_id'] == layout['reference_frame_id'] == 'frame_000033'
            views = Views(rows, 1520, 3, pose_indices=layout['pose_indices'])
            model = SMPLXGaussianModel(3)
            model.load_fresh_meshes(root/'fit/fitted.npz')
            initial = model.initialize_surface(20908, 6060)
            offset = model.enable_shared_static_offset(root/'fit/fitted.npz', io.ASSETS/'smplx/SMPLX_NEUTRAL.npz')
            assert torch.equal(model.binding, torch.arange(20908, device='cuda'))
            params = (model._xyz, model._features_dc, model._features_rest, model._opacity,
                      model._scaling, model._rotation, model.static_offset)
            names = ('local_xyz', 'sh_dc', 'sh_rest', 'opacity', 'logscale', 'rotation', 'static_offset')
            before = [p.detach().clone() for p in params]
            pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
            bg = torch.zeros(3, device='cuda')
            records, gradients = [], []
            for i, row in enumerate(rows):
                budget()
                pose = layout['pose_indices'][i]
                with torch.no_grad():
                    model.select_mesh_by_timestep(pose)
                    expected = model.vertices_sequence[pose][model.faces].mean(1)
                    center_error = float((model.get_xyz-expected).abs().max())
                    rot = model.face_orien_mat
                    orthogonal_error = float((rot.transpose(1, 2)@rot-torch.eye(3, device='cuda')).abs().max())
                    det_min = float(torch.linalg.det(rot).min())
                    assert center_error < 2.e-6 and orthogonal_error < 2.e-5 and det_min > .9999
                    assert bool(torch.isfinite(model.get_scaling).all() and (model.get_scaling > 0).all())
                    xyz = model.vertices_sequence[pose].cpu().numpy().astype(np.float64)
                    E, K = np.asarray(row['w2c']), np.asarray(row['K'])
                    camera_xyz = xyz@E[:3, :3].T+E[:3, 3]
                    assert np.all(camera_xyz[:, 2] > 0)
                    q = camera_xyz@K.T
                    expected_uv = q[:, :2]/q[:, 2:]-.5
                    clip = np.c_[xyz, np.ones(len(xyz))]@views.cameras[i].full_proj_transform.numpy()
                    native_uv = ((clip[:, :2]/clip[:, 3:]+1)*row['wh']-1)/2
                    projection_error = float(np.abs(native_uv-expected_uv).max())
                    assert projection_error < .001
                    records.append(dict(frame=row['frame'], pose_index=pose,
                        centroid_max_abs_m=center_error, frame_orthogonal_max_abs=orthogonal_error,
                        frame_determinant_min=det_min, all_vertices_projection_max_abs_px=projection_error))
                if row['frame'] in plan['backward_frames']:
                    cam, gt, alpha = views.get(i)
                    assert tuple(gt.shape) == (3, 1520, 2704)
                    pkg = render_pose(render, cam, model, pipe, bg, pose)
                    photo, _, _ = foreground_loss(pkg['render'], gt, alpha)
                    changed_gt = gt.clone()
                    changed_gt[:, alpha[0] <= .99] = .37
                    other = foreground_loss(pkg['render'], changed_gt, alpha)[0]
                    outside_effect = float((photo-other).detach().abs())
                    prediction_gradient = torch.autograd.grad(photo, pkg['render'], retain_graph=True)[0]
                    outside_gradient = float(prediction_gradient[:, alpha[0] <= .99].abs().max())
                    aa = render_pose(render, cam, model, pipe, bg, pose,
                        override_color=torch.ones_like(model.get_xyz))['render'][:1]
                    loss = photo+.1*(aa-alpha).abs().mean()
                    grad = torch.autograd.grad(loss, params)
                    assert outside_effect < 1.e-7 and outside_gradient == 0
                    assert all(bool(torch.isfinite(g).all()) for g in grad)
                    assert float(grad[-1].norm()) > 0
                    gradients.append(dict(frame=row['frame'], loss=float(loss.detach()),
                        background_RGB_loss_effect=outside_effect, photometric_background_gradient=outside_gradient,
                        visible_gaussians=int(pkg['visibility_filter'].sum()),
                        gradient_norms={name: float(g.norm()) for name, g in zip(names, grad)}))
                    del pkg, photo, other, changed_gt, prediction_gradient, aa, loss, grad, gt, alpha
                if (i+1)%10 == 0:
                    status('checking_poses', poses_completed=i+1)
            assert len(gradients) == len(plan['backward_frames'])
            assert all(torch.equal(p, b) for p, b in zip(params, before))
            assert getattr(model, 'optimizer', None) is None
            assert not (root/'gaussian-run').exists()
            inputs_unchanged()
            budget()
            torch.cuda.synchronize()
            result = dict(state='complete', updated=io.now(), frames=len(records),
                initialization=initial, zero_offset=offset, pose_layout=layout,
                records=records, native_backward=gradients, official_input_provenance=provenance,
                elapsed_seconds=time.monotonic()-began,
                peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                optimizer_created=False, optimizer_updates=0, learned_parameters_unchanged=True,
                production_source_changed=False, training_started=False, quality_accepted=False,
                plan_sha256=io.sha(plan_path),
                limitation='Checks that the existing consumer preserves saved geometry and native pixels. Does not make the seven incorrect elbow fits physically correct or admit full-body Gaussian training.')
            io.save(out/'result.json', result)
            status('complete', poses_completed=len(records))
            print(__import__('json').dumps({k: result[k] for k in
                ('state', 'frames', 'elapsed_seconds', 'peak_allocated_GiB', 'optimizer_updates')}), flush=True)
    except BaseException as error:
        io.save(out/'failure.json', dict(error=repr(error), traceback=traceback.format_exc(), training_started=False))
        status('failed')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    run(parser.parse_args().plan)
