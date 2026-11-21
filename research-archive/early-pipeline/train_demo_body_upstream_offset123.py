"""Enable only the upstream commented shared vertex-offset group at1e-6.

Reuse the immutable original-defaults launcher and public training function.
Start fresh from the same VHAP mesh, not the frozen-offset Gaussian checkpoint.
"""
import argparse
import difflib
import inspect
from pathlib import Path
import traceback
import train_demo_body_upstream123 as base
from run_official_gaussians123 import read, save, sha, now

OUT = base.DEMO/'body-gaussian-upstream-offset'


def revised_train():
    original = inspect.getsource(base.train)
    changed = original
    def replace(old, new):
        nonlocal changed
        assert changed.count(old) == 1, old
        changed = changed.replace(old, new, 1)
    replace('        select_mesh_by_timestep = SMPLXGaussianModel.select_mesh_by_timestep', '''        enable_shared_static_offset = SMPLXGaussianModel.enable_shared_static_offset
        shared_offset_vertices = SMPLXGaussianModel.shared_offset_vertices
        def training_setup(self, training_args):
            super().training_setup(training_args)
            self.optimizer.add_param_group({'params': [self.static_offset], 'lr': 1.e-6, 'name': 'static_offset'})
        select_mesh_by_timestep = SMPLXGaussianModel.select_mesh_by_timestep''')
    replace('    model = Body(3)', '''    model = Body(3)
    offset_receipt = model.enable_shared_static_offset(OUT/'fit/fitted.npz', ASSETS/'smplx/SMPLX_NEUTRAL.npz')
    assert torch.equal(model.static_offset.detach().cpu(), torch.as_tensor(data['static_offset'][0]))
    initial_offset = model.static_offset.detach().clone()
    model.select_mesh_by_timestep(0)
    save(run/'offset-initialization.json', offset_receipt)''')
    replace("torch.save(dict(step=step, capture=model.capture()), folder/'checkpoint.pt')",
            "torch.save(dict(step=step, capture=model.capture(), static_offset=model.static_offset.detach().clone()), folder/'checkpoint.pt')")
    replace("frame_ids=data['frame_ids'], static_offset=data['static_offset'])",
            "frame_ids=data['frame_ids'], static_offset=model.static_offset.detach().cpu().numpy())")
    replace('            for i, row in enumerate(rows):\n                model.select_mesh_by_timestep(layout',
            '            mesh_vertices = []\n            for i, row in enumerate(rows):\n                mesh_vertices.append(model.shared_offset_vertices(layout[\'pose_indices\'][i]).cpu().numpy())\n                model.select_mesh_by_timestep(layout')
    replace('            model.select_mesh_by_timestep(old)', '''            np.savez_compressed(folder/'current-meshes.npz', vertices_camera=np.asarray(mesh_vertices), faces=data['faces'],
                frame_ids=np.asarray([r['frame'] for r in rows]))
            offset_delta = (model.static_offset-initial_offset).norm(dim=-1)
            save(folder/'offset-update.json', dict(initial_offset_sha256=sha(OUT/'fit/fitted.npz'),
                lr=1.e-6, changed_vertices=int((offset_delta>0).sum()), median_m=float(offset_delta.median()),
                p90_m=float(torch.quantile(offset_delta,.9)), max_m=float(offset_delta.max()),
                shared_across_frames=True, pose_shape_expression_frozen=True))
            model.select_mesh_by_timestep(old)''')
    replace("vertex_offset='Frozen final VHAP shared offset'", "vertex_offset='Shared canonical vertexoffset optimized at1e-6 from final VHAP values'")
    replace("assert not model.optimizer.state and all(g['name'] != 'static_offset' for g in model.optimizer.param_groups)",
            "assert not model.optimizer.state and len([g for g in model.optimizer.param_groups if g['name'] == 'static_offset' and g['lr'] == 1.e-6]) == 1\n            assert float(model.static_offset.grad.norm()) > 0")
    replace('static_offset_optimizer_present=False', 'static_offset_optimizer_present=True, static_offset_gradient_norm=float(model.static_offset.grad.norm())')
    replace('frozen_static_offset=True', 'frozen_static_offset=False, shared_static_offset_lr=1.e-6')
    namespace = dict(base.train.__globals__)
    exec(compile(changed, '<upstream-defaults-enable-shared-offset-only>', 'exec'), namespace)
    return namespace['train'], original, changed


def prepare():
    base.OUT = OUT
    base.prepare()
    plan = read(OUT/'plan.json')
    baseline = read(base.DEMO/'body-gaussian-upstream-defaults/plan.json')
    assert plan['options'] == baseline['options']
    assert sha(OUT/'fit/fitted.npz') == sha(base.DEMO/'body-gaussian-upstream-defaults/fit/fitted.npz')
    plan.update(user_authorization=['vertexoffset开启，让他拟合mesh，然后其他参数不变'],
        supersedes='Only the frozen shared vertexoffset from the preceding default run',
        shared_static_offset_enabled=True, shared_static_offset_lr=1.e-6,
        static_vertex_offset='Load VHAP absolute offset once before SMPL-X LBS and optimize a single shared10475x3 tensor at the upstream commented rate1e-6.',
        mesh_pose='Body pose, beta, expression and cameras frozen; shared vertexoffset deforms all meshes through official SMPL-X LBS.',
        baseline_run=str(base.DEMO/'body-gaussian-upstream-defaults'),
        only_parameter_change='Activate shared static_offset optimizer group at1e-6; all OptimizationParams and initial Gaussian seed/count unchanged.',
        old_gaussian_checkpoint_loaded=False, quality_accepted=False)
    plan['source_hashes'][str(Path(__file__).resolve())] = sha(__file__)
    save(OUT/'plan.json', plan)
    _, original, changed = revised_train()
    (OUT/'default-launcher-function.py').write_text(original)
    (OUT/'offset-launcher-function.py').write_text(changed)
    (OUT/'enable-offset-only.patch').write_text(''.join(difflib.unified_diff(
        original.splitlines(True), changed.splitlines(True), fromfile='default-adapter',tofile='shared-offset-adapter')))
    print('SHARED_OFFSET_PREPARED_OTHER_OPTIONS_EQUAL', flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['prepare','train']); args=parser.parse_args()
    try:
        base.OUT=OUT
        if args.mode=='prepare': prepare()
        else: revised_train()[0]()
    except BaseException as error:
        if OUT.exists():
            save(OUT/(args.mode+'-failure.json'),dict(updated=now(),error=repr(error),traceback=traceback.format_exc(),auto_retry=False))
            save(OUT/'progress.json',dict(updated=now(),state='failed_preserved',pid=None,error=repr(error),quality_accepted=False))
        raise
