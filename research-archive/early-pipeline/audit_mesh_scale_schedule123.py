"""CPU audit plus reviewable original-repo adaptation; never changes training code."""
import ast
import difflib
import json
import shutil
import numpy as np
from plyfile import PlyData
import audit_gaussianavatars_params123 as a
import run_official_gaussians123 as io


def stats(x):
    return dict(zip(('min', 'median', 'p90', 'p99', 'max'),
                    map(float, np.quantile(x, [0, .5, .9, .99, 1]))))


def main():
    root = a.O/'shared-vertex-offset-adc-1000'
    with np.load(a.lp(root/'fit/fitted.npz'), allow_pickle=False) as f:
        fit = dict(f)
    v = fit['vertices_world'].astype(float); vc = fit['vertices_camera'].astype(float)
    E = fit['w2c'].astype(float); R, t = E[:, :3, :3], E[:, :3, 3]
    centers = -np.einsum('nji,nj->ni', R, t)
    radius = float(1.1*np.linalg.norm(centers-centers.mean(0), axis=1).max())
    tri = v[0][fit['faces']]; e1 = tri[:, 1]-tri[:, 0]; e2 = tri[:, 2]-tri[:, 0]
    edge = np.linalg.norm(e1, axis=1)
    face_scale = (edge+np.linalg.norm(np.cross(e1, e2), axis=1)/edge)/2
    projection_parity = float(np.abs(np.einsum('nij,nkj->nki', R, v)+t[:, None]-vc).max())
    extent = float(np.linalg.norm(np.ptp(v[0], axis=0)))
    q = a.read(root/'export/quality-review.json')
    sources = {str(p.relative_to(a.A/'upstream')): a.sha(p)
               for p in (a.A/'upstream').rglob('*.py')}
    scale = dict(world_scale=float(fit['world_scale']), camera_basis=json.loads(str(fit['camera_length_contract_json'].item())),
        reference_body_camera_bbox_m=np.ptp(vc[50], axis=0).tolist(),
        R_orthogonality_max=float(np.abs(R@R.transpose(0, 2, 1)-np.eye(3)).max()),
        world_to_camera_max_abs_error_m=projection_parity,
        original_radius_rule_on_current_camera_numbers=radius,
        original_radius_is_not_independent_metric_calibration=True,
        local_spatial_lr_scale=1., local_densification_extent=extent,
        triangle_scale_m=stats(face_scale),
        current_initial_xyz_lr_world_coefficient_mm=stats(.005*face_scale*1000),
        shared_offset_lr_model_m=1.e-6,
        shared_offset_actual_m=q['geometry'][1]['offset_m'],
        no_evidence_of_meter_millimeter_mismatch=True,
        original_head_asset_metric_comparison_not_performed=True)
    init = dict(original='one Gaussian per triangle at centroid; local xyz0, local scale1, opacity0.1; inherited create_from_pcd',
        current='200000 area-weighted triangle samples; density-dependent tangential width and0.35 normal multiplier',
        smplx_triangles=len(fit['faces']), points_per_triangle_mean=200000/len(fit['faces']),
        source_local='src/avatar_core/scene/smplx_gaussian_model.py:122',
        source_original='scene/flame_gaussian_model.py:38; scene/gaussian_model.py:172')
    with a.lp(a.O/'shared-vertex-offset-1000/gaussian-run/step-001000/bound-local.ply').open('rb') as stream:
        d = PlyData.read(stream)['vertex'].data
        init['actual_initial_count_per_triangle'] = stats(np.bincount(d['binding_0'].astype(int), minlength=len(fit['faces'])))
    schedule = dict(original_steps=600000, target_steps=18000, time_ratio=.03,
        position_lr_max_steps=[600000, 18000], densify_from_iter=[10000, 300],
        densification_interval=[2000, 60], densify_until_iter=[600000, 18000],
        opacity_reset_interval=[60000, 1800], sh_degree_interval=[1000, 30],
        large_splat_pruning_start=[60000, 1800], screen_size_threshold_px=20,
        evaluation_snapshot_interval=[60000, 1800], hard_review_step=1000,
        actual_trial_budget=1000, opacity_reset_not_exercised_in_first1000=True,
        do_not_multiply_lr_or_loss_weights_by_inverse_time_ratio=True)
    # Prepare literal original-repo default/timing edits; no replacement trainer.
    original_changes = []
    originals = a.A/'upstream'
    path = originals/'arguments/__init__.py'; before = path.read_text(encoding='utf-8'); after = before
    for old, new in [('self.iterations = 600_000', 'self.iterations = 18_000'),
        ('self.position_lr_max_steps = 600_000', 'self.position_lr_max_steps = 18_000'),
        ('self.densification_interval = 2_000', 'self.densification_interval = 60'),
        ('self.opacity_reset_interval = 60_000', 'self.opacity_reset_interval = 1_800'),
        ('self.densify_from_iter = 10_000', 'self.densify_from_iter = 300'),
        ('self.densify_until_iter = 600_000', 'self.densify_until_iter = 18_000')]:
        assert after.count(old) == 1, old
        after = after.replace(old, new)
    original_changes.append(('arguments/__init__.py', before, after))
    before = (originals/'train.py').read_text(encoding='utf-8')
    after = before.replace('# Every 1000 its we increase', '# Every 30 its we increase').replace(
        'if iteration % 1000 == 0:', 'if iteration % 30 == 0:').replace('default=60_000', 'default=1_800')
    assert after != before
    original_changes.append(('train.py', before, after))
    diff = ''
    for name, before, after in original_changes:
        ast.parse(after)
        diff += ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                         fromfile='a/'+name, tofile='b/'+name))
    a.lp(a.A/'original-repo-18k-timing.patch').write_text(diff, encoding='utf-8', newline='\n')
    # Minimal existing SMPL-X bridge: call inherited upstream initialization.
    path = a.P/'src/avatar_core/scene/smplx_gaussian_model.py'
    before = path.read_text(encoding='utf-8')
    needle = '        assert count >= len(self.faces)\n'
    assert before.count(needle) == 1
    block = '''        if count == len(self.faces):
            self.binding = torch.arange(len(self.faces), device='cuda')
            self.binding_counter = torch.ones(len(self.faces), dtype=torch.int32, device='cuda')
            self.create_from_pcd(None, spatial_lr_scale=1.0)
            return dict(initial_points=len(self.faces), covered_triangles=len(self.faces),
                        triangles=len(self.faces), spatial_lr_scale=1.0,
                        position_initialization='upstream one Gaussian per triangle at centroid',
                        scale_initialization='upstream local scale1',
                        color_initialization='upstream create_from_pcd')
'''
    after = before.replace(needle, needle+block)
    ast.parse(after)
    bridge_diff = ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
        fromfile='a/src/avatar_core/scene/smplx_gaussian_model.py', tofile='b/src/avatar_core/scene/smplx_gaussian_model.py'))
    a.lp(a.A/'original-one-per-triangle-bridge.patch').write_text(bridge_diff, encoding='utf-8', newline='\n')
    # Schedule checks compare normalized progress, not rendered quality.
    assert a.events(300, 60, 18000, 1000) == list(range(360, 961, 60))
    assert len(a.events(300, 60, 18000, 18000)) == len(a.events(10000, 2000, 600000, 600000)) == 294
    assert list(range(1800, 18000, 1800)) == [i*.03 for i in range(60000, 600000, 60000)]
    assert [30, 60, 90] == [i*.03 for i in (1000, 2000, 3000)]
    plan = dict(updated=io.now(), state='original_repo_patch_prepared_not_applied',
        original_commit='29b0021308d45c6dce0b522044a4893e189c7a23',
        initial_points=len(fit['faces']), initialization='upstream create_from_pcd, one per face',
        schedule=schedule, shared_static_offset_lr=1.e-6,
        retain_mask_native_pixels_frozen_fit_and_cameras=True,
        no_automatic_scene_scale_change=True, no_custom_mesh_solver=True,
        no_new_training_started=True, risk_approval_required_before_training=True,
        risks=['Fewer initial points may temporarily reduce coverage.',
               'SH3 would be reached at step90 before all94 frames have been seen, and may absorb geometry errors.',
               'Opacity resets can temporarily increase holes; first1000 does not exercise reset1800.',
               'Original head defaults are not independently validated for full-body clothing.'],
        not_single_factor='Original initialization and time-strategy adaptation are a declared combined profile, not unique causal proof.',
        remaining_minimal_integration='Connect this profile to the existing SMPL-X/mask input bridge while reusing original trainer operations. Do not start the old custom config or claim the draft is executable.',
        patches=['original-repo-18k-timing.patch','original-one-per-triangle-bridge.patch'])
    result = dict(updated=io.now(), initialization=init, scale=scale, schedule=schedule,
        sources=sources, source_readonly=True, original_numeric_defaults_match_is_not_full_recipe_equivalence=True,
        current_quality_accepted=False, next_profile=plan)
    a.save(a.A/'initialization-scale-schedule-audit.json', result)
    a.save(a.A/'original-repo-short-profile-plan.json', plan)
    print(json.dumps(dict(triangles=init['smplx_triangles'], radius=radius, local_extent=extent,
                         offset_median_mm=scale['shared_offset_actual_m']['median']*1000,
                         training_source_unchanged=True, schedule_contract_checks=4)))


if __name__ == '__main__': main()
