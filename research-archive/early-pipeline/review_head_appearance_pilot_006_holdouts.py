"""Apply the existing 006 protected gates to the head-only appearance pilot."""

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_face_photometric_006_holdouts as held
import run_head_appearance_pilot_006 as pilot


read, pin, require, sha = held.read, held.pin, held.require, held.sha
PLAN = held.trial.ACCEPTANCE.with_name('smpl-head-appearance-pilot-preregistration.json')
PLAN_SHA = 'f1026da9d703692c2584ac2a8c10933bc0bd38ece06c1088cc511fc145bfefe0'


def preflight():
    pin(PLAN, PLAN_SHA)
    registered = read(PLAN)
    receipt, views = held.baseline()
    require(registered['state'] == 'registered_before_pilot_optimization'
            and registered['subject'] == '006' and registered['baseline_commit'] == held.trial.BASE_COMMIT
            and registered['locked_metrics_plan_sha256'] == receipt['acceptance_plan_sha256']
            and registered['gates'] == read(held.trial.ACCEPTANCE)['gates']
            and registered['protected_holdouts'] == list(held.FRAMES)
            and registered['optimizer']['iterations'] == pilot.STEPS == 600
            and registered['geometry']['verify_bitwise'] is True, 'Pilot changed the protected metric or geometry contract.')
    require(registered['baseline_terminal']['sha256'] == pilot.STOCK_RESULT == receipt['baseline_terminal_sha256'],
            'Pilot starts from a different baseline terminal.')
    for name in ('baseline_terminal', 'fit', 'rows', 'selection', 'head_ceiling', 'direct_gradient',
                 'heldout_owner_attribution', 'native_and_dolly_cameras'):
        item = registered[name]
        pin(Path(item['path']), item['sha256'])

    held.trial.base.common.git_state(held.trial.BASE_COMMIT)
    sys.path.insert(0, str(pilot.PROJECT/'src'))
    from shuolee_portrait import runtime as rt
    from shuolee_portrait.bound_replay import _inputs, inspect_bound_ply
    from shuolee_portrait.training import camera

    require(rt.PROJECT == pilot.PROJECT, 'Protected replay must use stock 79b906f.')
    pose_path = held.spatial.cold.prior.POSE_ARCHIVE/'evaluation-fitted.npz'
    pin(pose_path, receipt['evaluation_fit_sha256'])
    with np.load(pose_path, allow_pickle=False) as archive:
        poses = dict(archive)
    all_holdouts = list(held.spatial.cold.prior.HELD)
    with np.load(pilot.STOCK/'fit/fitted.npz', allow_pickle=False) as fit:
        require(poses['frame_ids'].tolist() == all_holdouts
                and np.array_equal(poses['shared_betas'], fit['shared_betas'])
                and np.array_equal(poses['faces'], fit['faces'])
                and set(all_holdouts).isdisjoint(fit['frame_ids'].tolist()),
                'Pilot fit and the frozen protected poses differ.')
        trained_frames = len(fit['frame_ids'])
    for view in views:
        saved = view.spec['camera']
        slot = all_holdouts.index(view.spec['frame'])
        view.pose_index, view.camera_index = saved['pose_index'], slot
        view.row = dict(frame=view.spec['frame'], image=view.spec['image'], wh=saved['wh'],
                        K=saved['K'], w2c=saved['w2c'])
        require(trained_frames+slot == view.pose_index
                and poses['source_frames'][slot] == view.spec['source_frame_index']
                and poses['image_sha256'][slot] == view.spec['source_sha256']
                and np.array_equal(poses['w2c'][slot], saved['w2c'])
                and camera(view.row, slot, pose_index=view.pose_index)[1] == saved, 'Protected camera or pose changed.')
    receipt.update(pilot_acceptance_plan_sha256=PLAN_SHA, stock_renderer_commit=held.trial.BASE_COMMIT,
                   baseline_render_calls=0, body_fit_updates=0)

    result_path = pilot.OUT/'result.json'
    if not result_path.is_file():
        return receipt, None
    result = read(result_path)
    internal_path = pilot.OUT/'preregistration.json'
    pin(internal_path, result['preregistration_sha256'])
    internal = read(internal_path)
    require(result['state'] == 'diagnostic_completed' and result['head_optimizer_updates'] == pilot.STEPS
            and result['fit_updates'] == internal['fit_updates'] == 0 and result['reused_terminal'] is True
            and result['start_terminal_sha256'] == internal['source_terminal_sha256'] == pilot.STOCK_RESULT
            and result['exact_frozen_and_geometry_ply_bytes'] is True
            and internal['appearance_steps'] == pilot.STEPS and internal['no_densification_pruning_or_resets'] is True
            and internal['code_sha256'] == sha(Path(pilot.__file__))
            and result['snapshot'] == pilot.SNAPSHOT, 'Wrong completed 600-step appearance pilot.')
    for name in pilot.FROZEN_FILES:
        digest = internal['input_sha256'][str(pilot.STOCK/name)]
        pin(pilot.STOCK/name, digest)
        pin(pilot.OUT/name, digest)
    snapshot = pilot.OUT/pilot.SNAPSHOT
    bound_path = snapshot/'bound-replay.json'
    pin(bound_path, result['bound_manifest_sha256'])
    bound = read(bound_path)
    bound_ply = snapshot/'bound-local.ply'
    require(result['bound_ply'] == str(bound_ply.relative_to(pilot.OUT))
            and result['bound_ply_sha256'] == bound['bound_ply_sha256'], 'Pilot bound model descriptor changed.')
    pin(bound_ply, result['bound_ply_sha256'])
    _, mesh, facts = _inputs(pilot.OUT, snapshot, pilot.ASSET, bound)
    stock = pilot.STOCK/'gaussian-run/step-010000'
    pin(snapshot/'reference-mesh.npz', internal['input_sha256'][str(stock/'reference-mesh.npz')])
    stock_bound = stock/'bound-local.ply'
    pin(stock_bound, internal['input_sha256'][str(stock_bound)])
    binding, _ = inspect_bound_ply(stock_bound, len(poses['faces']))
    editable = pilot.head_mask(poses['faces'], binding)
    changed = pilot.verify_attributes(stock_bound, bound_ply, editable)
    require(len(binding) == bound['points'] and int(editable.sum()) == internal['editable_points']
            and changed == result['changed_head_rows_by_ply']['bound-local'] > 0
            and set(all_holdouts).isdisjoint(row['frame'] for row in facts['frames']),
            'Pilot changes escaped the registered strict head subset.')
    receipt.update(pilot_result_sha256=sha(result_path), pilot_bound_sha256=result['bound_manifest_sha256'],
                   changed_strict_head_points=changed)
    return receipt, SimpleNamespace(rt=rt, run=pilot.OUT, snapshot=snapshot, asset=pilot.ASSET,
                                   poses=poses, views=views, mesh=mesh)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'render'))
    args = parser.parse_args()
    if args.action == 'check':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    receipt, state = preflight()
    if args.action == 'check':
        print(json.dumps(dict(receipt, state='pilot_protected_inputs_verified', terminal_ready=state is not None, gpu_calls=0)))
        return
    require(state is not None, 'Wait for the pilot terminal.')
    output = pilot.OUT/'protected-holdouts'
    require(not output.exists(), 'Preserve any existing protected result.')
    records, arrays = held.render(state, receipt['limits'])
    output.mkdir(exist_ok=False)
    alpha_path = output/'candidate-native-alpha.npz'
    np.savez_compressed(alpha_path, **arrays)
    target = output/'result.json'
    held.spatial.save(target, dict(receipt, state='pilot_protected_gate_complete', records=records,
        candidate_rendered_frames=len(records), candidate_alpha_sha256=sha(alpha_path),
        protected_gate_passed=all(row['passed'] for row in records), reviewer_sha256=sha(Path(__file__))))
    print(json.dumps(dict(receipt=str(target), sha256=sha(target))))


if __name__ == '__main__':
    main()
