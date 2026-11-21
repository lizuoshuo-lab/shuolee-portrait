"""Add one physically tracked hand-surface correspondence to the 168 control."""
from datetime import datetime, timezone
from importlib.util import find_spec
import inspect
import os
from pathlib import Path
import sys
import time
import types

import numpy as np

from extract_video21_shoe_depth162 import replace
import probe_video21_elbow_visibility168 as previous
from track_video21_wrist169 import OUT as MATERIAL
from update_leafwoman129_state import atomic, read, sha


BASE = previous.baseline
OUT = BASE.BASE.OUT / 'right-hand-surface170'


def fit_shard(folder, shard):
    code = inspect.getsource(BASE.fit_shard)
    code = replace(code, '        def anchored_loader(root,anchor,receipt):', '''        target_hand = plan['hand_constraints'][frame]
        hand_and_foot = (target_hand['used_anchor'], target_hand['near_shoe'])
        attachment_ids = torch.as_tensor([a['vertices'] for a in hand_and_foot], dtype=torch.long)
        attachment_bary = torch.as_tensor([a['barycentric'] for a in hand_and_foot], dtype=bary.dtype)
        observed = torch.as_tensor(target_hand['used_anchor']['source_uv'], dtype=bary.dtype)
        depth_target = target_hand['used_anchor']['target_log_depth_over_near_shoe']
        depth_scale = target_hand['used_anchor']['depth_px_per_log_ratio']

        def anchored_loader(root,anchor,receipt):''')
    code = replace(code, "if loss_type!='smplify' or not use_depth:", "if loss_type!='smplify':")
    code = replace(code, '''                    z=(obj.vertices[:,ids,2]*bary).sum(2)+cam.translation[:,2,None]
                    logz=z.clamp_min(.05).log()
                    mismatch=pixel_scale*(logz[:,0]-logz[:,1]-target)
                    calls+=1
                    return original+loss.robustifier(mismatch).sum()*loss.data_weight.square()''', '''                    points = (obj.vertices[:, attachment_ids, :] * attachment_bary[None, :, :, None]).sum(2)
                    logz = (points[..., 2] + cam.translation[:, 2, None]).clamp_min(.05).log()
                    depth_error = depth_scale * (logz[:, 0] - logz[:, 1] - depth_target)
                    image_error = cam(points[:, :1]) - observed
                    cost = loss.robustifier(depth_error).sum() + loss.robustifier(image_error).sum()
                    calls += 1
                    return original + cost * loss.data_weight.square()''')
    code = replace(code, '        arrays,receipt=solve(saved_seed=seed,replay_only=False,**inputs)',
                   '''        assert not use_depth and frame in plan['occluded_fit_frames']
        observations = inputs['keypoints'].copy()
        assert observations[0, 3, 2] > 0
        observations[0, 3, 2] = 0
        arrays,receipt=solve(saved_seed=seed,replay_only=False,**dict(inputs,keypoints=observations))''')
    code = replace(code, 'assert (calls>0)==use_depth and receipt', 'assert calls>0 and receipt')
    code = replace(code, 'depth_term_enabled=use_depth,depth_calls=calls,seed_replay_max_px=difference,',
                   'depth_term_enabled=False,depth_calls=0,hand_surface_calls=calls,\n'
                   '            excluded_coco25_index=3,effective_elbow_confidence=0.,seed_replay_max_px=difference,')
    scope = dict(BASE.__dict__, __file__=__file__)
    exec(compile(code, '<matched-visible-hand-surface>', 'exec'), scope)
    scope['fit_shard'](folder, shard)


def prepare_constraints(plan, preflight):
    assert preflight['holdouts'] == plan['retained_holdouts']
    K = np.asarray(plan['K_integer_centers'])
    assert np.array_equal(preflight['full_K'], K)
    result = {}
    for frame in plan['frames']:
        row = preflight['frames'][frame]
        near = row['right_foot']
        eligible = {}
        for a in row['observations']:
            hit = a['bindings']['seed_149']
            if hit['bound'] and hit['pixel_distance'] == 0. and hit['arm_surface'] == 'hand':
                scale = min(K[0, 0], K[1, 1]) / np.sqrt(
                    1 / np.sin(np.deg2rad(a['parallax_deg']))**2 + 1 / np.sin(near['angle_rad'])**2)
                eligible[a['material']] = dict(a, vertices=hit['vertices'], barycentric=hit['barycentric'],
                    depth_px_per_log_ratio=float(scale))
        assert {'hand_dorsum_0', 'finger_skin_1'} <= eligible.keys()
        result[frame] = dict(used_anchor=eligible['hand_dorsum_0'], near_shoe=near,
            excluded_anchor=eligible['finger_skin_1'])
    return result


def main():
    assert os.name == 'nt' and not OUT.exists()
    if find_spec('omegaconf') is None:
        raise RuntimeError('Run in the existing VPoser environment.')
    prior = previous.OUT
    p, old = read(prior / 'plan.json'), read(prior / 'result.json')
    assert old['completed'] and old['plan_sha256'] == sha(prior / 'plan.json')
    assert old['fit_result_sha256'] == sha(prior / 'control/result.json') and old['frames'] == p['frames']
    assert all(sha(path) == digest for path, digest in p['source_hashes'].items())
    material, review = MATERIAL / 'mesh-binding-preflight.json', MATERIAL / 'visual-decision.json'
    checked = read(material)
    assert checked['source_receipt_sha256'] == sha(review) and read(review)['actual_visual_review']
    assert checked['canonical_masked_fit_sha256'] == old['fit_result_sha256']
    assert sha(BASE.CAMERA) == checked['camera_sha256'] and sha(BASE.BASE.original.ASSET) == checked['official_skinning_sha256']
    assert not set(p['frames']) & set(p['retained_holdouts'])
    constraints = prepare_constraints(p, checked)
    inputs = [Path(__file__), Path(inspect.getsourcefile(replace)), Path(BASE.__file__),
              prior / 'plan.json', prior / 'result.json', prior / 'control/result.json', material, review]
    registered = dict(p, updated=datetime.now(timezone.utc).isoformat(), runner_sha256=sha(__file__),
        source_hashes=p['source_hashes'] | {str(path):sha(path) for path in inputs},
        branches=['hand_surface'], hand_constraints=constraints,
        hypothesis='The visible hand position and scale-free depth relative to a verified shoe anchor constrain the ambiguous hidden-elbow fit.',
        single_change='One bound hand correspondence contributes its native XY and relative-depth residual. Reuse the original robustifier and stage data weight.',
        fixed='Same 168 masked-elbow fit seed and observations, original solver, full K, shared betas and original stage weights. Both detector wrist aliases remain as in 168.',
        protection='Unfitted finger-skin anchor in the same windows, visible wrist and shoulder, knees, face, feet and original native/side review. All globally reserved source frames excluded.',
        criteria='Look for improvement in the unfitted finger-skin projection and relative depth, protect visible wrist, face and feet, and require the existing pose gate before any Gaussian admission.',
        association_limit='Skin or shoe rays bind surface proxies to exact visible triangles on the frozen seed. Not a verified chair contact plane, skeletal landmark or metric scale.',
        gaussian_started=False, quality_accepted=False)
    atomic(OUT / 'plan.json', registered)
    atomic(OUT / 'launch.json',dict(pid=os.getpid(), command=[sys.executable, *sys.argv], frames=p['frames']))
    scope = dict(BASE.BASE.__dict__, fit_shard=fit_shard)
    stage = types.FunctionType(BASE.BASE.stage.__code__, scope)
    indices = [p['source_frames'].index(frame) for frame in p['frames']]
    started = time.monotonic()
    try:
        rows = stage(OUT.name + '/control', registered, indices, p['shared_shape'])
        assert all(r['hand_surface_calls']>0 and r['effective_elbow_confidence']==0.
                   and not r['depth_term_enabled'] for r in rows)
        assert all(sha(path) == digest for path, digest in registered['source_hashes'].items())
        atomic(OUT / 'result.json', dict(completed=True, frames=p['frames'], plan_sha256=sha(OUT / 'plan.json'),
            fit_result_sha256=sha(OUT / 'control/result.json'), seconds=time.monotonic()-started,
            gaussian_started=False))
    except BaseException as exc:
        atomic(OUT / 'failure.json', dict(error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
