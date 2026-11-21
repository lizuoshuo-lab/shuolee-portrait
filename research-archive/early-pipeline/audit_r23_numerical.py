"""Independently verify frozen inputs, stable garment lineage, and R23 endpoints."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData
import torch

from review_round21_matched_geometry import _array, _pin, _read, _require, _sha


LAB = Path('/opt/gs-slam-lab')
OUTPUT = Path('/mnt/c/Users/Administrator/Documents/New project/outputs/smpl-delivery-round23/input-audit')
PARAMS = ('_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity')
STEPS = (10000, 18000)
PROBES = {'016': 'probe-03', '006': 'probe-01'}


def ply_parameters(path):
    rows = PlyData.read(path, mmap=False)['vertex'].data
    def fields(names):
        return np.stack([rows[name] for name in names], axis=1).astype(np.float32, copy=False)
    out = dict(_xyz=fields(('x', 'y', 'z')),
               _features_dc=fields(tuple(f'f_dc_{i}' for i in range(3)))[:, None, :],
               _features_rest=np.empty((len(rows), 0, 3), np.float32),
               _scaling=fields(tuple(f'scale_{i}' for i in range(3))),
               _rotation=fields(tuple(f'rot_{i}' for i in range(4))),
               _opacity=fields(('opacity',)))
    if 'binding_0' in rows.dtype.names:
        out['binding'] = rows['binding_0'].astype(np.int64)
    _require(all(np.isfinite(value).all() for value in out.values()), 'Invalid PLY values.')
    return out


def tensor_hash(value):
    return sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def garment_seed(subject):
    run = LAB/f'shuolee-smpl-delivery-round22-{subject}'/'run'
    out = run/'gaussian-run'
    raw, snapshot = out/'raw-step-018000', out/'step-018000'
    receipt = _read(raw/'state.json')
    checkpoint = raw/'state.pt'
    _pin(checkpoint, receipt['state_sha256'])
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)['state']
    files = receipt['files_sha256']
    bound = _read(snapshot/'bound-replay.json')
    bound_hash = bound['bound_ply_sha256']
    _pin(raw/'bound-local.ply', bound_hash)
    _pin(snapshot/'bound-local.ply', bound_hash)
    _pin(raw/'world.ply', files['world.ply'])
    _pin(snapshot/'world.ply', files['world.ply'])
    local, world = [ply_parameters(raw/name) for name in ('bound-local.ply', 'world.ply')]
    _require(all(np.array_equal(local[key], _array(state['parameters'][key]['value'])) for key in PARAMS),
             'R22 PLY parameters do not match the checkpoint rows.')
    _require(np.array_equal(local['binding'], _array(state['model']['binding'])) and
             np.array_equal(local['_opacity'], world['_opacity']), 'R22 local and world row identities differ.')
    ids = _array(state['model']['densification_ids'])
    next_id = state['model']['next_densification_id']['value']
    _require(len(ids) == len(local['binding']) and (np.diff(ids) > 0).all() and next_id > ids[-1], 'Invalid R22 lineage.')
    guide = _read(out/'clothing-surface.json')
    support = out/guide['support_file']
    _pin(support, guide['support_file_sha256'])
    _pin(snapshot/'reference-mesh.npz', guide['active_mesh_sha256'])
    with np.load(snapshot/'reference-mesh.npz', allow_pickle=False) as data:
        faces = data['faces']
    with np.load(support, allow_pickle=False) as data:
        eligible = data['eligible_vertices'][faces].all(axis=1)[local['binding']]
    seed = dict(ids=ids[eligible], owner=local['binding'][eligible], next=next_id,
                xyz=world['_xyz'][eligible], log_sigma=world['_scaling'][eligible],
                opacity=local['_opacity'][eligible, 0], fixed_ids=ids[~eligible],
                fixed={key:value[~eligible] for key,value in local.items()}, faces=faces)
    seed['paths'] = dict(raw_state=checkpoint, bound_ply=snapshot/'bound-local.ply', world_ply=raw/'world.ply',
                         support=support, fit=run/'fit/fitted.npz', training_inputs=run/'data/training-inputs.json',
                         reference_mesh=snapshot/'reference-mesh.npz')
    return seed, run, snapshot


def motion_scope(run, old, endpoint, seed, manifest, state):
    poses = []
    for group in ('posed-replay', 'heldout-replay'):
        receipt = _read(old/group/'result.json')
        mesh = old/group/receipt['mesh_file']
        _pin(mesh, receipt['mesh_sha256'])
        if group == 'posed-replay':
            _pin(old/'bound-replay.json', receipt['bound_replay_sha256'])
        else:
            _require(receipt['bound_ply_sha256'] == state['source_sha256'], 'Old pose replay uses another PLY.')
        with np.load(mesh, allow_pickle=False) as data:
            poses.append((data['frame_ids'].tolist(), data['vertices_world'], data['faces']))
    train_rows = _read(seed['paths']['training_inputs'])['records']
    held_receipt = _read(old/'heldout-replay/result.json')
    train = {row['frame']:row for row in train_rows}
    held = held_receipt['records']
    frame_ids = poses[0][0]+poses[1][0]
    roster = [train[frame] for frame in poses[0][0]]+held
    motion = endpoint/'drive/motion.npz'
    _pin(motion, manifest['files_sha256']['motion.npz'])
    _pin(endpoint/'reference-mesh.npz', _sha(seed['paths']['reference_mesh']))
    _pin(run/'fit/fitted.npz', _sha(seed['paths']['fit']))
    _pin(run/'data/training-inputs.json', _sha(seed['paths']['training_inputs']))
    with np.load(motion, allow_pickle=False) as data:
        _require(data['frame_ids'].tolist() == frame_ids == manifest['frame_ids']
                 and data['reference_index'].item() == manifest['reference_index']
                 and all(np.array_equal(data['faces'], row[2]) for row in poses)
                 and np.array_equal(data['vertices_world'], np.concatenate([row[1] for row in poses]))
                 and np.array_equal(data['clothing_support_face_ids'], state['support_face_ids'].numpy())
                 and np.array_equal(data['clothing_stable_ids'], state['stable_ids'].numpy())
                 and data['next_clothing_stable_id'].item() == state['next_stable_id'], 'Saved drive motion differs.')
        for key in ('K', 'w2c', 'wh'):
            _require(np.array_equal(data[f'camera_{key}'], [row[key] for row in roster]), 'Native camera rows changed.')
    return frame_ids, roster, held


def movement(before, after):
    mm = np.linalg.norm((after.astype(np.float64)-before)*1000., axis=1)
    return dict(rows=len(mm), p50_mm=float(np.median(mm)), p95_mm=float(np.quantile(mm, .95)),
                max_mm=float(mm.max()), above_1mm_fraction=float(np.mean(mm>1.)),
                above_5mm_fraction=float(np.mean(mm>5.)), above_10mm_fraction=float(np.mean(mm>10.)))


def child_opacity(value):
    logit = torch.tensor(value, dtype=torch.float32)
    alpha = -torch.expm1(.5*torch.log1p(-logit.sigmoid()))
    return float(torch.logit(alpha, eps=1e-7))


def lineage(seed, state, config):
    # Stock capture tuple: active SH, six parameters, binding, counters, statistics, Adam, LR factor.
    active, *params, binding, binding_counter, _, _, _, adam, _ = state['garment']
    _require(active == 0 and binding is None and binding_counter is None
             and all(torch.isfinite(value).all() for value in params), 'Garment tensors are invalid.')
    parameters = {key: value.detach().numpy() for key,value in zip(PARAMS, params, strict=True)}
    ids, owner = [state[key].numpy() for key in ('stable_ids', 'support_face_ids')]
    live = {int(id_):(int(face), float(logit), i, False)
            for i,(id_,face,logit) in enumerate(zip(seed['ids'], seed['owner'], seed['opacity'], strict=True))}
    next_id = seed['next']
    events = state['split_events']
    _require([event['step'] for event in events] == [i for i in config['splits'] if i <= state['step']],
             'The actual event schedule differs.')
    for event in events:
        parents = event.get('parent_ids', [])
        _require(len(parents) == len(set(parents)) == event['parents'] <= config['max_parents_per_split']
                 and event['children'] == 2*len(parents) and all(id_ in live for id_ in parents), 'Split lineage differs.')
        if not parents:
            continue
        _require(event['first_child_id'] == next_id, 'A split reused a stable ID.')
        records = [live.pop(id_) for id_ in parents]
        for i, (face, logit, seed_index, _) in enumerate(records*2):
            live[next_id+i] = (face, child_opacity(logit), seed_index, True)
        next_id += event['children']
    _require(ids.shape == owner.shape == (len(parameters['_xyz']),) and (np.diff(ids) > 0).all()
             and ids.tolist() == sorted(live) and state['next_stable_id'] == next_id, 'Endpoint ID set differs.')
    want_owner, want_logit, seed_index, children = map(np.asarray, zip(*(live[int(i)] for i in ids), strict=True))
    _require(np.array_equal(owner, want_owner), 'A surviving garment changed its inherited triangle.')
    actual_logit = parameters['_opacity'][:, 0]
    _require(np.array_equal(actual_logit[~children], want_logit[~children].astype(np.float32)),
             'An original garment opacity logit changed.')
    child_alpha_error = float(np.max(np.abs(actual_logit[children]-want_logit[children]), initial=0.))
    _require(child_alpha_error <= 2e-5, 'A child opacity differs from recursive inherited peak rule.')
    caps = np.full(len(seed['faces']), -np.inf, np.float32)
    np.maximum.at(caps, seed['owner'], seed['log_sigma'].max(axis=1))
    over = parameters['_scaling'].max(axis=1)-caps[owner]
    _require(float(over.max()) <= 2e-6, 'A garment sigma exceeds its inherited support cap.')
    opacity = [group for group in adam['param_groups'] if group['name'] == 'opacity']
    _require(len(opacity) == 1 and opacity[0]['lr'] == 0 and all(param not in adam['state'] for param in opacity[0]['params']),
             'Frozen garment opacity accumulated optimizer state.')
    stats = dict(initial_garment_rows=len(seed['ids']), original_rows_remaining=int((~children).sum()),
                 live_inherited_children=int(children.sum()), total_committed_splits=sum(item['parents'] for item in events),
                 split_events=[{key:event[key] for key in ('step','eligible','tangent_eligible','parents','children')} for event in events],
                 original_opacity_logits_bit_exact=True, child_logit_max_abs_error=child_alpha_error,
                 max_excess_over_seed_owner_cap_log_sigma=float(over.max()),
                 max_excess_over_seed_owner_cap_sigma_mm=float(np.max(np.exp(parameters['_scaling'].max(axis=1))-np.exp(caps[owner]))*1000.),
                 original_same_id_canonical_center=movement(seed['xyz'][seed_index[~children]], parameters['_xyz'][~children]))
    if children.any():
        stats['child_to_original_ancestor_center_includes_split_offset'] = movement(seed['xyz'][seed_index[children]], parameters['_xyz'][children])
    return parameters, stats


def alpha_image(path, wh, box=None):
    with Image.open(path) as image:
        _require(image.size == tuple(wh), 'Unexpected native alpha resolution.')
        rgb = np.array((image if box is None else image.crop(box)).convert('RGB'))
    _require(np.array_equal(rgb[:, :, 0], rgb[:, :, 1]) and np.array_equal(rgb[:, :, 0], rgb[:, :, 2]),
             'The alpha image is not achromatic.')
    return rgb[:, :, 0]


def native_alpha(old, endpoint, rows, held):
    old_files = old/'heldout-replay'
    new_files = endpoint/'drive-replay'
    replay = _read(new_files/'result.json')['records']
    by_name = {row['frame']:row for row in rows}
    by_replay = {row['frame']:row for row in replay}
    source_rows = {row['frame']:row for row in _read(old.parent.parent/'data/inputs.json')['records']}
    out = []
    for previous in held:
        frame = previous['frame']
        row, current, source = by_name[frame], by_replay[frame], source_rows[frame]
        _require(current['role'] == 'pose_conditioned_holdout' and current['pose_index'] == previous['pose_index']
                 and all(current[k] == row[k] == source[k] for k in ('source_sha256', 'mask_sha256')),
                 'R23 holdout replay does not bind the same actual source and mask.')
        box, wh = current['box_xyxy'], previous['wh']
        x0,y0,x1,y1 = box
        _require(0 <= x0 < x1 <= wh[0] and 0 <= y0 < y1 <= wh[1], 'Crop extends beyond native frame.')
        before_file = old_files/previous['files']['alpha']['file']
        _pin(before_file, previous['files']['alpha']['sha256'])
        after_file = new_files/current['files']['alpha']
        before = alpha_image(before_file, wh, box)
        after = alpha_image(after_file, (x1-x0,y1-y0))
        mask_file = Path(source['mask'])
        _pin(mask_file, current['mask_sha256'])
        with Image.open(mask_file) as img:
            _require(img.size == tuple(wh), 'Native target mask has different resolution.')
            full_mask = np.array(img.convert('L'))
        mask = full_mask[y0:y1,x0:x1]
        foreground = mask.astype(np.float32)/255. > .99
        _require(int(foreground.sum()) == int((full_mask.astype(np.float32)/255. > .99).sum()) > 0,
                 'Crop discards target foreground.')
        def measured(pixels):
            values = pixels.astype(np.float64)/255.
            return dict(foreground_mean=float(values[foreground].mean()),
                        foreground_below_point9_fraction=float((values[foreground]<.9).mean()),
                        local_zero_mask_exterior_mean=float(values[mask == 0].mean()),
                        crop_soft_alpha_l1_per_foreground=float(np.abs(values-mask/255.).sum()/foreground.sum()))
        out.append(dict(frame=frame, box_xyxy=box, pixels=int(foreground.sum()),
                        old_alpha_sha256=_sha(before_file), new_alpha_sha256=_sha(after_file),
                        r22=measured(before), r23=measured(after)))
    return dict(scope='Exact fitted heldout pose, native K/w2c, source/mask and same unresized crop. PNG alpha is 8-bit; exterior excludes outside-crop pixels.', records=out)


def read_state(path):
    # Optimizer learning rates include NumPy scalars in these trusted snapshots.
    with torch.serialization.safe_globals([np.core.multiarray.scalar, np.dtype, type(np.dtype('float64')), type(np.dtype('int64'))]):
        return torch.load(path, map_location='cpu', weights_only=True)


def audit(subject, steps, probe):
    seed, _, old = garment_seed(subject)
    run = LAB/f'shuolee-smpl-delivery-round23-{subject}'/(PROBES[subject] if probe else 'run')
    config = _read(run/'gaussian-run/config.json')
    states, output = {}, []
    for step in steps:
        endpoint = run/'gaussian-run'/f'raw-step-{step:06d}'
        file = endpoint/'main-state.pt'
        state = read_state(file)
        _require(state['step'] == len(state['draws']) == step and state['options'] == config['optimization']
                 and state['source_sha256'] == _sha(seed['paths']['bound_ply']), 'R23 state source or schedule changed.')
        manifest = _read(endpoint/'drive/drive.json')
        frames, rows, held = motion_scope(run, old, endpoint, seed, manifest, state)
        train = len(frames)-len(held)
        _require(all(isinstance(i, int) and 0 <= i < train for i in state['draws']), 'An optimizer draw uses a holdout or invalid frame.')
        paths = {name:endpoint/'drive'/name for name in ('fixed-bound.ply', 'clothing-canonical.ply')}
        for name,path in paths.items():
            _pin(path, manifest['files_sha256'][name])
        fixed = ply_parameters(paths['fixed-bound.ply'])
        _require(all(np.array_equal(value, fixed[key]) for key,value in seed['fixed'].items())
                 and state['fixed_identity'] == {key:tensor_hash(value) for key,value in fixed.items()},
                 'A protected non-garment parameter or owner changed from its R22 seed.')
        parameters, stats = lineage(seed, state, config)
        cloth = ply_parameters(paths['clothing-canonical.ply'])
        _require(all(np.array_equal(value, cloth[key]) for key,value in parameters.items()),
                 'Canonical bundle PLY differs from the saved optimizer rows.')
        states[step] = (state, parameters)
        output.append(dict(step=step, checkpoint_sha256=_sha(file), fixed_rows=len(seed['fixed_ids']),
                           exact_frozen_noncloth_parameters=True, exact_frozen_pose_cameras=len(frames),
                           train_draw_views=len(set(state['draws'])), **stats,
                           native_alpha=native_alpha(old, endpoint, rows, held)))
    paired = None
    if len(states) == 2:
        first, last = steps
        a,b = [states[i][0] for i in steps]
        _require(b['draws'][:first] == a['draws'] and a['split_events'] == b['split_events'], '10k is not the exact draw/topology prefix.')
        ids_a,ids_b = [item['stable_ids'].numpy() for item in (a,b)]
        ids,ia,ib = np.intersect1d(ids_a,ids_b,assume_unique=True,return_indices=True)
        _require(np.array_equal(a['support_face_ids'].numpy()[ia], b['support_face_ids'].numpy()[ib]), 'Same-ID motion owner changed.')
        delta = [states[i][1]['_xyz'][index] for i,index in zip(steps,(ia,ib),strict=True)]
        paired = dict(exact_draw_prefix=True, exact_split_history=True, shared_ids=len(ids),
                      canonical_center=movement(*delta))
    report = dict(schema='shuolee-r23-independent-numerical/1', subject=subject,
                  kind='single-update native probe' if probe else 'formal', run=str(run),
                  baseline={key:dict(file=str(file),sha256=_sha(file)) for key,file in seed['paths'].items()},
                  endpoints=output, same_run=paired, physical_clothing_surface_ground_truth=False)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    tag = 'probe' if probe else ('paired' if paired else str(steps[0]))
    target = OUTPUT/f'{subject}-numerical-{tag}.json'
    with target.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(f'{target} sha256={_sha(target)}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject', choices=('016','006'))
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--step', type=int, choices=STEPS)
    args = parser.parse_args()
    torch.set_num_threads(4)
    audit(args.subject, (1,) if args.probe else ((args.step,) if args.step else STEPS), args.probe)
