"""Independent CPU audit of native R27 facial geometry, protection, and topology."""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from audit_r23_numerical import LAB, PARAMS, ply_parameters, _pin, _read, _require, _sha
import json


REPO = Path('/mnt/d/Projects/Shuolee-smpl-face-surface-r27')
WORKSPACE = Path('/mnt/c/Users/Administrator/Documents/New project')
OUT = WORKSPACE/'outputs/smpl-delivery-round27/input-audit'
COMMIT = '7a329e69cd879b74d9575a67a5188772bba25bee'
CAP_TOLERANCE_M = 1e-7


def arr(tensor):
    return tensor.detach().cpu().numpy()


def npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def save(path, document):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temporary.replace(path)


def select(ids, requested):
    order = ids.argsort()
    positions = np.searchsorted(ids[order], requested).clip(0, len(ids)-1)
    rows = order[positions]
    _require(np.array_equal(ids[rows], requested), 'Requested stable IDs are absent.')
    return rows


def same(a, b):
    return np.array_equal(a, b)


def quantiles(values):
    a = np.asarray(values)
    _require(np.isfinite(a).all(), 'Non-finite geometry.')
    if not a.size:
        return None
    return dict(zip(('p05','p50','p95','max'), (*np.quantile(a, [.05,.5,.95]).tolist(),float(a.max())), strict=True))


def local_metrics(raw, rows):
    xyz, logs, q = (raw[key][rows].astype(np.float64) for key in ('_xyz','_scaling','_rotation'))
    mat = Rotation.from_quat(q[:, [1,2,3,0]]).as_matrix()
    widths = np.exp(logs)
    covariance = (mat*widths[:, None, :]**2) @ mat.transpose(0,2,1)
    tangent = covariance[:, (0,2), :][:, :, (0,2)]
    return dict(center=xyz[:,1], normal_sigma=np.sqrt(covariance[:,1,1]),
                tangent_major=np.sqrt(np.linalg.eigvalsh(tangent)[:,1]),
                angle=np.degrees(np.arccos(np.abs(mat[np.arange(len(rows)),1,logs.argmin(1)]).clip(0,1))),
                principal=np.sort(widths,axis=1))


def poses_and_cameras(base, snap, cfg, receipt):
    motion = npz(snap/'drive/motion.npz')
    meta = _read(snap/'drive/drive.json')
    for name, digest in meta['files'].items():
        _pin(snap/'drive'/name, digest)
    old_snap = base/'gaussian-run/step-018000'
    native, metrics = [], []
    for dir_ in (old_snap/'posed-replay', old_snap/'heldout-replay', base/'gaussian-run/face-supplement'):
        record_name = 'replay.json' if dir_.name == 'face-supplement' else 'result.json'
        doc = _read(dir_/record_name)
        source_path = dir_/(doc['mesh_file'] if dir_.name != 'face-supplement' else 'poses.npz')
        _pin(source_path, doc['mesh_sha256'] if dir_.name != 'face-supplement' else doc['poses_sha256'])
        frozen = npz(source_path)
        _require(same(frozen['faces'],motion['faces']), 'R27 topology changed.')
        metrics.append(frozen['vertices_world'])
        native.extend(doc['records'])
    frames = [v['frame'] for v in native]
    _require(frames == motion['frame_ids'].tolist() and len(set(frames)) == 32, 'Native roster changed.')
    _require(same(motion['vertices_world'], np.concatenate(metrics,axis=0)), 'Named posed mesh coordinates changed.')
    _require(meta['roles'] == ['training']*24+['pose_conditioned_holdout']*4+['face_only']*4
             and frames == cfg['train_frames']+cfg['holdout_frames']+cfg['auxiliary_frames'], 'Camera role changed.')
    for slot, (old, new) in enumerate(zip(native,meta['cameras'],strict=True)):
        cam = old['camera']
        _require(all(same(cam[key],new[key]) for key in ('frame','K','w2c','wh','pixel_convention')) and
                 new['pose_index'] == slot, 'Native K, pose or pixel convention changed.')
    _require(_sha(snap/'reference-mesh.npz') == _sha(old_snap/'reference-mesh.npz'),
             'Frozen reference mesh/static_offset changed.')
    return motion, meta, dict(camera_count=len(native), full_meshes_exact=True, camera_values_exact=True,
                               roles=meta['roles'], motion_sha256=_sha(snap/'drive/motion.npz'),
                               camera_manifest_sha256=_sha(snap/'drive/drive.json'))


def lineage(events, seed, next_id, material, cfg):
    pos = {int(stable):int(row) for row,stable in enumerate(seed)}
    seen_ids = set(pos)
    accepted, labels = [], []
    exact_views = set(cfg['train_frames']+cfg['auxiliary_frames'])
    for event in events:
        parents, child_sets = event.get('parent_ids', []), event.get('child_ids', [])
        _require(len(parents)==len(child_sets)==event['committed_parents'], 'Unaligned split receipt.')
        if not parents:
            continue
        _require(set(v['frame'] for v in event['views']) == exact_views and len(event['views']) == 28
                 and all(v['gate']['accepted'] for v in event['views']), 'Committed split without complete native guard.')
        _require(len(set(parents)) == len(parents) and all(p in pos for p in parents), 'Replaced parent absent.')
        for parent, children, owner, label, vote in zip(parents, child_sets, event['owner_faces'],
                          event['parent_material_labels'], event['source_view_counts'],strict=True):
            origin = pos.pop(parent)
            _require(material['observed'][origin] and not material['possible_protected_material'][origin] and
                     not material['unknown_unobserved_elongated'][origin] and label==int(material['labels'][origin]) and
                     label in (0,1) and owner==int(material['owner_faces'][origin]) and vote>=2 and len(children)==2,
                     'Split touched protected or unsupported material.')
            if label==1:
                _require(material['geometry_eligible'][origin], 'Split touched untrusted skin.')
            for child in children:
                _require(child == next_id and child not in seen_ids, 'Stable child ID reused or out of sequence.')
                pos[child] = origin
                seen_ids.add(child)
                next_id += 1
            accepted.append(dict(step=event['step'], parent_id=parent, child_ids=children, label=label,
                                 owner=owner, original_id=int(seed[origin]), native_views=28))
            labels.append(label)
    return pos,next_id,accepted,labels


def geometry_measures(before, after, owners, motion, current, origin):
    posed = motion['vertices_world'][:, motion['faces'][owners]]
    first, second = posed[:,:,1]-posed[:,:,0], posed[:,:,2]-posed[:,:,0]
    first,second = first.astype(np.float64),second.astype(np.float64)
    edge = np.linalg.norm(first,axis=2)
    height = np.linalg.norm(np.cross(first,second),axis=2)/edge
    scale = (edge+height)*.5
    _require(np.isfinite(scale).all() and (scale>0).all(), 'Invalid physical triangle scale.')
    local_a, local_b = local_metrics(before,origin),local_metrics(after,current)
    report, cap = {},{}
    for key in ('center','normal_sigma','tangent_major'):
        a = np.abs(local_a[key])*scale
        b = np.abs(local_b[key])*scale
        report[key+'_mm'] = dict(r22=quantiles(a*1000.),r27=quantiles(b*1000.),
                                 signed_change=quantiles((b-a)*1000.))
        if key != 'tangent_major':
            cap[key] = dict(max_increase_m=float(np.maximum(b-a,0).max()),
                            rows_above_tolerance=int(np.any(b-a>CAP_TOLERANCE_M,axis=0).sum()),
                            row_pose_pairs_above_tolerance=int((b-a>CAP_TOLERANCE_M).sum()))
    _require(not any(v['row_pose_pairs_above_tolerance'] for v in cap.values()),
             'Trusted skin exceeds original independent center/full-covariance cap.')
    report['minimum_axis_to_normal_angle_degrees'] = dict(r22=quantiles(local_a['angle']),r27=quantiles(local_b['angle']))
    return report,cap,local_a


def audit(subject, mode, step):
    base = LAB/f'shuolee-smpl-delivery-round22-{subject}'/'run'
    run = LAB/f'shuolee-smpl-delivery-round27-{subject}'/mode
    cfg = _read(run/'gaussian-run/config.json')
    raw0 = base/'gaussian-run/raw-step-018000'
    snap = run/'gaussian-run'/f'raw-step-{step:06d}'
    receipt = _read(snap/'result.json')
    state_path = snap/'main-state.pt'
    _pin(state_path,receipt['state_sha256'])
    raw_state = raw0/'state.pt'
    before_receipt = _read(raw0/'state.json')
    _pin(raw_state,before_receipt['state_sha256'])
    state0 = torch.load(raw_state,map_location='cpu',weights_only=True)['state']
    with torch.serialization.safe_globals([np.core.multiarray.scalar,np.dtype,np.dtypes.Float64DType]):
        state = torch.load(state_path,map_location='cpu',weights_only=True)
    _require(state['step']==receipt['step']==step and len(state['draws'])==step and
             set(state['draws'])<=set(range(24)) and state['pinned_inputs']==cfg['pinned_inputs'] and
             state['optimization']==cfg['optimization'], 'Training draw or frozen config identity differs.')
    mat_path = OUT/f'{subject}-material-visibility.npz'
    mat_receipt = mat_path.with_suffix('.json')
    mat_doc = _read(mat_receipt)
    _require(mat_doc['heldout_source_images_read']==0 and not
             set(mat_doc['retained_holdouts']) & set(mat_doc['classification_frames']), 'Held-out evidence leaked.')
    pins = dict(bound_ply=raw0/'bound-local.ply',raw_state=raw_state,fit=base/'fit/fitted.npz',
                material=mat_path,material_receipt=mat_receipt,
                aux_replay=base/'gaussian-run/face-supplement/replay.json')
    for name,path in pins.items():
        _pin(path,cfg['pinned_inputs'][name])
    _pin(mat_path,mat_doc['material_visibility_npz_sha256'])
    material = npz(mat_path)
    ids0 = arr(state0['model']['densification_ids']['tensor'])
    _require(same(ids0,material['stable_ids']), 'Material stable row IDs are unbound.')
    old = ply_parameters(raw0/'bound-local.ply')
    _require(all(same(old[key],arr(state0['parameters'][key]['value']['tensor'])) for key in PARAMS)
             and same(old['binding'],arr(state0['model']['binding']['tensor'])) and
             same(old['binding'],material['owner_faces']), 'R22 seed PLY differs from exact train state or owner labels.')
    ids = arr(state['stable_ids'])
    live_model = state['model']
    live = {key:arr(value) for key,value in zip(PARAMS,live_model[1:7],strict=True)}
    live['binding'] = arr(live_model[7])
    actual_ply = ply_parameters(snap/'drive/bound-local.ply')
    _require(all(same(live[key],actual_ply[key]) for key in (*PARAMS,'binding')) and len(ids)==receipt['points']
             and len(ids)==len(np.unique(ids)) and live_model[0]==0, 'Saved PLY does not represent this SH0 checkpoint.')
    pos,next_id,events,_ = lineage(state['split_events'],ids0,state0['model']['next_densification_id']['value'],material,cfg)
    _require(set(pos)==set(ids.tolist()) and next_id==state['next_stable_id'], 'Unexpected live topology or child ID clock.')
    row0 = np.asarray([pos[int(id_)] for id_ in ids])
    orig = np.isin(ids,ids0)
    _require(same(ids0[row0],arr(state['original_row_ids'])) and same(ids[orig],ids0[row0[orig]])
             and same(live['binding'],old['binding'][row0]), 'Original ID or owner changed.')
    aliases = dict(labels='labels',spill='silhouette_spill',observed='observed',skin='geometry_eligible',
                   geometry_only='unknown_unobserved_elongated',possible_protected_material='possible_protected_material')
    for key,old_key in aliases.items():
        if mode == 'probe' and key == 'possible_protected_material' and key not in state:
            continue
        _require(same(arr(state[key]),material[old_key][row0]),f'Material lineage changed: {key}')
    labels,seen,skin,unseen = (arr(state[key]) for key in ('labels','observed','skin','geometry_only'))
    protected = material['possible_protected_material'][row0]
    fixed = ~seen & ~unseen
    _require(orig[~seen|protected|(labels>=2)].all() and np.all(labels[unseen]==0) and not np.any(seen&unseen),
             'Protected topology changed or geometry-only row entered observed scope.')
    _require(same(arr(live_model[8]),np.bincount(live['binding'],minlength=len(live_model[8]))),
             'Binding counters differ from actual owner rows.')
    per_param = {}
    for name in PARAMS:
        moved = np.any((live[name]!=old[name][row0]).reshape(len(ids),-1),axis=1)
        allowed = orig if name in ('_opacity','_features_rest') else fixed if name=='_scaling' else ~seen
        _require(not np.any(moved & allowed),f'Protected field {name} was edited.')
        per_param[name] = dict(frozen_verified_rows=int(allowed.sum()),observed_changes_by_label=[
            int((moved&seen&(labels==i)).sum()) for i in range(6)])
    original_log, final_log = old['_scaling'][row0[unseen]].astype(np.float64),live['_scaling'][unseen].astype(np.float64)
    delta = final_log-original_log
    major = original_log.argmax(axis=1)
    others = np.arange(3)[None,:] != major[:,None]
    ratio = np.exp(delta[np.arange(len(major)),major])
    _require(not np.any(delta[others]!=0) and np.all(ratio<=1.+2e-7) and np.all(ratio>=.93-4e-7) and
             same(arr(state['caps'])[unseen],old['_scaling'][row0[unseen]]), 'Geometry-only row exceeded one-axis 7% cap.')
    motion,meta,pose = poses_and_cameras(base,snap,cfg,receipt)
    _require(same(motion['stable_ids'],ids) and same(motion['labels'],labels) and same(motion['spill'],arr(state['spill'])),
             'Motion archive changed row IDs or material evidence.')
    selected = np.flatnonzero(skin)
    geom, cap, local_a = geometry_measures(old,live,live['binding'][selected],motion,selected,row0[selected])
    captured_reference = arr(state['geometry_reference'])
    recorded_ref = captured_reference[select(arr(state['geometry_reference_ids']),ids[selected])]
    checked_ref = np.column_stack((local_a['center'],local_a['normal_sigma'],local_a['principal']))
    discrepancy = np.abs(recorded_ref-checked_ref)
    width_error_budget = np.maximum(local_a['principal'][:, -1], 1.)*5e-7
    _require(np.all(discrepancy <= width_error_budget[:, None]),
             f'Stored skin reference differs from independent R22 covariance: {discrepancy.max(axis=0)}.')
    grad_doc = run/'gaussian-run/native-gradient-audit.json'
    _pin(grad_doc,state['frozen_and_geometry']['native_gradient_audit_sha256'])
    gradients = _read(grad_doc)
    summary_grad = {k:{field: gradients[k][field] for field in ('active_rows_after_filter','active_unknown_rows_after_filter',
        'rows_with_gradient_by_material','active_mass_after_filter')} for k in ('rgb','alpha')}
    _require(all(v['active_unknown_rows_after_filter']>0 for v in summary_grad.values()), 'Native unknown material did not receive measured RGB/alpha gradient.')
    return dict(schema='shuolee-r27-independent-face-invariants/1',pass_=True,subject=subject,mode=mode,step=step,
        formal_commit=COMMIT,script_sha256=_sha(Path(__file__)),raw_checkpoint=str(state_path),
        checkpoint_sha256=receipt['state_sha256'],config_sha256=_sha(run/'gaussian-run/config.json'),
        r22_state_sha256=before_receipt['state_sha256'],r22_bound_ply_sha256=cfg['pinned_inputs']['bound_ply'],
        material_npz_sha256=cfg['pinned_inputs']['material'],r27_endpoint_result_sha256=_sha(snap/'result.json'),
        native_draw_count=len(state['draws']),unique_native_draw_slots=len(set(state['draws'])),
        raw_points=len(ids),original_points=len(ids0),frozen_irrelevant_rows=int(fixed.sum()),
        parameter_protection=per_param,geometry_only=dict(rows=int(unseen.sum()),modified_rows=int((delta!=0).any(1).sum()),
            original_major_scale_ratio=quantiles(ratio),max_reduction_fraction=float((1-ratio).max()) if len(ratio) else 0.,
            scope='One original longest width only. No RGB/alpha support is asserted for these rows.'),
        material_split=dict(protected_ambiguous_rows=int(protected.sum()),known_non_skin_rows=int((labels>=2).sum()),
                            committed_parent_counts_by_label={str(k):sum(e['label']==k for e in events) for k in (0,1)},
                            committed_events=events),
        trusted_skin=dict(rows=len(selected),original_surviving_rows=int((skin&orig).sum()),
            row_pose_checks=len(selected)*len(motion['frame_ids']),cap_tolerance_m=CAP_TOLERANCE_M,
            hard_cap=cap,physical_metrics=geom,reference_local_max_abs=discrepancy.max(axis=0).tolist()),
        frozen_pose_and_camera=pose,native_gradient_audit_sha256=_sha(grad_doc),native_initial_gradients=summary_grad,
        scope='CPU-only verification of geometry, frozen inputs, protection and runtime artifacts. '
              'Training gradient receipts describe native observations; model/render quality requires separate visual review.'),state['draws']


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument('subject',choices=('016','006'))
    p.add_argument('--steps',type=int,nargs='+',default=(10000,18000))
    p.add_argument('--mode',choices=('run','probe'),default='run')
    p.add_argument('--no-write',action='store_true')
    args=p.parse_args()
    outputs=[]
    for step in args.steps:
        receipt,draws=audit(args.subject,args.mode,step)
        outputs.append((receipt,draws))
        if not args.no_write:
            save(OUT/f'{args.subject}-face-invariants-{step:06d}.json',receipt)
        print(json.dumps(dict(subject=args.subject,step=step,rows=receipt['raw_points'],
            trusted=receipt['trusted_skin']['rows'],geometry_only=receipt['geometry_only'],
            max_violations=receipt['trusted_skin']['hard_cap'], committed=receipt['material_split']['committed_parent_counts_by_label'])),flush=True)
    if len(outputs)==2 and not args.no_write:
        first,second=outputs
        _require(second[1][:len(first[1])]==first[1],'R27 10k endpoint is not the exact draw prefix of 18k.')
        save(OUT/f'{args.subject}-face-invariants-paired.json',dict(
            schema='shuolee-r27-independent-face-invariants-paired/1',subject=args.subject,formal_commit=COMMIT,
            matching_draw_prefix=True,prefix_steps=len(first[1]),formal=[dict(step=doc['step'],
                checkpoint_sha256=doc['checkpoint_sha256'], independent_receipt_sha256=_sha(
                OUT/f'{args.subject}-face-invariants-{doc["step"]:06d}.json')) for doc,_ in outputs]))


if __name__=='__main__':
    main()

