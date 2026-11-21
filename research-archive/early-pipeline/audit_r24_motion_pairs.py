"""Separate R24 motion blending from learned node motion in frozen near pairs."""

import argparse
import json

import numpy as np
from scipy.spatial.transform import Rotation

from audit_r23_garment_seams import frames_and_scales, measures, neighbors
from audit_r23_numerical import LAB, garment_seed, read_state, _pin, _read, _require, _sha
from audit_r24_surface import OUTPUT


def endpoint(subject,number,probe=False):
    arm = LAB/f'shuolee-smpl-delivery-round{number}-{subject}'
    return arm/('probe-01/gaussian-run/raw-step-000001' if probe else 'run/gaussian-run/raw-step-018000')


def load_motion(path):
    manifest = _read(path/'drive/drive.json')
    file = path/'drive/motion.npz'
    _pin(file,manifest['files_sha256']['motion.npz'])
    with np.load(file,allow_pickle=False) as archive:
        return dict(archive)


def unit_quat(matrix):
    shape = matrix.shape[:-2]
    return Rotation.from_matrix(matrix.reshape(-1,3,3)).as_quat().reshape(*shape,4)


def native_graph(xyz,graph,row,geometry,reference,learned):
    origin, rotation, scale = geometry
    rest = graph['node_rest_xyz']
    live = graph['node_centers'] if learned else rest
    node_owner = graph['node_owner_faces']
    ids, weight = graph['row_node_ids'][row],graph['row_node_weights'][row].astype(np.float64)
    ref_xyz = xyz+np.sum((live-rest)[ids]*weight[...,None],axis=1)
    rest_interpolation = np.sum(rest[ids]*weight[...,None],axis=1)
    offset = xyz-rest_interpolation
    node_delta = live-origin[reference,node_owner]
    result, scales = [], []
    node_quat = unit_quat(rotation[:,node_owner])
    for frame in range(len(rotation)):
        if frame == reference:
            result.append(ref_xyz)
            scales.append(np.ones(len(xyz)))
            continue
        q = node_quat[frame][ids]
        signs = np.where(np.sum(q*q[:,:1],axis=2,keepdims=True)<0,-1.,1.)
        q = np.sum(q*signs*weight[...,None],axis=1)
        R = Rotation.from_quat(q).as_matrix()
        s = scale[frame,node_owner]
        rs = np.sum(s[ids]*weight,axis=1)
        node_pose = origin[frame,node_owner]+np.einsum('nij,nj->ni',rotation[frame,node_owner],node_delta)*s[:,None]
        moved = np.sum(node_pose[ids]*weight[...,None],axis=1)+np.einsum('nij,nj->ni',R,offset)*rs[:,None]
        result.append(moved)
        scales.append(rs)
    return np.stack(result),np.stack(scales)


def pair_report(pairs,arm,frames,reference,ids,owners,common):
    i,j = pairs.T
    xyz,scale = arm
    length = np.linalg.norm(xyz[:,i]-xyz[:,j],axis=2)*1000.
    expected = length[reference]*(scale[:,i]+scale[:,j])*.5
    residual = np.abs(length-expected)
    active = np.arange(len(frames)) != reference
    fi,pi = np.unravel_index(residual.argmax(),residual.shape)
    return dict(near_reference_pairs=int(common.sum()),frozen_pairs=len(pairs),
        overall=measures(residual[active].ravel()),
        common_near=measures(residual[active][:,common].ravel()) if common.any() else None,
        worst=dict(frame=frames[fi],ids=ids[pairs[pi]].tolist(),owners=owners[pairs[pi]].tolist(),
            reference_mm=float(length[reference,pi]),posed_mm=float(length[fi,pi]),
            mean_scale_expected_mm=float(expected[fi,pi]),residual_mm=float(residual[fi,pi])))


def audit(subject,probe=False,opaque=False):
    seed,_,_ = garment_seed(subject)
    r23_path,r24_path = endpoint(subject,23),endpoint(subject,24,probe)
    r23,r24 = [read_state(path/'main-state.pt') for path in (r23_path,r24_path)]
    manifest = _read(r24_path/'drive/drive.json')
    motion = load_motion(r24_path)
    old_motion = load_motion(r23_path)
    for key in ('frame_ids','vertices_world','faces','reference_index','camera_K','camera_w2c','camera_wh'):
        _require(np.array_equal(motion[key],old_motion[key]), 'R23/R24 fitted motion or source camera differs.')
    same = np.intersect1d(np.intersect1d(seed['ids'],r23['stable_ids'].numpy()),r24['stable_ids'].numpy())
    a,b,c = [np.searchsorted(ids,same) for ids in (seed['ids'],r23['stable_ids'].numpy(),r24['stable_ids'].numpy())]
    owners = seed['owner'][a]
    _require(np.array_equal(owners,r23['support_face_ids'].numpy()[b]) and
             np.array_equal(owners,r24['support_face_ids'].numpy()[c]),'Stable same-ID owner changed.')
    reference = motion['reference_index'].item()
    frames = motion['frame_ids'].tolist()
    origin,basis,length = frames_and_scales(motion['vertices_world'],motion['faces'])
    scale = length/length[reference]
    rotation = basis@basis[reference].swapaxes(-2,-1)
    base = seed['xyz'][a]
    old_xyz = r23['garment'][1].detach().numpy()[b]
    current_xyz = r24['garment'][1].detach().numpy()[c]
    _require(np.array_equal(current_xyz,base),'The R24 original splat centre moved independently.')
    arms = {}
    for name,xyz in (('r22',base),('r23',old_xyz)):
        result = np.einsum('fnij,nj->fni',rotation[:,owners],xyz-origin[reference,owners])*scale[:,owners,None]+origin[:,owners]
        arms[name] = (result,scale[:,owners])
    graph_path = r24_path/'drive'/manifest['garment_graph']
    _pin(graph_path,manifest['files_sha256'][graph_path.name])
    with np.load(graph_path,allow_pickle=False) as archive:
        graph = dict(archive)
    arms['r24_untrained_blend'] = native_graph(base,graph,c,(origin,rotation,scale),reference,learned=False)
    arms['r24_actual_graph'] = native_graph(base,graph,c,(origin,rotation,scale),reference,learned=True)
    same_pairs,adjacent_pairs = neighbors(base,owners,motion['faces'])
    if opaque:
        opacity = 1./(1.+np.exp(-seed['opacity'][a]))
        _require(np.array_equal(seed['opacity'][a],r23['garment'][6].detach().numpy()[b,0]) and
                 np.array_equal(seed['opacity'][a],r24['garment'][6].detach().numpy()[c,0]),
                 'An original row alpha differs across arms.')
        same_pairs,adjacent_pairs = [pairs[(opacity[pairs]>=.1).all(1)] for pairs in (same_pairs,adjacent_pairs)]
    result = dict(schema='shuolee-r24-fixed-near-pair-transport/1',subject=subject,common_original_ids=len(same),
        frames=frames,reference=frames[reference],graph_state='one-step probe' if probe else '18,000-step actual endpoint',
        method='Freeze R22 1–6 mm center pairs, up to 12 per row, split by same or shared-edge owner. Report length residual beyond mean local stock/blended scale across poses.',
        caveat='Shared-edge geometric proximity may be folds, contact or separate materials. Per-splat distance residual does not establish visible seams or calibrated garment ground truth.',
        opaque_proxy='Both unchanged peak point opacities >= 0.1. Occlusion and actual rendered contribution not measured.' if opaque else None,
        graph_sha256=_sha(graph_path),r23_state_sha256=_sha(r23_path/'main-state.pt'),r24_state_sha256=_sha(r24_path/'main-state.pt'))
    for group,pairs in (('same_owner',same_pairs),('shared_edge_owner',adjacent_pairs)):
        i,j = pairs.T
        lengths = np.stack([np.linalg.norm(xyz[reference,i]-xyz[reference,j],axis=1) for xyz,_ in arms.values()])
        common = ((lengths>=.001)&(lengths<=.006)).all(axis=0)
        result[group] = {name:pair_report(pairs,arm,frames,reference,same,owners,common) for name,arm in arms.items()}
    target = OUTPUT/f'{subject}-motion-pairs-{ "opaque-" if opaque else "" }{ "probe" if probe else "18k" }.json'
    with target.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(f'{target} sha256={_sha(target)}',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject',choices=('016','006'))
    parser.add_argument('--probe',action='store_true')
    parser.add_argument('--opaque',action='store_true')
    args = parser.parse_args()
    audit(args.subject,args.probe,args.opaque)
