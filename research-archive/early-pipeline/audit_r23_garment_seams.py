"""Measure matched near-neighbour transport excess across inherited faces."""

import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree
from smplx.joint_names import JOINT_NAMES

from audit_r23_numerical import LAB, OUTPUT, garment_seed, read_state, _pin, _read, _require, _sha


LOWER, UPPER, MAX_NEIGHBORS = .001, .006, 12


def frames_and_scales(vertices, faces):
    tri = vertices[:, faces].astype(np.float64)
    edge, second = tri[:, :, 1]-tri[:, :, 0], tri[:, :, 2]-tri[:, :, 0]
    length = np.linalg.norm(edge,axis=-1,keepdims=True)
    tangent = edge/length
    normal = np.cross(tangent,second)
    normal /= np.linalg.norm(normal,axis=-1,keepdims=True)
    axis = np.cross(tangent,normal)
    scale = .5*(length[...,0]+np.abs(np.sum(axis*second,axis=-1)))
    return tri.mean(axis=2), np.stack((tangent,normal,axis),axis=-1), scale


def neighbors(xyz, owner, topology):
    distance, indices = cKDTree(xyz).query(xyz, k=MAX_NEIGHBORS+1, distance_upper_bound=UPPER, workers=2)
    start, end = np.where(np.isfinite(distance) & (distance>=LOWER))
    pairs = np.unique(np.sort(np.column_stack((start,indices[start,end])),axis=1),axis=0)
    shared = owner[pairs[:,0]] == owner[pairs[:,1]]
    a,b = [topology[owner[pairs[:,i]]] for i in (0,1)]
    edge_adjacent = (a[:,:,None] == b[:,None,:]).sum(axis=(1,2)) == 2
    return pairs[shared],pairs[edge_adjacent]


def measures(value):
    return dict(n=len(value),p50_mm=float(np.median(value)),p95_mm=float(np.quantile(value,.95)),
                p99_mm=float(np.quantile(value,.99)),max_mm=float(value.max()),
                exceed_2mm_fraction=float(np.mean(value>2.)))


def summarize(pairs, poses, scales, reference, ids, owner, names, regions):
    i,j = pairs.T
    difference = np.linalg.norm(poses[:,:,i]-poses[:,:,j],axis=-1)*1000.
    expected = difference[:,reference,None,:]*(scales[:,owner[i]]+scales[:,owner[j]])[None,:,:]*.5
    residual = difference-expected
    excess = np.abs(residual)
    best = {}
    for arm, values in zip(('r22','r23'),excess,strict=True):
        frame,row = np.unravel_index(values.argmax(),values.shape)
        left,right = pairs[row]
        arm_i = ('r22','r23').index(arm)
        best[arm] = dict(frame=names[frame],r22_reference_mm=float(difference[0,reference,row]),
            r23_reference_mm=float(difference[1,reference,row]), reference_pose_id=names[reference],
            posed_mm=float(difference[arm_i,frame,row]),expected_scaled_mm=float(expected[arm_i,frame,row]),
            signed_difference_mm=float(residual[arm_i,frame,row]), ids=ids[[left,right]].tolist(),
            faces=owner[[left,right]].tolist(), regions=[regions[k] for k in owner[[left,right]]])
    after_near = (difference[1,reference] >= LOWER*1000.) & (difference[1,reference] <= UPPER*1000.)
    keep_frames = np.arange(len(names)) != reference
    overall = {arm:measures(excess[i,keep_frames].ravel()) for i,arm in enumerate(('r22','r23'))}
    still_near = {arm:measures(excess[i,keep_frames][:,after_near].ravel())
                  for i,arm in enumerate(('r22','r23'))} if after_near.any() else {}
    return dict(frozen_pairs=len(pairs),still_near_pairs=int(after_near.sum()),overall=overall,
                still_near=still_near,worst=best)


def main(subject):
    seed, old, _ = garment_seed(subject)
    root = LAB/f'shuolee-smpl-delivery-round23-{subject}'/'run/gaussian-run/raw-step-018000'
    state = read_state(root/'main-state.pt')
    proof = _read(root/'drive/drive.json')
    _require(_sha(root/'main-state.pt') == _read(OUTPUT/f'{subject}-numerical-paired.json')['endpoints'][-1]['checkpoint_sha256'],
             'R23 seam input differs from the independent numerical receipt.')
    motion = root/'drive/motion.npz'
    _pin(motion,proof['files_sha256']['motion.npz'])
    with np.load(motion,allow_pickle=False) as archive:
        ref = archive['reference_index'].item()
        names = archive['frame_ids'].tolist()
        verts = archive['vertices_world']
        topology = archive['faces']
    shared,ia,ib = np.intersect1d(seed['ids'],state['stable_ids'].numpy(),assume_unique=True,return_indices=True)
    owner = seed['owner'][ia]
    _require(np.array_equal(owner,state['support_face_ids'].numpy()[ib]),'Common R22 and R23 row owner changed.')
    old_xyz,new_xyz = seed['xyz'][ia],state['garment'][1].detach().numpy()[ib]
    centroids,basis,face_scale = frames_and_scales(verts,topology)
    scales = face_scale/face_scale[ref]
    rotations = basis @ basis[ref].swapaxes(-2,-1)
    ref_origins = centroids[ref,owner]
    points = []
    for xyz in (old_xyz,new_xyz):
        posed = (np.einsum('fnij,nj->fni',rotations[:,owner],xyz-ref_origins)
                 *scales[:,owner,None]+centroids[:,owner])
        _require(np.max(np.abs(posed[ref]-xyz))<1e-9,'Reference similarity is not identity.')
        points.append(posed)
    poses = np.stack(points)
    assets = Path(_read(old/'run.json')['assets'])/'smplx/SMPLX_NEUTRAL.npz'
    with np.load(assets,allow_pickle=False) as model:
        _require(np.array_equal(model['f'],topology),'Body regions and fitted topology differ.')
        dominant = model['weights'][topology].mean(axis=1).argmax(axis=1)
    regions = [JOINT_NAMES[i] for i in dominant]
    same,adjacent = neighbors(old_xyz,owner,topology)
    report = dict(schema='shuolee-r23-transport-near-pairs/1',subject=subject,shared_original_ids=len(shared),
        frame_count=len(names),rule='Frozen R22 Euclidean center neighbors 1–6 mm and at most twelve per row. Same or shared-edge owner only.',
        caveat='Near Gaussians can be different cloth folds, contact, or density samples. No physical same-material labels. Both arms use the same stock triangle motion; excess is a diagnostic, not seam ground truth.',
        same_owner=summarize(same,poses,scales,ref,shared,owner,names,regions),
        edge_adjacent_owner=summarize(adjacent,poses,scales,ref,shared,owner,names,regions))
    target = OUTPUT/f'{subject}-transport-near-pairs.json'
    with target.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
    print(f'{target} sha256={_sha(target)}')


if __name__ == '__main__':
    main(sys.argv[1])
