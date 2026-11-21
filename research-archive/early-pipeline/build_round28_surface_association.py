"""Bind native skin observations to a local, frozen SMPL-X surface proxy."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from build_round26_native_face_material import MATERIAL_REVIEW, array_sha
from build_round27_material_visibility import source_grid
from trace_round22_face_pixels import (DELIVERY, REVIEW_SHA, checked, fields, load_subject, need,
                                      pin, ply, posed_model, read, sha)


PROJECT_R28 = Path('/mnt/d/Projects/Shuolee-smpl-face-surface-r28')
TRANSFER = PROJECT_R28.with_name(PROJECT_R28.name+'-transfer')
R27 = DELIVERY.with_name('smpl-delivery-round27')/'input-audit'
OUT = DELIVERY.with_name('smpl-delivery-round28')/'input-audit'


def mesh_depth_at_skin(skin, box, vertices_world, faces, K, w2c):
    from shuolee_portrait.face_supplement import _front_depth

    depth = np.full(skin.shape, np.inf, dtype=np.float32)
    ys, xs = np.nonzero(skin)
    if not len(ys):
        return depth
    camera_vertices = vertices_world@w2c[:3,:3].T+w2c[:3,3]
    need(bool((camera_vertices[:,2] > .2).all()), 'Frozen proxy mesh crosses near camera plane.')
    projected = camera_vertices@K.T
    triangles = (projected[:,:2]/projected[:,2:])[faces]
    x0, y0, x1, y1 = box
    overlaps = ((triangles.max(1) >= (x0,y0)) & (triangles.min(1) <= (x1,y1))).all(1)
    points = np.column_stack((xs+x0+.5, ys+y0+.5))
    depth[ys,xs] = _front_depth(points, camera_vertices, faces[overlaps], K).astype(np.float32)
    return depth


def run(subject):
    from PIL import Image

    sys.path[:0] = [str(PROJECT_R28/'src'), str(PROJECT_R28/'src/avatar_core')]
    from shuolee_portrait.face_surface_association_r28 import SurfaceView, build_surface_association

    torch.set_num_threads(2)
    context = load_subject(subject, read(pin(DELIVERY/'review-receipt.json', REVIEW_SHA)))
    model, slots = posed_model(context)
    baseline = Path(context['arm']['run'])
    pose = context['plan']['poses']['r22']
    receipt_path = R27/f'{subject}-material-visibility.json'
    receipt = read(receipt_path)
    npz_path = pin(receipt_path.with_suffix('.npz'), receipt['material_visibility_npz_sha256'])
    need(receipt['baseline_bound_ply_sha256'] == context['hashes']['bound_local'] and
         receipt['baseline_raw_ply_sha256'] == context['hashes']['world'] and
         receipt['heldout_source_images_read'] == 0, 'R27 material rows differ from the R22 start.')
    with np.load(npz_path, allow_pickle=False) as archive:
        ids, owners, selected = (archive[name] for name in ('stable_ids','owner_faces','geometry_eligible'))
        need(np.array_equal(ids, context['stable']) and np.array_equal(owners, context['owner']),
             'Material IDs and owner faces differ from R22.')
    names = list(slots)
    main_path = pin(baseline/'data/training-inputs.json', receipt['training_inputs_sha256'])
    main = {row['frame']:row for row in read(main_path)['records']}
    fit_path = pin(baseline/'fit/fitted.npz', receipt['fit_sha256'])
    with np.load(fit_path, allow_pickle=False) as fit:
        need(fit['frame_ids'].tolist() == names and np.array_equal(fit['faces'], model.faces.numpy()),
             'Original mesh pose sequence changed.')
        reference = slots[fit['reference_frame_id'].item()]
        observations = {name:(fit['observed_native_keypoints'][i], fit['joints_projected_native'][i])
                        for i,name in enumerate(names)}
    proof = pose['proof']
    admission_path = checked(proof['supplement_admission'])
    admission = read(admission_path)
    auxiliary = {row['frame']:row for row in pose['supplement']['frames']}
    aux_names = list(auxiliary)
    need(aux_names == admission['chosen_frame_ids'] == receipt['admitted_auxiliary_frames'],
         'Auxiliary frame roles differ.')
    sources = {row['frame']:row for row in read(pin(baseline/'data/inputs.json',
               admission['input_sha256']['native_inputs']))['records'] if row['frame'] in auxiliary}
    candidates = pin(baseline/'fit/joint-candidates.npz', admission['input_sha256']['joint_candidates'])
    with np.load(candidates, allow_pickle=False) as data:
        candidates_names = data['frame_ids'].tolist()
        for name in aux_names:
            i = candidates_names.index(name)
            need(data['image_sha256'][i] == sources[name]['source_sha256'] and
                 data['mask_sha256'][i] == sources[name]['mask_sha256'], 'Auxiliary landmarks changed source.')
            observations[name] = (data['observed_native_keypoints'][i],data['joints_projected_native'][i])
    observe = pin(baseline/'data/observations.npz', admission['input_sha256']['observations'])
    with np.load(observe, allow_pickle=False) as data:
        source_names = data['frame_ids'].tolist()
        joints = [data['joint_names'].tolist().index(f'face-{i}') for i in range(17)]
        detector = {name:tuple(data[key][source_names.index(name),joints]
                    for key in ('keypoints','scores','flip_disagreement')) for name in names+aux_names}
    held_root = baseline/'gaussian-run/step-018000/heldout-replay'
    held = read(held_root/'result.json')
    held_path = pin(held_root/held['mesh_file'], held['mesh_sha256'])
    aux_path = checked(proof['supplement_poses'])
    faces = model.faces.numpy()
    with np.load(held_path, allow_pickle=False) as data:
        need(data['frame_ids'].tolist() == receipt['retained_holdouts'] and
             np.array_equal(data['faces'], faces), 'Evaluation pose roster changed.')
        held_vertices = torch.from_numpy(data['vertices_world'])
    with np.load(aux_path, allow_pickle=False) as data:
        need(data['frame_ids'].tolist() == aux_names and np.array_equal(data['faces'],faces),
             'Auxiliary pose roster changed.')
        aux_vertices = torch.from_numpy(data['vertices_world'])
    model.vertices_sequence = torch.cat((model.vertices_sequence,held_vertices,aux_vertices))
    frame_slots = {**slots, **{name:i+28 for i,name in enumerate(aux_names)}}
    need(len(frame_slots) == 28 and set(frame_slots) == set(receipt['classification_frames']),
         'Association requires exactly the authorized source frames.')
    material_path = pin(DELIVERY/'input-audit/face-pixel-attribution.json', MATERIAL_REVIEW)
    annotations = read(material_path)['subjects'][subject]['native_annotation']
    views, view_receipts = [], []
    for saved in receipt['views']:
        frame, box = saved['frame'], tuple(saved['box'])
        slot = frame_slots[frame]
        row = main[frame] if slot < 24 else sources[frame]
        require_hashes = (row['source_sha256'],row['mask_sha256'])
        need(require_hashes == (saved['source_sha256'],saved['mask_sha256']), 'Wrong native image or matte.')
        if slot < 24:
            camera, valid = row, None
        else:
            saved_aux = auxiliary[frame]
            camera = saved_aux['camera']
            with Image.open(pin(saved_aux['valid_path'], saved_aux['valid_mask_sha256'])) as image:
                valid = np.asarray(image.convert('L')) > 0
            need(int(valid.sum()) == saved_aux['valid_pixels'] and tuple(saved_aux['box']) == box,
                 'Auxiliary face-only mask changed.')
        obs, projected = observations[frame]
        grid, interior, scope, _ = source_grid(row, obs, projected, detector[frame], box, annotations.get(frame,{}), valid)
        need(all(array_sha(array) == saved[key] for array,key in
                 ((grid,'label_mask_sha256'),(interior,'interior_sha256'),(scope,'pixel_scope_sha256'))),
             'Source skin/exclusion pixels differ from the signed R27 observation.')
        K, w2c = np.asarray(camera['K']), np.asarray(camera['w2c'])
        skin = grid == 1
        excluded = np.isin(grid, [2,3,4,5])
        depth = mesh_depth_at_skin(skin,box,model.vertices_sequence[slot].numpy(),faces,K,w2c)
        views.append(SurfaceView(slot=slot,source_bin=saved['source_bin'],K=K,w2c=w2c,box=box,
                                 skin=skin,mesh_depth=depth,excluded=excluded,frame=frame))
        view_receipts.append(dict(frame=frame,pose_index=slot,source_bin=saved['source_bin'],K=K.tolist(),
            w2c=w2c.tolist(),box=box,source_sha256=require_hashes[0],mask_sha256=require_hashes[1],
            material_grid_sha256=saved['label_mask_sha256'],skin_pixels=int(skin.sum()),
            skin_depth_pixels=int(np.isfinite(depth).sum()),mesh_proxy_depth_sha256=array_sha(depth)))
    indices = np.flatnonzero(selected)
    centers = []
    with torch.no_grad():
        for slot in range(32):
            model.select_mesh_by_timestep(slot)
            centers.append(model.get_xyz[indices].numpy().copy())
    association, report = build_surface_association(model.vertices_sequence.numpy(),faces,ids[indices],owners[indices],
        np.stack(centers),np.ones(len(indices),bool),views,reference)
    OUT.mkdir(parents=True,exist_ok=True)
    target = OUT/f'{subject}-surface-association.npz'
    receipt_target = target.with_suffix('.json')
    need(not target.exists() and not receipt_target.exists(), 'The R28 association artifact already exists.')
    np.savez_compressed(target,**association.arrays())
    document = dict(schema='shuolee-r28-native-surface-association/1',subject=subject,
        baseline_bound_ply_sha256=context['hashes']['bound_local'],baseline_raw_state_sha256=context['hashes']['state'],
        material_receipt_sha256=sha(receipt_path),material_npz_sha256=sha(npz_path),
        association_npz_sha256=sha(target),eligible_skin_rows=len(indices),training_frames=names,
        auxiliary_frames=aux_names,holdout_source_images_read=0,view_receipts=view_receipts,association=report,
        source_sha256=sha(__file__),geometry_source='Frozen posed SMPL-X proxy with perspective-correct depth at native pixel centers.',
        fit_sha256=sha(fit_path),heldout_mesh_sha256=sha(held_path),auxiliary_mesh_sha256=sha(aux_path))
    receipt_target.write_text(json.dumps(document,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(subject=subject,archive=str(target),summary=report),ensure_ascii=False),flush=True)


def run_legacy(source_path):
    sys.path[:0] = [str(TRANSFER/'src'), str(TRANSFER/'src/avatar_core')]
    from scene.smplx_gaussian_model import SMPLXGaussianModel
    from shuolee_portrait.face_surface_association_r28 import SurfaceView, build_surface_association

    torch.set_num_threads(2)
    source = read(source_path)
    need(source['schema'] == 'shuolee-r28-legacy-bound-source/1', 'Expected an immutable legacy source.')
    subject = source['subject']
    receipt_path = OUT/f'{subject}-material-visibility.json'
    receipt = read(receipt_path)
    npz_path = pin(receipt_path.with_suffix('.npz'), receipt['material_visibility_npz_sha256'])
    local_path = checked(source['local_ply'])
    need(receipt['baseline_bound_ply_sha256'] == sha(local_path) and
         receipt['baseline_raw_ply_sha256'] == source['world_ply']['sha256'] and
         receipt['heldout_source_images_read'] == 0, 'Wrong source for the transfer material evidence.')
    with np.load(npz_path, allow_pickle=False) as data:
        ids, owners, skin_rows = (data[key] for key in ('stable_ids','owner_faces','geometry_eligible'))
    local = ply(local_path)
    need(np.array_equal(ids,np.arange(len(local))) and np.array_equal(owners,local['binding_0']),
         'Transfer material row order differs from its own bound PLY.')
    main_rows = source['selected_records']
    frames = [row['frame'] for row in main_rows]
    with np.load(checked(source['pose_mesh']), allow_pickle=False) as data:
        vertices, faces = data['vertices_world'], data['faces']
        need(data['frame_ids'].tolist() == frames, 'Training mesh roster differs.')
    with np.load(checked(source['heldout_pose_mesh']), allow_pickle=False) as data:
        need(data['frame_ids'].tolist() == [row['frame'] for row in source['heldout_records']]
             and np.array_equal(data['faces'],faces), 'Evaluation pose roster differs.')
        vertices = np.concatenate((vertices,data['vertices_world']))
    model = SMPLXGaussianModel(0)
    model._xyz = torch.from_numpy(fields(local,'xyz'))
    model.binding = torch.from_numpy(owners)
    model.faces = torch.from_numpy(faces).long()
    model.vertices_sequence = torch.from_numpy(vertices)
    model.num_timesteps = len(vertices)
    views, pins = [], []
    with np.load(checked(receipt['native_skin_grid']), allow_pickle=False) as data:
        need(data['frame_ids'].tolist() == frames, 'Signed source-grid order differs from the fit.')
        for i, (row,saved) in enumerate(zip(main_rows,receipt['views'],strict=True)):
            grid, box = data[f'grid_{i:03d}'], tuple(map(int,data['boxes'][i]))
            need(row['frame'] == saved['frame'] and array_sha(grid) == saved['label_mask_sha256'],
                 'Signed transfer source-grid hash mismatch.')
            K,w2c = np.asarray(row['K']),np.asarray(row['w2c'])
            skin = grid == 1
            depth = mesh_depth_at_skin(skin,box,vertices[i],faces,K,w2c)
            views.append(SurfaceView(i,row['source_bin'],K,w2c,box,skin,depth,
                                    np.isin(grid,[2,3,4,5]),row['frame']))
            pins.append(dict(frame=row['frame'],pose_index=i,source_bin=row['source_bin'],box=box,
                source_sha256=row['source_sha256'],mask_sha256=row['mask_sha256'],
                material_grid_sha256=saved['label_mask_sha256'],mesh_proxy_depth_sha256=array_sha(depth)))
    indices = np.flatnonzero(skin_rows)
    centers = []
    with torch.no_grad():
        for slot in range(len(vertices)):
            model.select_mesh_by_timestep(slot)
            centers.append(model.get_xyz[indices].numpy().copy())
    association, report = build_surface_association(vertices,faces,ids[indices],owners[indices],np.stack(centers),
        np.ones(len(indices),bool),views,source['reference_pose_index'])
    target = OUT/f'{subject}-surface-association.npz'
    need(not target.exists(), 'The transfer association already exists.')
    np.savez_compressed(target,**association.arrays())
    document = dict(schema='shuolee-r28-native-surface-association/1',subject=subject,
        baseline_bound_ply_sha256=sha(local_path),baseline_raw_state_sha256=None,legacy_source_sha256=sha(source_path),
        material_receipt_sha256=sha(receipt_path),material_npz_sha256=sha(npz_path),association_npz_sha256=sha(target),
        eligible_skin_rows=len(indices),training_frames=frames,auxiliary_frames=[],holdout_source_images_read=0,
        view_receipts=pins,association=report,source_sha256=sha(__file__),
        geometry_source='Own archived SMPL-X mesh, depth on its exact native source cameras.',fit_sha256=source['fit_sha256'])
    target.with_suffix('.json').write_text(json.dumps(document,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(subject=subject,associated_rows=report['associated_rows'],archive=str(target)),ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--subject',choices=('016','006'))
    source.add_argument('--legacy-source',type=Path)
    args = parser.parse_args()
    if args.legacy_source:
        run_legacy(args.legacy_source)
    else:
        run(args.subject)

