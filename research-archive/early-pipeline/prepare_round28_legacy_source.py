"""Register an existing native SH0 bound model and its formal evaluation poses for R28."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from plyfile import PlyData
import torch

from trace_round22_face_pixels import DELIVERY, need, pin, read, sha


PROJECT=Path('/mnt/d/Projects/Shuolee-smpl-face-surface-r28')
OUT=DELIVERY.with_name('smpl-delivery-round28')/'input-audit'


def descriptor(path, digest=None):
    path=Path(path).resolve(strict=True)
    observed=sha(path)
    need(digest is None or observed==digest,f'Source bytes changed: {path}')
    return dict(file=str(path),sha256=observed)


def write_json(path, payload):
    with path.open('x',encoding='utf-8') as stream:
        json.dump(payload,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


def model_pairs(local_path,world_path,points,faces):
    bound,world=(PlyData.read(path,mmap=False)['vertex'].data for path in (local_path,world_path))
    need(len(bound)==len(world)==points and 'binding_0' in bound.dtype.names and 'binding_0' not in world.dtype.names,
         'The legacy PLY pair lost its row layout or binding.')
    names=tuple(f'f_dc_{i}' for i in range(3))+('opacity',)
    need(all(np.array_equal(bound[key],world[key]) for key in names)
         and all(not key.startswith('f_rest_') for key in world.dtype.names),
         'Legacy bound and world PLY differ in row appearance or SH degree.')
    binding=bound['binding_0']
    need(np.array_equal(binding,np.rint(binding)) and np.all((binding>=0)&(binding<faces)),
         'The legacy PLY owner triangle indices are invalid.')


def offset_sequence(fit,asset,reference_mesh):
    from shuolee_portrait.training import _frozen_smpl_surface

    base,linear,clothing,_,error=_frozen_smpl_surface(fit,asset)
    offset=reference_mesh['static_offset']
    need(offset.shape==tuple(base.shape[1:]) and offset.dtype==np.float32 and np.isfinite(offset).all()
         and not np.any(offset[~clothing]),'The existing bound offset is outside its verified clothing scope.')
    with torch.no_grad():
        delta=torch.from_numpy(offset)[...,None]
        vertices=torch.stack([v+(a@delta).squeeze(-1) for v,a in zip(base,linear,strict=True)]).numpy()
    return vertices,error


def check_selected(rows,replay,mesh,selection,replay_path):
    frames=[row['frame'] for row in rows]
    saved=replay['records']
    need(set(frames)==set(selection['selected_frames']) and len(set(frames))==len(frames)
         and frames==mesh['frame_ids'].tolist()==[row['frame'] for row in saved]
         and mesh['pose_indices'].tolist()==list(range(len(frames))),
         'Selected pose, fitted roster or source camera order differs.')
    result=[]
    for slot,(row,record) in enumerate(zip(rows,saved,strict=True)):
        name=row['frame']; cam=record['camera']; reason=selection['reasons'][name]
        need(all(record[key]==row[key] for key in ('source_frame_index','source_sha256','mask_sha256'))
             and all(cam[key]==row[key] for key in ('K','w2c','wh')) and record['pose_index']==slot
             and reason['source_index']==row['source_frame_index'], 'Selected frame or camera does not match native replay.')
        need(mesh['image_sha256'][slot]==row['source_sha256'] and mesh['mask_sha256'][slot]==row['mask_sha256']
             and int(mesh['source_frames'][slot])==row['source_frame_index'], 'The selected mesh uses other source pixels.')
        pin(row['image'],row['source_sha256']); pin(row['mask'],row['mask_sha256'])
        rgb=descriptor(replay_path.parent/record['image'],record['image_sha256'])
        result.append(dict(row,role='training',pose_index=slot,source_bin=reason['source_bin'],baseline_rgb=rgb))
    return result


def formal_registration(path,subject,fit,local,selection):
    review=read(path)
    need(review['optimizer_updates']==0,'Formal evaluation modified the source model.')
    if 'archive' in review:
        need(review['subject']==subject,'The formal evaluation belongs to another subject.')
        root=Path(review['archive'])
        fitted_digest=review['evaluation_fit_sha256']
        source=review['source_pins']
        unavailable=[]
    else:
        root=path.parent
        rendered=read(pin(root/'renders/result.json',review['render_receipt_sha256']))
        fitted=read(pin(root/'fit-result.json',rendered['fit_result_sha256']))
        source=read(pin(root/'inputs.json',fitted['input_receipt_sha256']))['input_sha256']
        fitted_digest=fitted['fitted_sha256']
        unavailable=review['unscored_outside_capture_arc']
        need(rendered['records']==review['records'] and fitted['evaluation_only'] and
             fitted['gaussian_optimizer_updates']==0 and set(review['protected_frames'])==set(selection['retained_holdouts']),
             'The formal evaluation records and reserved roles differ.')
    need(source[str(fit)]==sha(fit) and source[str(local)]==sha(local),
         'The formal evaluation was published from another base.')
    return review,pin(root/'evaluation-fitted.npz',fitted_digest),unavailable,source


def check_holdouts(raw,review,review_path,fit,selection,start,unavailable,source_pins):
    frames=fit['frame_ids'].tolist(); records=review['records']
    need(frames==[row['frame'] for row in records] and set(frames).isdisjoint(unavailable) and
         set(frames)|set(unavailable)==set(selection['retained_holdouts'])
         and not set(frames)&set(selection['selected_frames']), 'The formal evaluation roster differs from original holdouts.')
    result=[]
    for slot,(name,record) in enumerate(zip(frames,records,strict=True)):
        row=raw[name]; cam=record['posed_camera']; expected=fit['K_integer_centers'][slot].copy()
        expected[:2,2]+=.5
        need(np.array_equal(fit['w2c'][slot],cam['w2c']) and np.array_equal(expected,cam['K']) and
             np.array_equal(row['K'],cam['K']) and row['wh']==cam['wh'] and cam['frame']==name and
             cam['pose_index']==start+slot and int(fit['source_frames'][slot])==row['source_frame_index']==record['source_frame_index'] and
             fit['image_sha256'][slot]==row['source_sha256']==record['native_source_sha256'] and
             row['mask_sha256']==record['fit_mask_sha256'],
             f'Formal native evaluation camera, source or geometry differs: {name}')
        need(Path(row['image']).is_file() and Path(row['mask']).is_file(),f'Original heldout source is missing: {name}')
        fitted_mask=fit['mask_sha256'][slot].item()
        if fitted_mask==row['mask_sha256']:
            archived_fit_mask=dict(file=row['mask'],sha256=fitted_mask)
        else:
            matching=[Path(path) for path,digest in source_pins.items() if digest==fitted_mask
                      and Path(path).name==name+'.png' and Path(path).parent.name=='masks' and Path(path).is_file()]
            need(bool(matching),f'No exact historical fitting mask can verify the separately published heldout mask: {name}')
            archived_fit_mask=descriptor(matching[0],fitted_mask)
        panel=descriptor(review_path.parent/'renders'/record['image'],record['image_sha256'])
        result.append(dict(row,w2c=cam['w2c'],role='holdout',pose_index=start+slot,source_bin=None,
                           evaluation_role=record.get('role','within_capture_arc'),
                           formal_pose_screen_passed=record.get('pose_screen_passed'),
                           archived_fit_mask=archived_fit_mask,
                           formal_scoring_mask_sha256=record['scoring_mask_sha256'],
                           baseline_panel=panel))
    return result


def run(subject,source_root,native_root,formal_review_path):
    sys.path[:0]=[str(PROJECT/'src'),str(PROJECT/'src/avatar_core')]
    torch.set_num_threads(2)
    source_root=source_root.resolve(strict=True); native_root=native_root.resolve(strict=True)
    formal_review_path=formal_review_path.resolve(strict=True)
    terminal_path=source_root/'result.json'; terminal=read(terminal_path)
    registration=read(source_root/'run.json'); project_assets=Path(registration['assets'])
    asset_path=project_assets/'smplx/SMPLX_NEUTRAL.npz'
    replay_path=pin(source_root/terminal['bound_replay'],terminal['bound_replay_sha256'])
    replay=read(replay_path); snapshot=replay_path.parent.parent
    manifest_path=pin(snapshot/'bound-replay.json',replay['bound_replay_sha256']); manifest=read(manifest_path)
    expected=manifest['inputs_sha256']
    need(terminal['state']=='completed' and terminal['format_passed'] and terminal['sh_degree']==manifest['sh_degree']==0
         and replay['state']=='completed' and replay['optimizer_updates']==0 and manifest['points']==terminal['points'],
         'Use an existing completed verified SH0 endpoint.')
    fit_path=pin(source_root/'fit/fitted.npz',expected['fitted'])
    rows_path=pin(source_root/'data/training-inputs.json',expected['training_rows']); rows=read(rows_path)['records']
    local=pin(snapshot/'bound-local.ply',manifest['bound_ply_sha256'])
    world=pin(snapshot/'world.ply',expected['world_ply'])
    mesh_path=pin(replay_path.parent/replay['mesh_file'],replay['mesh_sha256'])
    reference_mesh_path=pin(snapshot/'reference-mesh.npz',expected['reference_mesh'])
    asset=pin(asset_path,manifest['asset_sha256'])
    official_path=source_root/'fit/official-fit-result.json'; official=read(official_path)
    pins=official['source']['input_files']; native_path=native_root/'data/inputs.json'; observations=native_root/'data/observations.npz'
    need(official['state']=='complete' and official['fit_sha256']==sha(fit_path)
         and official['training_inputs_sha256']==sha(rows_path),'The archived SMPL-X fit provenance differs.')
    native=read(pin(native_path,pins[str(native_path)])); observations=pin(observations,pins[str(observations)])
    raw={row['frame']:row for row in native['records']}
    selection_path=source_root/'fit/selection.json'; selection=read(selection_path)
    formal,formal_fit,unavailable,formal_pins=formal_registration(formal_review_path,subject,fit_path,local,selection)
    with np.load(fit_path,allow_pickle=False) as data:
        fit=dict(data)
    with np.load(mesh_path,allow_pickle=False) as data:
        main_mesh=dict(data)
    with np.load(formal_fit,allow_pickle=False) as data:
        held_fit=dict(data)
    with np.load(reference_mesh_path,allow_pickle=False) as data:
        reference_mesh=dict(data)
    model_pairs(local,world,terminal['points'],len(fit['faces']))
    need(np.array_equal(fit['faces'],main_mesh['faces']) and np.array_equal(fit['faces'],held_fit['faces'])
         and np.array_equal(fit['faces'],reference_mesh['faces']) and np.array_equal(fit['shared_betas'],held_fit['shared_betas'])
         and fit['frame_ids'].tolist()==main_mesh['frame_ids'].tolist()
         and terminal['reference_frame_id']==reference_mesh['reference_frame_id'].item(),
         'The original fit, formal fit and bound mesh disagree on shape, topology or frame names.')
    selected=check_selected(rows,replay,main_mesh,selection,replay_path)
    heldout=check_holdouts(raw,formal,formal_review_path,held_fit,selection,len(selected),unavailable,formal_pins)
    selected_vertices,selected_error=offset_sequence(fit,asset,reference_mesh)
    difference=float(np.max(np.abs(selected_vertices-main_mesh['vertices_world'])))
    need(difference<2e-6,'CPU official surface replay disagrees with the exact selected CUDA pose archive.')
    held_vertices,heldout_error=offset_sequence(held_fit,asset,reference_mesh)
    need(np.isfinite(held_vertices).all(),'The official heldout surface is not finite.')
    OUT.mkdir(parents=True,exist_ok=True)
    holds=OUT/f'{subject}-legacy-heldout-poses.npz'
    with holds.open('xb') as stream:
        np.savez_compressed(stream,vertices_world=held_vertices,faces=held_fit['faces'],frame_ids=held_fit['frame_ids'],
                            pose_indices=np.asarray([row['pose_index'] for row in heldout]))
    output=OUT/f'{subject}-legacy-source.json'
    result=dict(schema='shuolee-r28-legacy-bound-source/1',subject=subject,source_root=str(source_root),snapshot=str(snapshot),
        fit_sha256=sha(fit_path),training_inputs_sha256=sha(rows_path),fit=descriptor(fit_path),training_inputs=descriptor(rows_path),
        local_ply=descriptor(local),world_ply=descriptor(world),pose_mesh=descriptor(mesh_path),reference_mesh=descriptor(reference_mesh_path),
        heldout_pose_mesh=descriptor(holds),heldout_formal_fit=descriptor(formal_fit),formal_review=descriptor(formal_review_path),
        source_video=descriptor(registration['video'],registration['video_sha256']),native_inputs=descriptor(native_path),
        native_observations=descriptor(observations),asset=descriptor(asset),terminal=descriptor(terminal_path),
        bound_manifest=descriptor(manifest_path),selected_pose_replay=descriptor(replay_path),
        selected_records=selected,heldout_records=heldout,auxiliary_records=[],
        reserved_holdouts=selection['retained_holdouts'],unavailable_holdouts=unavailable,
        reference_frame=terminal['reference_frame_id'],reference_pose_index=terminal['reference_pose_index'],
        base_step=terminal['steps'],point_count=terminal['points'],
        selected_cpu_mesh_max_abs_m=difference,selected_fit_cpu_replay_max_abs_m=selected_error,
        heldout_fit_cpu_replay_max_abs_m=heldout_error,
        stable_id_policy='Assign initial IDs by the verified bound-local PLY row index, starting from zero.',
        heldout_pose_policy='Replay official heldout fit and transport the exact saved static clothing offset by per-vertex LBS.',
        heldout_source_images_read=0,source_sha256=sha(__file__))
    write_json(output,result)
    print(dict(subject=subject,source=str(output),sha256=sha(output),selected=len(selected),holdout=len(heldout),
               auxiliary=0,points=terminal['points'],selected_cpu_mesh_max_abs_m=difference),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject',required=True)
    for field in ('source-root','native-root','formal-review'):
        parser.add_argument(f'--{field}',type=Path,required=True)
    args=parser.parse_args()
    run(args.subject,args.source_root,args.native_root,args.formal_review)
