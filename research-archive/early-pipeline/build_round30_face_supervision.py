"""Sign visibility-safe native face masks and angular training weights for R30."""

import argparse
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import binary_erosion

from build_round26_native_face_material import MATERIAL_REVIEW, array_sha, local_hull, protected_faces
from build_round27_material_visibility import decode
from build_round29_native_coverage import (OUT as R29, SUBJECTS, legacy_inputs, old_inputs, r22_inputs,
                                         raster_surface, regions)
from prepare_round28_legacy_source import descriptor, write_json
from trace_round22_face_pixels import DELIVERY, checked, material_mask, need, pin, read, sha


PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-face-surface-r30')
OUT = DELIVERY.with_name('smpl-delivery-round30')/'input-audit'
POLICY = dict(yaw_sector_deg=20, pitch_sector_deg=15, max_yaw_deg=78., max_pitch_deg=55.,
              minimum_span_px=24., minimum_confidence=.50, max_pair_error_px=6.,
              max_pair_error_over_span=.08, minimum_face_pixels=40, minimum_skin_pixels=12,
              minimum_gradient=2./255., max_saturated_fraction=.30, minimum_surface_facing=.15)


def native_face_regions(record, faces, head_faces, yaw, pitch):
    """Use measured native facial features and visible posed head triangles in one native crop."""
    from shuolee_portrait.face_supplement import REGIONS

    c = POLICY
    x0,y0,x1,y1 = record.box
    empty = np.zeros((y1-y0,x1-x0),bool)
    angle = dict(head_yaw_deg=float(yaw),head_pitch_deg=float(pitch))
    if abs(yaw)>c['max_yaw_deg'] or abs(pitch)>c['max_pitch_deg']:
        reason = 'rear_head' if abs(yaw)>=90. else 'head_angle_outside_visible_face_range'
        return empty,empty.copy(),dict(angle,status=reason,quality=0.)
    obs,proj = record.obs,record.proj
    finite = np.isfinite(obs).all(1)&np.isfinite(proj).all(1)
    reliable = finite&(obs[:,2]>=c['minimum_confidence'])
    inner = obs[67:118,:2][reliable[67:118]]
    span = float(np.ptp(inner,axis=0).max()) if len(inner)>=8 else 0.
    if span<c['minimum_span_px']:
        return empty,empty.copy(),dict(angle,status='insufficient_measured_face_span',quality=0.,face_span_px=span)
    error = np.linalg.norm(obs[:,:2]-proj,axis=1)
    paired = reliable&(error<=min(c['max_pair_error_px'],c['max_pair_error_over_span']*span))
    x0,y0,x1,y1 = record.box
    mask = decode(pin(record.row['mask'],record.row['mask_sha256']),record.box,'L')
    matte = mask>=252
    if record.allowed is not None:
        need(record.allowed.shape==matte.shape and np.all(matte[record.allowed]), 'Changed admitted source mask.')
        matte &= record.allowed
    for l,t,r,b in record.row.get('other_person_boxes_xyxy',()):
        left,top,right,bottom = max(x0,int(l)),max(y0,int(t)),min(x1,int(r)),min(y1,int(b))
        if left<right and top<bottom:
            matte[top-y0:bottom-y0,left-x0:right-x0] = False
    for positions in (obs[:,:2],proj):
        pixel = np.rint(np.where(np.isfinite(positions),positions,0.)-[x0,y0]).astype(np.int64)
        x,y = pixel.T
        inside = (x>=0)&(x<x1-x0)&(y>=0)&(y<y1-y0)
        paired &= inside&matte[y.clip(0,y1-y0-1),x.clip(0,x1-x0-1)]
    groups = {name:REGIONS[name][paired[REGIONS[name]]] for name in ('nose','mouth','right_eye','left_eye')}
    counts = {name:len(ids) for name,ids in groups.items()}
    detail = dict(angle,face_span_px=span,paired_points=counts)
    if counts['nose']<3 or counts['mouth']<4 or max(counts['right_eye'],counts['left_eye'])<3:
        return empty,empty.copy(),dict(detail,status='insufficient_original_and_fitted_face_features',quality=0.)
    core = np.flatnonzero(paired[67:118])+67
    hull = local_hull(obs[core,:2],record.box)&local_hull(proj[core],record.box)
    depth, owner, _ = raster_surface(record.vertices,faces,record.row,record.box)
    camera = np.asarray(record.row['w2c'])
    position = -camera[:3,:3].T@camera[:3,3]
    triangles = record.vertices[faces]
    normals = np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    direction = position-triangles.mean(1)
    facing = (normals*direction).sum(1)>=c['minimum_surface_facing']*(
        np.linalg.norm(normals,axis=1)*np.linalg.norm(direction,axis=1))
    visible = np.isfinite(depth)&(owner>=0)&(head_faces&facing)[owner.clip(0)]
    excluded = np.isin(record.grid,[3,4])
    for name in ('hair','glasses_rim'):
        signed = record.annotation.get('masks',{}).get(name,{})
        if signed.get('validated'):
            full = material_mask(record.row['wh'],signed['polygons'])
            need(array_sha(full)==signed['full_roi_sha256'], 'Changed signed protected material polygon.')
            excluded |= full[y0:y1,x0:x1]
    face = binary_erosion(matte&hull&visible&~excluded,iterations=1)
    skin, _, protected, rgb = regions(record)
    skin &= face&~protected
    detail.update(face_pixels=int(face.sum()),skin_pixels=int(skin.sum()))
    if detail['face_pixels']<c['minimum_face_pixels'] or detail['skin_pixels']<c['minimum_skin_pixels']:
        return empty,empty.copy(),dict(detail,status='insufficient_visible_native_face_or_skin',quality=0.)
    luma = rgb@np.asarray([.2126,.7152,.0722],np.float32)
    interior = binary_erosion(face)[1:-1,1:-1]
    dx = (luma[1:-1,2:]-luma[1:-1,:-2])*.5
    dy = (luma[2:,1:-1]-luma[:-2,1:-1])*.5
    sharpness = float(np.sqrt(np.mean(dx[interior]**2+dy[interior]**2))) if interior.any() else 0.
    saturated = float(((luma[face]<=2./255.)|(luma[face]>=253./255.)).mean())
    median = float(np.median(error[np.concatenate(list(groups.values()))]))
    detail.update(luma_gradient_rms=sharpness,saturated_fraction=saturated,median_feature_error_px=median)
    if sharpness<c['minimum_gradient'] or saturated>c['max_saturated_fraction']:
        return empty,empty.copy(),dict(detail,status='insufficient_native_detail_or_exposure',quality=0.)
    quality = float(np.clip(np.sqrt(span/72.),.5,1.5)*np.clip(sharpness/.03,.35,1.)
                    *(1.-saturated)/(1.+(median/3.)**2))
    return face,skin,dict(detail,status='visible_face',quality=quality)


def directions(path):
    from shuolee_portrait.face_supplement import _directions

    with np.load(path,allow_pickle=False) as data:
        yaw,pitch,_ = _directions(data,0)
        frames = data['frame_ids'].tolist()
    need(len(set(frames))==len(frames) and np.isfinite(yaw).all() and np.isfinite(pitch).all(),
         'Invalid fitted face direction roster.')
    return dict(zip(frames,zip(yaw,pitch,strict=True),strict=True))


def r22_pose_paths(subject, material):
    root = Path(f'/opt/gs-slam-lab/shuolee-smpl-delivery-round22-{subject}/run')
    provenance = material['provenance']
    fit = pin(root/'fit/fitted.npz',material['fit_sha256'])
    candidates = root/'fit/joint-candidates.npz'
    if not material['admitted_auxiliary_frames']:
        return fit,None
    # This candidate archive is signed by the same R22 auxiliary admission used for the native images and poses.
    pins = list(root.rglob('face-supplement/admission.json'))
    if not pins:
        pins = [path for path in root.rglob('admission.json')
                if read(path).get('chosen_frame_ids')==material['admitted_auxiliary_frames']]
    receipts = [read(path) for path in pins if read(path).get('chosen_frame_ids')==material['admitted_auxiliary_frames']]
    hashes = {receipt['input_sha256']['joint_candidates'] for receipt in receipts}
    need(provenance['fit_sha256']==sha(fit) and len(hashes)==1, 'Auxiliary angle source is not uniquely pinned.')
    return fit,pin(candidates,hashes.pop())


def angular_weights(details):
    yaw = np.asarray([row['head_yaw_deg'] for row in details],np.float32)
    pitch = np.asarray([row['head_pitch_deg'] for row in details],np.float32)
    yaw_step,pitch_step = POLICY['yaw_sector_deg'],POLICY['pitch_sector_deg']
    yaw_sector = (np.floor(yaw/yaw_step+.5)*yaw_step).astype(np.int16)
    pitch_sector = (np.floor(pitch/pitch_step+.5)*pitch_step).astype(np.int16)
    quality = np.asarray([row['quality'] for row in details],np.float32)
    active = quality>0
    occupied = np.unique(np.column_stack((yaw_sector[active],pitch_sector[active])),axis=0)
    weight = np.zeros_like(quality)
    for y,p in occupied:
        selected = active&(yaw_sector==y)&(pitch_sector==p)
        weight[selected] = quality[selected]/(quality[selected].sum()*len(occupied))
    return dict(head_yaw_deg=yaw,head_pitch_deg=pitch,yaw_sector=yaw_sector,pitch_sector=pitch_sector,
                face_visible=active,quality=quality,face_weight=weight),occupied


def run(subject):
    sys.path[:0] = [str(PROJECT/'src'),str(PROJECT/'src/avatar_core')]
    import torch

    torch.set_num_threads(2)
    legacy_path,material_path,material,_,_,prior = old_inputs(subject)
    legacy = legacy_path.is_file()
    loader = legacy_inputs if legacy else r22_inputs
    model,_,asset,_,records = loader(legacy_path if legacy else subject,material,prior)
    faces = model.faces.numpy()
    head = protected_faces(asset,faces)
    if legacy:
        source = read(legacy_path)
        fit,aux = checked(source['fit']),None
    else:
        fit,aux = r22_pose_paths(subject,material)
    fitted = directions(fit)
    auxiliary = directions(aux) if aux else {}
    train_frames = material['selected_frames']
    aux_frames = material['admitted_auxiliary_frames']
    reserved = material.get('reserved_holdouts',material['retained_holdouts'])
    need(set(train_frames+aux_frames).isdisjoint(reserved), 'Evaluation frames cannot supervise R30.')
    coverage_path = R29/f'{subject}-native-coverage.json'
    coverage = read(coverage_path)
    need(coverage['training_frames']==train_frames and coverage['auxiliary_frames']==aux_frames
         and coverage['reserved_holdouts']==reserved and coverage['heldout_source_images_read']==0,
         'R29 archived native evidence differs from R30 training roles.')
    pin(coverage_path.with_suffix('.npz'),coverage['archive_sha256'])
    arrays, details, frames, slots, boxes = {}, [], [], [], []
    for i,record in enumerate(records):
        row = record.row
        frame = row['frame']
        need(frame==(train_frames+aux_frames)[i] and frame not in reserved and record.slot==prior['source_slots'][i],
             'R30 source roster changed.')
        yaw,pitch = (fitted if frame in train_frames else auxiliary)[frame]
        face,skin,details_row = native_face_regions(record,faces,head,yaw,pitch)
        need(not np.any(skin&~face), 'Skin mask escaped the verified native face.')
        arrays[f'face_{i:03d}'],arrays[f'skin_{i:03d}'] = face,skin
        details.append(dict(details_row,frame=frame,slot=record.slot,box=list(record.box),
            role='main' if frame in train_frames else 'face_only',source_sha256=row['source_sha256'],
            mask_sha256=row['mask_sha256'],K=row['K'],w2c=row['w2c'],
            face_mask_sha256=array_sha(face),skin_mask_sha256=array_sha(skin)))
        frames.append(frame)
        slots.append(record.slot)
        boxes.append(record.box)
    angular,occupied = angular_weights(details)
    active = angular['face_visible']
    need(len(details)==len(train_frames)+len(aux_frames) and active.any()
         and np.isclose(angular['face_weight'].sum(),1.), 'No verified native face supervision.')
    for i,row in enumerate(details):
        row.update(yaw_sector=int(angular['yaw_sector'][i]),pitch_sector=int(angular['pitch_sector'][i]),
                   face_visible=bool(active[i]),face_weight=float(angular['face_weight'][i]))
    OUT.mkdir(parents=True,exist_ok=True)
    output = OUT/f'{subject}-face-supervision.npz'
    receipt_path = output.with_suffix('.json')
    need(not output.exists() and not receipt_path.exists(), 'Preserve existing signed R30 supervision.')
    with output.open('xb') as stream:
        np.savez_compressed(stream,source_slots=np.asarray(slots,np.int64),frame_ids=np.asarray(frames),
                            boxes=np.asarray(boxes,np.int32),**angular,**arrays)
    occupied_yaw,occupied_pitch = (np.unique(occupied[:,i]).tolist() for i in range(2))
    receipt = dict(schema='shuolee-r30-native-face-supervision/1',subject=subject,archive_sha256=sha(output),
        coverage_receipt_sha256=sha(coverage_path),baseline_bound_ply_sha256=material['baseline_bound_ply_sha256'],
        material_receipt_sha256=sha(material_path),fit=descriptor(fit),auxiliary_fit=descriptor(aux) if aux else None,
        heldout_source_images_read=0,training_frames=train_frames,auxiliary_frames=aux_frames,reserved_holdouts=reserved,
        views=details,policy=POLICY,source_sha256=sha(__file__),visible_face_frames=int(active.sum()),
        occupied_yaw_pitch_sectors=occupied.tolist(),missing_yaw_sectors=sorted(set(range(-60,61,20))-set(occupied_yaw)),
        missing_pitch_sectors=sorted(set(range(-30,31,15))-set(occupied_pitch)))
    write_json(receipt_path,receipt)
    print(dict(subject=subject,verified_faces=int(active.sum()),total=len(active),
               sectors=occupied.tolist(),receipt=descriptor(receipt_path)),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject',choices=SUBJECTS)
    run(parser.parse_args().subject)
