"""Pin native face visibility for evaluation, independently of candidate renders."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import review_smpl_delivery_round29 as prior
from build_round26_native_face_material import MATERIAL_REVIEW, array_sha, protected_faces
from build_round27_material_visibility import decode, source_grid
from trace_round22_face_pixels import DELIVERY as R22


OUT = prior.DELIVERY.with_name('smpl-delivery-round30')/'evaluation-visibility'
BASELINE_SHA = '50700ff5793262e973e966fbd1ecde5037ec172257018c92c3bb2cbc04bf14fe'
SCHEMA = 'shuolee-r30-evaluation-native-face-visibility/1'
FIELDS = ('observed_native_keypoints','joints_projected_native')
pin,checked,read,sha,require,descriptor = prior.pin,prior.checked,prior.read,prior.sha,prior.require,prior.descriptor


def baseline(subject):
    index = pin(prior.DELIVERY/'review-receipt.json',BASELINE_SHA)
    signed = read(index)['subjects'][subject]
    review = read(checked(signed['review']))
    plan = read(checked(review['plan']))
    require(review['subject']==plan['subject']==subject and review['static_camera_count']==50 and
            review['state']=='paired_native_face_and_static_views_rendered',
            'The previous independent native and fixed-camera results differ.')
    return signed,review,plan


def fitted_inputs(subject,plan):
    frozen = plan['frozen']
    fits = dict(selected_fit=frozen['fit'])
    if plan['legacy_source']:
        source = read(checked(plan['legacy_source']))
        fits['holdout'] = source['heldout_formal_fit']
        metadata = {row['frame']:row for row in source['selected_records']+source['heldout_records']}
        return fits,source['native_observations'],metadata,{},None

    r22_index = pin(R22/'review-receipt.json',prior.base.BASELINE_SHA)
    r22_result = read(checked(read(r22_index)['subjects'][subject]['review']))
    r22_plan = read(checked(r22_result['plan']))
    proof = r22_plan['poses']['r22']['proof']
    fits['holdout'] = proof['head_alignment_holdout_poses']
    admission = read(checked(proof['supplement_admission']))
    run = Path(frozen['fit']['file']).parent.parent
    fits['auxiliary_face'] = descriptor(pin(run/'fit/joint-candidates.npz',
                                            admission['input_sha256']['joint_candidates']))
    metadata = {row['frame']:row for row in read(checked(frozen['selected']))['records']}
    annotation_path = pin(R22/'input-audit/face-pixel-attribution.json',MATERIAL_REVIEW)
    annotations = read(annotation_path)['subjects'][subject]['native_annotation']
    return fits,frozen['observations'],metadata,annotations,descriptor(annotation_path)


def native_row(row,metadata):
    source = {**metadata.get(row['frame'],{}),**row}
    source.update(image=str(checked(row['image'])),source_sha256=row['image']['sha256'],
                  mask=str(checked(row['mask'])),mask_sha256=row['mask']['sha256'])
    return source


def roster_inputs(plan,files,observations):
    with ExitStack() as stack:
        fits = {role:stack.enter_context(np.load(checked(item),allow_pickle=False)) for role,item in files.items()}
        geometry = {role:stack.enter_context(np.load(checked(item),allow_pickle=False))
                    for role,item in plan['frozen']['meshes'].items()}
        obs = stack.enter_context(np.load(checked(observations),allow_pickle=False))
        contour = [obs['joint_names'].tolist().index(f'face-{index}') for index in range(17)]
        frames = obs['frame_ids'].tolist()
        contour_data = [obs[key][:,contour] for key in ('keypoints','scores','flip_disagreement')]
        for role,archive in geometry.items():
            names = archive['frame_ids'].tolist()
            require(names==[row['frame'] for row in plan['frozen']['rows'] if row['role']==role],
                    'Frozen pose rows differ from their source role.')
            fit = fits[role]
            pose_names = fit['frame_ids'].tolist()
            pose_fields = {key:fit[key] for key in FIELDS}
            vertices,faces = archive['vertices_world'],archive['faces']
            require(np.array_equal(faces,fit['faces']), 'The frozen fit and replay mesh topology differ.')
            for row in (row for row in plan['frozen']['rows'] if row['role']==role):
                frame = row['frame']
                slot,detector_slot = pose_names.index(frame),frames.index(frame)
                if 'w2c' in fit.files:
                    require(np.allclose(fit['w2c'][slot],row['w2c'],rtol=0.,atol=1e-6),
                            'The frozen fitted pose used a different original camera.')
                yield row, vertices[names.index(frame)],faces,{key:value[slot] for key,value in pose_fields.items()}, \
                    tuple(value[detector_slot] for value in contour_data)


def evaluate_visibility(subject):
    from build_round30_face_supervision import directions, native_face_regions

    signed,_,plan = baseline(subject)
    fits,observations,metadata,annotations,annotation = fitted_inputs(subject,plan)
    angles = {role:directions(checked(item)) for role,item in fits.items()}
    masks,frames = {},{}
    head_faces = None
    for row,vertices,faces,fit,contour in roster_inputs(plan,fits,observations):
        if head_faces is None:
            head_faces = protected_faces(Path(plan['asset']),faces)
        frame = row['frame']
        box = plan['rois'][frame]['face']['box']
        fields = dict(role=row['role'],box=box,source=row['image'],matte=row['mask'])
        if box is None:
            frames[frame] = dict(fields,state='unscored',reason='no_native_detector_crop')
            continue
        yaw,pitch = angles[row['role']][frame]
        source = native_row(row,metadata)
        observed,projected = (fit[key] for key in FIELDS[-2:])
        annotation_row = annotations.get(frame,{})
        allowed = None
        if row.get('verified_face_mask'):
            x0,y0,x1,y1 = row['verified_box']
            l,t,r,b = box
            require(x0<=l<r<=x1 and y0<=t<b<=y1,'The auxiliary crop escaped its frozen verification.')
            allowed = decode(checked(row['verified_face_mask']),mode='L')[t-y0:b-y0,l-x0:r-x0]>0
        grid,_,_,_ = source_grid(source,observed,projected,contour,box,annotation_row,allowed)
        record = SimpleNamespace(row=source,box=box,vertices=vertices,obs=observed,proj=projected,
                                 grid=grid,rgb=decode(source['image'],box).astype(np.float32)/255.,
                                 allowed=allowed,annotation=annotation_row)
        face,_,detail = native_face_regions(record,faces,head_faces,yaw,pitch)
        face = np.asarray(face,dtype=np.bool_)
        require(face.shape==(box[3]-box[1],box[2]-box[0]), 'Trusted face mask is off its native crop.')
        fields.update(yaw_deg=float(yaw),pitch_deg=float(pitch),evidence=detail)
        if detail['status']!='visible_face':
            require(not face.any(),'An unverified face region retained score pixels.')
            frames[frame] = dict(fields,state='unscored',reason=detail['status'])
            continue
        require(face.any(),'A verified face region has no score pixels.')
        key = f'mask_{frame}'
        masks[key] = face
        frames[frame] = dict(fields,state='scored',mask_key=key,mask_sha256=array_sha(face),pixels=int(face.sum()))
    require(set(frames)=={row['frame'] for row in plan['frozen']['rows']} and
            not set(frames)&set(plan['unavailable_holdout_frames']),
            'The evaluation visibility roster changed.')
    return masks,dict(schema=SCHEMA,subject=subject,usage='evaluation_only',
        previous_review=signed['review'],fitted=fits,observations=observations,annotation=annotation,
        meshes=plan['frozen']['meshes'],frames=frames,unavailable=plan['unavailable_holdout_frames'])


def write(subject):
    masks,report = evaluate_visibility(subject)
    OUT.mkdir(parents=True,exist_ok=True)
    arrays = OUT/f'{subject}.npz'
    stage = arrays.with_suffix('.pending.npz')
    np.savez_compressed(stage,**masks)
    if arrays.is_file():
        pin(arrays,sha(stage))
        stage.unlink()
    else:
        stage.replace(arrays)
    receipt = prior.legacy.save_exact(OUT/f'{subject}.json',dict(report,arrays=descriptor(arrays)))
    print(json.dumps(dict(receipt=receipt,scored=len(masks),native=len(report['frames']))),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject',choices=prior.SUBJECTS)
    write(parser.parse_args().subject)
