"""Recheck the same R22 face artifact rays against R26 and R27 endpoints."""

import argparse
import json
import sys

import numpy as np
import torch

from build_round27_material_visibility import OUT
from compare_round26_fixed_face_rays import LAB, LUMA, describe_ray, descriptor, load_endpoint, world_pose_in_viewer
from trace_round22_face_pixels import (DELIVERY, PICKS, PROJECT, REVIEW_SHA, VIEW, checked, code, load_subject,
                                       need, parity_panel, pin, pixel, preprocess, read, sha)


ROLES=('observed','geometry_eligible','silhouette_spill','unknown_unobserved_elongated','possible_protected_material')


def endpoint_view(model, world, crop, pairs, archived=None):
    camera=world_pose_in_viewer(world,model['transform'])
    pre=preprocess(model['arrays'],camera)
    proof=None
    if archived is not None:
        need(archived['fixed_world_camera']==world and archived['panel']['crop_xyxy']==crop and
             all(np.allclose(archived['camera'][key],camera[key],rtol=0,atol=1e-12) for key in ('wh','K','w2c')),
             'Independent review changed a fixed marked ray.')
        proof,trace=parity_panel(pre,checked(archived['panel']),crop,pairs)
        need(proof['passed'],'CPU reconstruction of an independent CUDA panel differed by more than one code.')
    else:
        x0,y0,*_=crop
        trace={xy:pixel(pre,(x0+xy[0],y0+xy[1])) for pair in pairs for xy in pair}
    return proof,trace


def summarize_pair(original, candidate, old_trace, new_trace, points, roles):
    artifact,control=(describe_ray(original,candidate,old_trace[xy],new_trace[xy]) for xy in points)
    for target in (artifact,control):
        for row in target['largest_signed_rgb_row_changes']:
            stable=row['stable_id']
            if stable<0:
                # Children allocated after the shared baseline can reuse an
                # integer across experiments without being the same Gaussian.
                row.update(stable_id=-stable-1,allocation_branch='r26',r27_frozen_source_roles=None)
            else:
                row['r27_frozen_source_roles']=roles.get(stable)
    contrast={phase:float(LUMA@(np.asarray(artifact[phase]['black'])-np.asarray(control[phase]['black'])))
              for phase in ('before','after')}
    return dict(artifact=dict(xy=points[0],**artifact),control=dict(xy=points[1],**control),
                artifact_minus_control_luma_codes=contrast)


def run(subject, step, verify_qa):
    sys.path[:0]=[str(PROJECT/'src'),str(PROJECT/'src/avatar_core')]
    torch.set_num_threads(2)
    base=load_subject(subject,read(pin(DELIVERY/'review-receipt.json',REVIEW_SHA)))
    base['lookup']={int(value):i for i,value in enumerate(base['stable'])}
    evidence_path=OUT/f'{subject}-material-visibility.json'
    evidence=read(evidence_path)
    archive=pin(evidence_path.with_suffix('.npz'),evidence['material_visibility_npz_sha256'])
    with np.load(archive,allow_pickle=False) as data:
        need(np.array_equal(data['stable_ids'],base['stable']) and np.array_equal(data['owner_faces'],base['owner']),
             'R27 source roles lost baseline row identity.')
        flags=np.column_stack([data[name] for name in ROLES])
    root=LAB/f'shuolee-smpl-delivery-round27-{subject}'
    side='r27_10' if step==10000 else 'r27'
    roster=base['plan']['static_grid']['camera_roster']
    reference=base['plan']['poses']['r22']['new']['reference_frame_id']
    models={'r26':load_endpoint(LAB/f'shuolee-smpl-delivery-round26-{subject}',18000,reference),
            side:load_endpoint(root,step,reference)}
    historical=models['r26']
    stable=historical['stable']
    lineage=np.where(np.isin(stable,base['stable']),stable,-stable-1)
    historical=dict(historical,stable=lineage,lookup={int(stable):i for i,stable in enumerate(lineage)})
    qa,qa_proof=None,None
    if verify_qa:
        review_path=root/'view-comparison/result.json'
        review=read(review_path)
        plan_path=checked(review['plan']); plan=read(plan_path)
        need(plan['static_grid']['camera_roster']==roster and plan_path==review_path.with_name('plan.json'),
             'The signed R27 review did not use the pinned camera roster.')
        qa={}
        qa_proof=dict(result=descriptor(review_path),plan=descriptor(plan_path),static={})
        for name,model in models.items():
            arm=plan['static_grid']['arms'][name]
            static_path=checked(review['static'][name]['receipt']); static=read(static_path)
            need(arm['world_to_viewer']==model['transform'] and arm['ply_sha256']==static['ply_sha256']==
                 model['hashes']['portable_ply']['sha256'] and static['records']==review['static'][name]['records'] and
                 static['view_count']==len(roster), 'The signed static result differs from the exact endpoint package.')
            qa[name]=static['records']; qa_proof['static'][name]=descriptor(static_path)
    views={}
    for suffix,(description,pairs) in PICKS[subject].items():
        key=VIEW+suffix
        world=roster[key]['world_camera']; crop=roster[key]['head_crop_xyxy']
        baseline=base['grid']['records'][key]
        base_proof,base_trace=endpoint_view(dict(base,transform=base['arm']['world_to_viewer']),world,crop,pairs,baseline)
        results={'r22':(base_proof,base_trace)}
        for name,model in models.items():
            results[name]=endpoint_view(model,world,crop,pairs,qa[name][key] if qa is not None else None)
        x0,y0,*_=crop
        raw=preprocess(models[side]['raw_arrays'],world)
        world_export_max=0
        material_roles={}
        for xy_pair in pairs:
            for xy in xy_pair:
                item=results[side][1][xy]
                native=pixel(raw,(x0+xy[0],y0+xy[1]))
                world_export_max=max(world_export_max,int(np.abs(code(item['rgb'])-code(native['rgb'])).max()),
                                     int(np.abs(code(item['opacity'])-code(native['opacity']))))
        need(world_export_max<=1,'The same marked R27 rays changed in portable export.')
        for name,model in [('r22',base),*models.items()]:
            for xy_pair in pairs:
                for xy in xy_pair:
                    for row in results[name][1][xy]['rows']:
                        stable=int(model['stable'][row]); pos=base['lookup'].get(stable)
                        if pos is not None and stable not in material_roles:
                            material_roles[stable]=[key for key,active in zip(ROLES,flags[pos],strict=True) if active]
        pairs_by_baseline={name:[summarize_pair(model,models[side],results[name][1],results[side][1],pair,material_roles)
                                  for pair in pairs] for name,model in [('r22',base),('r26',historical)]}
        views[key]=dict(description=description,source_world_camera=world,crop=crop,
            candidate_raw_to_portable_max_code=world_export_max,
            parity_by_arm={name:result[0] for name,result in results.items()},pairs_by_baseline=pairs_by_baseline)
    result=dict(schema='shuolee-r27-paired-fixed-marked-face-rays/1',subject=subject,step=step,side=side,
        source_sha256=sha(__file__),verified_against_gpu_static=verify_qa,qa=qa_proof,
        baseline_r22=base['hashes'],endpoint_r26=models['r26']['hashes'],endpoint_r27=models[side]['hashes'],
        source_material=descriptor(evidence_path),views=views,
        scope='Identical historical synthetic face rays and nearby controls. There is no real RGB target for these camera rays. '
              'Color differences quantify model output and stable row contributions; they do not verify a physical correction.')
    directory=OUT.parent/'ray-comparison'; directory.mkdir(parents=True,exist_ok=True)
    path=directory/f'{subject}-{step}-{"signed" if verify_qa else "cpu"}.json'
    with path.open('x',encoding='utf-8') as file:
        json.dump(result,file,indent=2,ensure_ascii=False,allow_nan=False)
        file.write('\n')
    print(dict(file=str(path),sha256=sha(path),verified_against_gpu_static=verify_qa),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject',choices=('016','006'))
    parser.add_argument('step',type=int,choices=(10000,18000))
    parser.add_argument('--verify-qa',action='store_true')
    args=parser.parse_args()
    run(args.subject,args.step,args.verify_qa)
