"""Recheck R22 marked face rays on the R26 portable endpoints using stock CPU compositing."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from build_round26_native_face_material import LABELS, OUT
from trace_round22_face_pixels import (DELIVERY, PICKS, PROJECT, REVIEW_SHA, VIEW, checked, code,
                                       fields, geometry, load_subject, need, numbered, parity_panel,
                                       pin, pixel, ply, preprocess, read, sha)


LAB = Path('/opt/gs-slam-lab')
REPORTS = OUT.parent/'ray-comparison'
LUMA = np.asarray([.2126,.7152,.0722])
PARAMETERS = dict(xyz=list('xyz'), rotation=[f'rot_{i}' for i in range(4)],
                  scale=[f'scale_{i}' for i in range(3)], opacity=['opacity'],
                  color=[f'f_dc_{i}' for i in range(3)])


def descriptor(path):
    return dict(file=str(path),sha256=sha(path))


def world_pose_in_viewer(camera, transform):
    from shuolee_portrait.head_support_views import to_portable

    orientation = np.asarray(transform['rotation'])
    need(transform['scale']==1. and np.allclose(orientation@orientation.T,np.eye(3),rtol=0,atol=1e-12)
         and np.linalg.det(orientation)>.999999, 'The portable coordinate mapping is not a rigid transform.')
    return dict(camera,w2c=to_portable(camera['w2c'],transform).tolist())


def load_endpoint(root, step, reference_frame):
    run=root/'run'/'gaussian-run'
    latest=read(run/'result.json')
    endpoint=next(item for item in latest['endpoints'] if item['step']==step)
    snapshot=run/endpoint['path']
    receipt=read(snapshot/'result.json')
    need(receipt==endpoint and snapshot.name==f'raw-step-{step:06d}', 'The terminal receipt differs from its checkpoint.')
    state_path=pin(snapshot/'main-state.pt',receipt['state_sha256'])
    portable_path=snapshot/'portable-result.json'
    portable=read(portable_path)
    source=pin(snapshot/'world.ply',portable['source_world_sha256'])
    portable_ply=pin(snapshot/portable['ply'],portable['sha256'])
    camera=pin(snapshot/portable['camera'],portable['camera_sha256'])
    provenance=camera.with_name('camera_provenance.json')
    proof=read(provenance)
    transform=proof['world_to_viewer']
    parity=read(snapshot/'native-ply-parity.json')
    need(portable['state']=='completed' and portable['step']==step and portable['format_passed']
         and parity['passed'] and parity['sha256']==portable['source_world_sha256']
         and parity['reference_frame_id']==reference_frame and proof['cameras_share_model_world']
         and proof['files_sha256']['cameras.json']==portable['camera_sha256'], 'The portable model is not the pinned R26 reference pose.')
    drive_path=snapshot/'drive/drive.json'; drive=read(drive_path)
    motion=pin(drive_path.parent/'motion.npz',drive['files']['motion.npz'])
    local_path=pin(drive_path.parent/'bound-local.ply',drive['files']['bound-local.ply'])
    with np.load(motion,allow_pickle=False) as archive:
        stable, labels, spill=(archive[name] for name in ('stable_ids','labels','spill'))
        need(archive['frame_ids'][drive['reference_pose_index']].item()==reference_frame,
             'The bound package reference pose differs.')
    local, world, view=map(ply,(local_path,source,portable_ply))
    need(len(stable)==len(labels)==len(spill)==len(local)==len(world)==len(view)==receipt['points']
         and len(np.unique(stable))==len(stable) and world.dtype.names==view.dtype.names
         and drive['schema']=='shuolee-face-bound-v1' and drive['sh_degree']==0,
         'The endpoint row identities or SH0 format differ.')
    for name in (*PARAMETERS['opacity'],*PARAMETERS['color']):
        need(np.array_equal(local[name],world[name]) and np.array_equal(world[name],view[name]),
             'The export changed point color or opacity, or lost row identity.')
    for name in set(world.dtype.names)-set('xyz')-set('n'+axis for axis in 'xyz')-set(PARAMETERS['rotation']):
        need(np.array_equal(world[name],view[name]), 'World PLY portable appearance or scale differs.')
    R, origin = np.asarray(transform['rotation']), np.asarray(transform['origin_world'])
    np.testing.assert_array_equal(((fields(world,'xyz')-origin)@R.T).astype(np.float32),fields(view,'xyz'))
    q=numbered(world,'rot',4).astype(np.float64)
    expected=Rotation.from_matrix(R@Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()).as_quat()[:,[3,0,1,2]]
    np.testing.assert_array_equal(expected.astype(np.float32),numbered(view,'rot',4))
    need(np.array_equal(local['binding_0'],np.rint(local['binding_0'])), 'The local PLY lost integer owner triangles.')
    return dict(arrays=geometry(view), raw_arrays=geometry(world), stable=stable, local=local,
                labels=labels, spill=spill, transform=transform,
                hashes=dict(endpoint=descriptor(snapshot/'result.json'),portable=descriptor(portable_path),
                    source_world=descriptor(source),portable_ply=descriptor(portable_ply),provenance=descriptor(provenance),
                    checkpoint=dict(file=str(state_path),sha256=receipt['state_sha256']),
                    drive=descriptor(drive_path),motion=descriptor(motion),local=descriptor(local_path)),
                lookup={int(value):i for i,value in enumerate(stable)})


def describe_ray(old, new, before, after):
    def by_id(endpoint, item):
        return {int(endpoint['stable'][row]):(int(row),float(alpha)) for row,alpha in zip(item['rows'],item['weights'])}
    left,right=by_id(old,before),by_id(new,after)
    records=[]
    total_delta=np.zeros(3)
    categories={name:np.zeros(3) for name in ('edited_parameters','insertion_or_removal','frozen_parameter_rows_weight_change')}
    for stable in sorted(left.keys()|right.keys()):
        _,w0=left.get(stable,(None,0.)); _,w1=right.get(stable,(None,0.))
        i,j=old['lookup'].get(stable),new['lookup'].get(stable)
        rgb0=old['arrays'][4][i].astype(np.float64) if i is not None else np.zeros(3)
        rgb1=new['arrays'][4][j].astype(np.float64) if j is not None else np.zeros(3)
        delta=w1*rgb1-w0*rgb0
        changed=[]
        if i is not None and j is not None:
            for name,names in PARAMETERS.items():
                if any(old['local'][field][i]!=new['local'][field][j] for field in names):
                    changed.append(name)
        group=('insertion_or_removal' if i is None or j is None else
               'edited_parameters' if changed else 'frozen_parameter_rows_weight_change')
        categories[group]+=delta
        total_delta+=delta
        records.append(dict(stable_id=stable,alpha_before=w0,alpha_after=w1,delta_rgb=delta.tolist(),
            category=group,changed_parameters=changed,
            owner_face_before=int(old['local']['binding_0'][i]) if i is not None else None,
            owner_face_after=int(new['local']['binding_0'][j]) if j is not None else None,
            label_after=LABELS[new['labels'][j]] if j is not None else None,
            silhouette_spill_after=bool(new['spill'][j]) if j is not None else None))
    np.testing.assert_allclose(total_delta,after['rgb']-before['rgb'],rtol=0,atol=4e-6)
    records.sort(key=lambda item:-sum(abs(value) for value in item['delta_rgb']))
    levels=lambda item:dict(black=code(item['rgb']).tolist(),white=code(item['rgb']+item['transmittance']).tolist(),
                            alpha=int(code(item['opacity'])),black_rgb_float01=item['rgb'].tolist(),alpha_float01=item['opacity'])
    return dict(before=levels(before),after=levels(after),delta_black_codes=(code(after['rgb'])-code(before['rgb'])).tolist(),
                delta_alpha=after['opacity']-before['opacity'],signed_float01_rgb_by_row_class={k:v.tolist() for k,v in categories.items()},
                contributing_rows_before=len(left),contributing_rows_after=len(right),largest_signed_rgb_row_changes=records[:10])


def run(subject, step, verify_qa):
    sys.path[:0]=[str(PROJECT/'src'),str(PROJECT/'src/avatar_core')]
    torch.set_num_threads(2)
    root=LAB/f'shuolee-smpl-delivery-round26-{subject}'
    context=load_subject(subject,read(pin(DELIVERY/'review-receipt.json',REVIEW_SHA)))
    context['lookup']={int(value):i for i,value in enumerate(context['stable'])}
    baseline_grid=context['plan']['static_grid']['camera_roster']
    reference=context['plan']['poses']['r22']['new']['reference_frame_id']
    candidate=load_endpoint(root,step,reference)
    side='r26_10' if step==10000 else 'r26'
    review_proof=None
    if verify_qa:
        review=root/'view-comparison'
        result_path=review/'result.json'
        result=read(result_path)
        plan_path=checked(result['plan'])
        qa_path=checked(result['static'][side]['receipt'])
        need(plan_path==review/'plan.json' and qa_path==review/'static-grid'/side/'result.json',
             'The independent receipt does not select the formal R26 review outputs.')
        plan,qa=read(plan_path),read(qa_path)
        arm=plan['static_grid']['arms'][side]
        need(plan['static_grid']['camera_roster']==baseline_grid and arm['world_to_viewer']==candidate['transform']
             and arm['ply_sha256']==qa['ply_sha256']==candidate['hashes']['portable_ply']['sha256']
             and qa['records']==result['static'][side]['records']
             and qa['side']==side and qa['view_count']==50, 'The independent replay does not bind to the exact 50-camera endpoint.')
        review_proof=dict(result=descriptor(result_path),plan=descriptor(plan_path),static=descriptor(qa_path))
    output={}
    for suffix,(description,pairs) in PICKS[subject].items():
        key=VIEW+suffix
        old=context['grid']['records'][key]
        panel=old['panel']; crop=panel['crop_xyxy']
        world=baseline_grid[key]['world_camera']
        camera=world_pose_in_viewer(world,candidate['transform'])
        baseline_camera=world_pose_in_viewer(world,context['arm']['world_to_viewer'])
        need(old['fixed_world_camera']==world and crop==baseline_grid[key]['head_crop_xyxy'] and
             all(np.allclose(old['camera'][name],baseline_camera[name],rtol=0,atol=1e-12)
                 for name in ('wh','K','w2c')),
             'The marked ray anchor changed its camera or head crop.')
        parity,trace=parity_panel(preprocess(context['arrays'],old['camera']),checked(panel),crop,pairs)
        need(parity['passed'],'The archived R22 ray proof no longer matches.')
        pre=preprocess(candidate['arrays'],camera)
        raw=preprocess(candidate['raw_arrays'],world)
        qa_proof=None
        if verify_qa:
            item=qa['records'][key]
            need(item['fixed_world_camera']==world and item['panel']['crop_xyxy']==crop and
                 all(np.allclose(item['camera'][name],camera[name],rtol=0,atol=1e-12) for name in ('wh','K','w2c')),
                 'A candidate QA ray camera or crop differs from the old ray.')
            qa_proof,new_trace=parity_panel(pre,checked(item['panel']),crop,pairs)
            need(qa_proof['passed'],'The CPU result does not match actual independent GPU ray PNG.')
        else:
            x0,y0,*_=crop
            new_trace={point:pixel(pre,(point[0]+x0,point[1]+y0)) for pair in pairs for point in pair}
        rows=[]
        x0,y0,*_=crop
        transform_error=0
        for a,b in pairs:
            for x,y in (a,b):
                native=pixel(raw,(x+x0,y+y0))
                portable=new_trace[(x,y)]
                error=max(abs(code(native['rgb'])-code(portable['rgb'])).max(),
                          abs(code(native['opacity'])-code(portable['opacity'])))
                transform_error=max(transform_error,int(error))
            artifact=describe_ray(context,candidate,trace[a],new_trace[a])
            control=describe_ray(context,candidate,trace[b],new_trace[b])
            contrast={side_:float(LUMA@(np.asarray(artifact[side_]['black'])-np.asarray(control[side_]['black'])))
                      for side_ in ('before','after')}
            rows.append(dict(artifact=dict(xy=a,**artifact),control=dict(xy=b,**control),
                             artifact_minus_control_luma_codes=contrast))
        need(transform_error<=1,'The same rays change more than one code through portable export.')
        output[key]=dict(description=description,source_world_camera=world,crop=crop,
            baseline_panel=panel,baseline_parity=parity,candidate_parity=qa_proof,
            candidate_panel=qa['records'][key]['panel'] if verify_qa else None,
            raw_to_portable_max_code=transform_error,pairs=rows)
    record=dict(schema='shuolee-r26-paired-fixed-marked-face-rays/1',subject=subject,step=step,
        verified_against_gpu_static=verify_qa,qa=review_proof,source_sha256=sha(__file__),
        endpoint=candidate['hashes'],baseline=context['hashes'],views=output,
        scope='Fixed R22 synthetic novel-camera rays with nearby controls; no RGB ground truth. RGB deltas describe actual compositing and do not establish repair or material ownership.')
    REPORTS.mkdir(parents=True,exist_ok=True)
    path=REPORTS/f'{subject}-{step}-{"signed" if verify_qa else "cpu"}.json'
    with path.open('x',encoding='utf-8') as stream:
        json.dump(record,stream,indent=2,ensure_ascii=False,allow_nan=False)
        stream.write('\n')
    print(dict(file=str(path),sha256=sha(path),verified_against_gpu_static=verify_qa),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject',choices=('016','006'))
    parser.add_argument('step',type=int,choices=(10000,18000))
    parser.add_argument('--verify-qa',action='store_true')
    args=parser.parse_args()
    run(args.subject,args.step,args.verify_qa)
