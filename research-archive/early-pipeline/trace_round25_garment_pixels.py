"""Read-only garment versus frozen-layer attribution on signed fixed views."""

import argparse
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
import sys
from unittest.mock import patch

import json
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import torch

from trace_round22_face_pixels import (LUMA, checked, code, fields, geometry, need, numbered,
                                       pin, pixel, ply, preprocess, read, sha)


OUT = Path(__file__).resolve().parents[1]/'outputs'
PROJECT = '/mnt/d/Projects/Shuolee-smpl-clothing-surface-r'
R24_REVIEW = 'd34eff119bba53fe8ea08d45d5e16cdfa1c4271f5bfb62a4b67cf97f3b85d4e9'
PLUS = 'normal_yaw_p90_pitch_p00'
MINUS = 'normal_yaw_m90_pitch_p00'

# Each artifact/control pair uses one same-row comparison on a saved 18k crop.
# Chosen on the R24 board before inspecting any layer attribution.
PICKS = {
    '016': {
        PLUS: [('shoulder_back', (153,151), (138,151)), ('waist', (205,263), (181,263)),
               ('shirt_hem', (236,350), (215,350)), ('lower_hem', (263,550), (245,550))],
        MINUS: [('shoulder_back', (177,151), (194,151)), ('waist', (103,302), (126,302)),
                ('lower_hem', (76,552), (106,552))],
    },
    '006': {
        PLUS: [('shoulder_back', (264,231), (242,231)), ('waist_seat', (312,286), (279,286)),
               ('coat_hem', (289,329), (265,329)), ('pants_cuff', (119,469), (106,469))],
        MINUS: [('shoulder_back', (32,239), (56,239)), ('waist', (111,302), (130,302)),
                ('pants_cuff', (259,470), (241,470))],
    },
}


def loaded_drive(path):
    from shuolee_portrait.drivable_clothing_bundle import load_drive_bundle

    original = torch.tensor

    def cpu_tensor(*args, **kwargs):
        if kwargs.get('device') == 'cuda':
            kwargs['device'] = 'cpu'
        return original(*args, **kwargs)

    with patch.object(torch, 'tensor', side_effect=cpu_tensor):
        return load_drive_bundle(path, device='cpu')


def np_tensor(value):
    return value.detach().cpu().numpy()


def world_rows(arm, drive_path):
    portable_path = pin(arm['ply'], arm['ply_sha256'])
    raw_path = checked(arm['world'])
    portable, raw = ply(portable_path), ply(raw_path)
    transform = arm['world_to_viewer']
    matrix, origin = np.asarray(transform['rotation']), np.asarray(transform['origin_world'])
    rotation_fields = [f'rot_{i}' for i in range(4)]
    copy_fields = set(raw.dtype.names)-{'x','y','z','nx','ny','nz',*rotation_fields}
    need(portable.dtype.names == raw.dtype.names and len(portable)==len(raw) and transform['scale']==1.
         and all(np.array_equal(raw[k],portable[k]) for k in copy_fields)
         and np.array_equal(((fields(raw,'xyz')-origin)@matrix.T).astype(np.float32),fields(portable,'xyz')),
         'Portable PLY does not preserve the raw row ordering.')
    q = numbered(raw,'rot',4).astype(np.float64)
    rotated = Rotation.from_matrix(matrix@Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()).as_quat()[:,[3,0,1,2]]
    need(np.array_equal(rotated.astype(np.float32), numbered(portable,'rot',4)), 'Portable quaternion rows differ.')
    scene, info = loaded_drive(drive_path.parent)
    scene.select_mesh_by_timestep(info['reference_index'])
    frozen = info['fixed_points']
    opacity = torch.cat((scene.fixed._opacity,scene.cloth._opacity)).detach().numpy()[:,0]
    need(frozen+info['clothing_points']==len(raw) and np.array_equal(opacity,raw['opacity'])
         and np.array_equal(np_tensor(scene.get_features)[:,0,:],numbered(raw,'f_dc',3)),
         'The fixed-first drive rows do not match raw color or opacity.')
    delta = max(float(np.max(abs(np_tensor(scene.get_xyz)-fields(raw,'xyz')))),
                float(np.max(abs(np_tensor(scene.get_scaling.log())-numbered(raw,'scale',3)))))
    posed_q = np_tensor(scene.get_rotation)
    q_error = np.minimum(abs(posed_q-q).max(1),abs(posed_q+q).max(1)).max()
    need(delta<2e-5 and q_error<2e-5, 'Reference drive geometry does not identify the world PLY rows.')
    return dict(portable=portable, scene=scene, drive=info,
                proof=dict(portable_sha256=sha(portable_path), raw_sha256=sha(raw_path), drive_sha256=sha(drive_path),
                           max_xyz_or_logscale_abs=delta, max_quaternion_abs=float(q_error)))


def row_identity(context, plan, arm_name):
    scene = context['scene']
    source_config = read(checked(plan['training_config']))
    baseline = Path(source_config['baseline_run'])
    archive = baseline/'gaussian-run/raw-step-018000'
    state = read(archive/'state.json')
    state_path = pin(archive/'state.pt', state['state_sha256'])
    stock = torch.load(state_path, weights_only=True, map_location='cpu')['state']
    guide = pin(baseline/'gaussian-run/shared-clothing-support.npz',source_config['source_guide_sha256'])
    with np.load(guide,allow_pickle=False) as data:
        eligible = data['eligible_vertices']
    owners = np_tensor(stock['model']['binding']['tensor']).astype(np.int64)
    faces = np_tensor(scene.fixed.faces)
    keep = ~eligible[faces].all(1)[owners]
    need(np.array_equal(owners[keep],np_tensor(scene.fixed.binding)) and all(torch.equal(
         stock['parameters'][name]['value']['tensor'][keep],getattr(scene.fixed,name).detach())
         for name in ('_xyz','_rotation','_scaling','_opacity','_features_dc','_features_rest')),
         'Frozen rows differ from the pinned original training state.')
    fixed_ids = np_tensor(stock['model']['densification_ids']['tensor'])[keep]
    garment_ids = np_tensor(scene.cloth.densification_ids)
    need(np.intersect1d(fixed_ids,garment_ids).size==0, 'Gaussian identities overlap between layers.')
    all_owners = np.r_[owners[keep],np_tensor(scene.cloth.support_face_ids)]
    asset = Path(plan['asset'])
    bound = read(baseline/'gaussian-run/step-018000/bound-replay.json')
    pin(asset,bound['asset_sha256'])
    with np.load(asset,allow_pickle=False) as body:
        weights = body['weights']
    dominant = weights[faces].mean(axis=1).argmax(axis=1)[owners[keep]]
    parts = np.full(len(fixed_ids),'frozen_body',dtype='U24')
    parts[np.isin(dominant,[12,15,22,23,24])] = 'frozen_head_neck'
    parts[np.isin(dominant,[20,21,*range(25,55)])] = 'frozen_wrist_hand'
    parts[np.isin(dominant,[7,8,10,11])] = 'frozen_ankle_foot'
    trust = getattr(scene.cloth,'anchor_trusted',None)
    trust = np.r_[np.zeros(len(fixed_ids),bool),np_tensor(trust)] if trust is not None else None
    context.update(stable=np.r_[fixed_ids,garment_ids],owner=all_owners,
        part=np.r_[parts,np.full(len(garment_ids),'garment',dtype='U24')], trust=trust)
    context['proof'].update(original_state_sha256=sha(state_path), protected_part_asset_sha256=sha(asset))


def samples(camera, panel, arrays, probes):
    pre = preprocess(arrays,camera)
    x0,y0,x1,y1=panel['crop_xyxy']
    w,h=x1-x0,y1-y0
    need(panel['columns']==['black','white','alpha'], 'Expected a fixed normal-distance panel.')
    with Image.open(checked(panel)) as image:
        observed = np.asarray(image.convert('RGB'),dtype=np.int16)
    need(observed.shape==(h,w*3,3), 'The signed camera crop does not match its panel.')
    coords = sorted({(x+dx,y+dy) for _,a,b in probes for x,y in (a,b) for dx in (-1,0,1) for dy in (-1,0,1)})
    recovered={}
    for x,y in coords:
        value=pixel(pre,(x+x0,y+y0))
        rendered=np.stack((code(value['rgb']),code(value['rgb']+value['transmittance']),
                           np.full(3,code(value['opacity']),dtype=np.int16)))
        qa=observed[y,x+np.arange(3)*w]
        recovered[x,y]=value | dict(max_code_error=int(abs(rendered-qa).max()),qa_black=qa[0].tolist(),qa_alpha=int(qa[2,0]))
    return pre,recovered


def attribute(context, pre, recovered, crop, probe):
    region, a, b=probe
    neighborhood = [recovered[x+dx,y+dy] for x,y in (a,b) for dx in (-1,0,1) for dy in (-1,0,1)]
    worst=max(item['max_code_error'] for item in neighborhood)
    result=dict(region=region,artifact_xy_px=[a[0]+crop[0],a[1]+crop[1]],control_xy_px=[b[0]+crop[0],b[1]+crop[1]],
                parity=dict(pixels=len(neighborhood),exact=sum(item['max_code_error']==0 for item in neighborhood),
                            max_code_error=worst,passed=worst<=1))
    if worst>1:
        return result
    left,right=recovered[a],recovered[b]
    control=float(right['rgb']@LUMA)
    groups=defaultdict(lambda:np.zeros(3))
    for row,weight in zip(right['rows'],right['weights'],strict=True):
        groups[context['part'][row]][1]+=weight
    contributors=[]
    for row,weight in zip(left['rows'],left['weights'],strict=True):
        row=int(row)
        part=context['part'][row].item()
        direct=float(weight)*(float(pre['rgb'][row]@LUMA)-control)
        groups[part] += (weight,0,direct)
        eig=np.linalg.eigvalsh(pre['projected'][row]+.3*np.eye(2))
        contributors.append(dict(row=row,stable_id=int(context['stable'][row]),face=int(context['owner'][row]),
            part=part,alpha_weight=float(weight),direct_luma=direct,sigma_ratio=float(np.sqrt(eig[1]/eig[0])),
            projected_xy_px=pre['xy'][row].tolist(),footprint_radius_px=int(pre['radius'][row]),
            r25_geometry_anchor_trusted=bool(context['trust'][row]) if context['trust'] is not None and part=='garment' else None))
    delta=float((left['rgb']-right['rgb'])@LUMA)
    background=-left['transmittance']*control
    need(abs(sum(row[2] for row in groups.values())+background-delta)<3e-6, 'Per-row contrast did not reconstruct the output.')
    contributors.sort(key=lambda row:abs(row['direct_luma']),reverse=True)
    return result | dict(artifact_rgb=left['qa_black'],control_rgb=right['qa_black'],
        artifact_alpha=left['opacity'],control_alpha=right['opacity'],delta_luma=delta,background_luma=background,
        groups={name:dict(zip(('artifact_alpha','control_alpha','direct_luma'),row.tolist())) for name,row in groups.items()},
        contributors=contributors)


def run(round_number, review_sha):
    arm_name=f'r{round_number}'
    sys.path.insert(0, PROJECT+str(round_number)+'/src')
    torch.set_num_threads(2)
    directory=OUT/f'smpl-delivery-round{round_number}'
    index=read(pin(directory/'review-receipt.json',review_sha))
    output={}
    for subject,views in PICKS.items():
        result=read(checked(index['subjects'][subject]['review']))
        plan=read(checked(result['plan']))
        arm=plan['static_grid']['arms'][arm_name]
        context=world_rows(arm,checked(plan['graphs'][arm_name]['manifest']))
        row_identity(context,plan,arm_name)
        arrays=geometry(context['portable'])
        subject_result=dict(input_proof=context['proof'],views={})
        grid=read(checked(result['static'][arm_name]['receipt']))
        need(grid['ply_sha256']==arm['ply_sha256'], 'The fixed rendered grid did not use the traced PLY.')
        for key,probes in views.items():
            row=grid['records'][key]
            if round_number==25:
                r24_index=read(pin(OUT/'smpl-delivery-round24/review-receipt.json',R24_REVIEW))
                r24=read(checked(r24_index['subjects'][subject]['review']))
                baseline=r24['static']['r24']['records'][key]
                need(row['fixed_world_camera']==baseline['fixed_world_camera'] and
                     row['panel']['crop_xyxy']==baseline['panel']['crop_xyxy'], 'R25 camera or crop changed from R24.')
            panel=row['panel']
            pre,results=samples(row['camera'],panel,arrays,probes)
            comparison=[attribute(context,pre,results,panel['crop_xyxy'],probe) for probe in probes]
            subject_result['views'][key]=dict(camera=row['camera'],panel=panel,pairs=comparison)
            compact=[dict(region=p['region'],parity=p['parity'],alpha=p.get('artifact_alpha'),luma=p.get('delta_luma'),
                          groups=p.get('groups')) for p in comparison]
            print(json.dumps(dict(subject=subject,view=key,pairs=compact),ensure_ascii=False),flush=True)
        output[subject]=subject_result
    target=directory/'input-audit/side-waist-hem-attribution.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x',encoding='utf-8') as stream:
        json.dump(dict(schema='shuolee-garment-side-pixel-attribution/1', round=round_number, review_sha256=review_sha,
                       source_sha256=sha(Path(__file__)),gpu_used=False,production_changes=False,subjects=output,
                       limitations=['Fixed camera side views have no true source image at the same pose and view.',
                                    'Frozen parts are assigned from dominant binding-face skinning weights. Their names do not identify physical material.',
                                    'A real render contribution can justify targeted observation review. Reassigning material requires independent native support.']),
                  stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(dict(output=str(target),sha256=sha(target)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('round',type=int,choices=(24,25))
    parser.add_argument('--review-sha')
    args=parser.parse_args()
    if args.round==25 and args.review_sha is None:
        parser.error('R25 needs an exact signed review SHA')
    run(args.round,args.review_sha or R24_REVIEW)
