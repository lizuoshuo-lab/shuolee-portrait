"""Recheck fixed R28 face rays and 016 hair color against the signed R22 and R27 models."""

import argparse
import json
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch

from build_round26_native_face_material import native_weights
from compare_round26_fixed_face_rays import LAB, LUMA, describe_ray, descriptor, load_endpoint, world_pose_in_viewer
from compare_round27_fixed_face_rays import endpoint_view
from trace_round22_face_pixels import (DELIVERY, PICKS, PROJECT, REVIEW_SHA, VIEW, checked, code, load_subject,
                                       need, pin, pixel, preprocess, read, sha)


OUT=DELIVERY.with_name('smpl-delivery-round28')
R27=DELIVERY.with_name('smpl-delivery-round27')
SWAP_SHA='04dd60b48c1a421ece4a213a6d4d7d6cbd6c64b05b3b10aaf6bb41d5cfd8b7b3'
REFERENCE_SHA='42f610c0b2876301c4c617a2eb89cd28ec04ebf9b2684355bcb930bd0a9f0bb2'
TINT_VIEW=VIEW+'p30'
HAIR_POLYGONS={
    'green_patch':[(249,104),(277,110),(296,134),(307,173),(289,176),(266,158),(255,133)],
    'blue_patch':[(221,174),(247,177),(271,191),(289,210),(282,231),(255,233),(234,221),(219,198)],
    'nearby_hair_control':[(180,114),(205,117),(220,134),(226,156),(210,168),(189,159)],
}


def save_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False)
        stream.write('\n')


def png(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB')).copy()


def registered_tint():
    path=OUT/'input-audit'/'016-fixed-hair-color-regions.json'
    if path.exists():
        return path,read(path)
    swap_path=pin(OUT/'input-audit/016-color-geometry-swap/result.json',SWAP_SHA)
    swap=read(swap_path)
    historical=read(pin(R27/'ray-comparison/016-18000-signed.json',REFERENCE_SHA))
    qa=read(checked(historical['qa']['result']))
    records={name:qa['static'][name]['records'][TINT_VIEW] for name in ('r22','r27')}
    x0,y0,x1,y1=swap['crop_xyxy']; width,height=x1-x0,y1-y0
    signed_panels={name:row['panel'] for name,row in records.items()}
    need(swap['registered_view']==TINT_VIEW and all(row['crop_xyxy']==swap['crop_xyxy'] for row in signed_panels.values()),
         'The hair color polygons would use another camera crop.')
    overview=Image.open(checked(swap['figure'])).convert('RGB')
    need(overview.size==(4*width,height+30),'The frozen color-swap board format differs.')
    draw=ImageDraw.Draw(overview)
    for index in range(4):
        for (name,points),color in zip(HAIR_POLYGONS.items(),('#56ee7d','#60a9ff','#ffe05b'),strict=True):
            draw.line([(x+index*width,y+30) for x,y in [*points,points[0]]],fill=color,width=2)
    image_path=path.with_suffix('.png'); image_path.parent.mkdir(parents=True,exist_ok=True)
    overview.save(image_path,compress_level=2)
    receipt=dict(schema='shuolee-r28-fixed-hair-color-regions/1',subject='016',view=TINT_VIEW,
        crop_xyxy=swap['crop_xyxy'],polygon_crop_local_pixels=HAIR_POLYGONS,
        source_swap=descriptor(swap_path),annotated_swap=descriptor(image_path),signed_baseline_panels=signed_panels,
        alpha_threshold_code=243,alpha_mask_policy='Each polygon is intersected with pixels whose signed R22 and R27 alpha is at least 243/255. The same frozen mask applies to every R28 endpoint.',
        selection_basis='Visual patches on the signed R22/R27 swap board, fixed before reading candidate R28 outputs. They are synthetic camera pixels.')
    save_json(path,receipt)
    return path,receipt


def baseline_region_masks(plan):
    x0,y0,x1,y1=plan['crop_xyxy']; w,h=x1-x0,y1-y0
    keep=np.ones((h,w),bool)
    for item in plan['signed_baseline_panels'].values():
        panel=png(checked(item))
        need(panel.shape==(h,4*w,3),'A baseline tint panel shape changed.')
        keep&=panel[:,2*w:3*w,0]>=plan['alpha_threshold_code']
    masks={}
    for name,points in plan['polygon_crop_local_pixels'].items():
        raster=Image.new('1',(w,h))
        ImageDraw.Draw(raster).polygon([tuple(p) for p in points],fill=1)
        masks[name]=np.asarray(raster)&keep
        need(int(masks[name].sum())>=20,'The fixed hair color region has insufficient shared alpha.')
    return masks


def region_pixels(model, world, crop, masks, archived=None):
    x0,y0,x1,y1=crop
    pre=preprocess(model['arrays'],world_pose_in_viewer(world,model['transform']))
    stack=np.stack(list(masks.values()))
    yy,xx=np.nonzero(stack.any(0))
    x,y=xx+x0,yy+y0
    rgb=np.zeros((len(x),3),np.int16); alpha=np.zeros(len(x),np.int16)
    # Reuse the stock float32 compositing reduction for exact per-pixel parity.
    for tile in np.unique(np.column_stack((x//16,y//16)),axis=0):
        here=(x//16==tile[0])&(y//16==tile[1])
        rows,weights=native_weights(pre,tuple(tile.tolist()),np.column_stack((x[here],y[here])))
        rgb[here]=code(weights@pre['rgb'][rows])
        alpha[here]=code(weights.sum(1))
    parity=None
    if archived is not None:
        panel=png(checked(archived['panel']))
        w=x1-x0; h=y1-y0
        need(panel.shape==(h,4*w,3),'The independent hair color panel shape changed.')
        rgb_error=np.abs(rgb-panel[yy,xx]).max(1)
        alpha_error=np.abs(alpha-panel[yy,xx+2*w,0])
        delta=np.maximum(rgb_error,alpha_error)
        need(delta.max()<=1,'Fixed hair color ROI CPU/GPU parity exceeds 1 code.')
        parity=dict(pixels=len(delta),exact_pixels=int((delta==0).sum()),
                    max_rgb_code=int(rgb_error.max()),max_alpha_code=int(alpha_error.max()))
    values={}
    for name,mask in masks.items():
        selected=mask[yy,xx]
        channels=rgb[selected].astype(np.float64)
        opacity=alpha[selected].astype(np.float64)
        green=channels[:,1]-.5*(channels[:,0]+channels[:,2])
        blue=channels[:,2]-.5*(channels[:,0]+channels[:,1])
        values[name]=dict(pixels=int(selected.sum()),mean_black_rgb_code=channels.mean(0).tolist(),
            green_excess_mean_code=float(green.mean()),blue_excess_mean_code=float(blue.mean()),
            green_excess_p90_code=float(np.quantile(green,.90)),blue_excess_p90_code=float(np.quantile(blue,.90)),
            mean_alpha_code=float(opacity.mean()),below_alpha_243_pixels=int((opacity<243).sum()))
    return values,parity


def describe_pair(base,candidate,old,new,points):
    artifact,control=(describe_ray(base,candidate,old[xy],new[xy]) for xy in points)
    for target in (artifact,control):
        for row in target['largest_signed_rgb_row_changes']:
            if row['stable_id']<0:
                row.update(stable_id=-row['stable_id']-1,allocation_branch='r27')
    contrast={phase:float(LUMA@(np.asarray(artifact[phase]['black'])-np.asarray(control[phase]['black'])))
              for phase in ('before','after')}
    return dict(artifact=dict(xy=points[0],**artifact),control=dict(xy=points[1],**control),
                artifact_minus_control_luma_codes=contrast)


def run(subject,step,verify_qa):
    sys.path[:0]=[str(PROJECT/'src'),str(PROJECT/'src/avatar_core')]
    torch.set_num_threads(2)
    tint_path,tint=registered_tint() if subject=='016' else (None,None)
    base=load_subject(subject,read(pin(DELIVERY/'review-receipt.json',REVIEW_SHA)))
    base['lookup']={int(value):i for i,value in enumerate(base['stable'])}
    base['transform']=base['arm']['world_to_viewer']
    root=LAB/f'shuolee-smpl-delivery-round28-{subject}'
    side='r28_10' if step==10000 else 'r28'
    roster=base['plan']['static_grid']['camera_roster']
    reference=base['plan']['poses']['r22']['new']['reference_frame_id']
    models={'r22':base,'r27':load_endpoint(LAB/f'shuolee-smpl-delivery-round27-{subject}',18000,reference),
            side:load_endpoint(root,step,reference)}
    historical=models['r27']; ids=historical['stable']
    lineage=np.where(np.isin(ids,base['stable']),ids,-ids-1)
    historical=dict(historical,stable=lineage,lookup={int(stable):i for i,stable in enumerate(lineage)})
    qa,qa_proof=None,None
    if verify_qa:
        review_path=root/'view-comparison/result.json'; review=read(review_path)
        plan_path=checked(review['plan']); plan=read(plan_path)
        need(plan['static_grid']['camera_roster']==roster and plan_path==review_path.with_name('plan.json'),
             'The signed R28 replay does not use the fixed camera roster.')
        qa={}; qa_proof=dict(result=descriptor(review_path),plan=descriptor(plan_path),static={})
        for name,model in models.items():
            arm=plan['static_grid']['arms'][name]
            static_path=checked(review['static'][name]['receipt']); static=read(static_path)
            digest=base['hashes']['portable'] if name=='r22' else model['hashes']['portable_ply']['sha256']
            need(arm['world_to_viewer']==model['transform'] and arm['ply_sha256']==static['ply_sha256']==digest and
                 static['records']==review['static'][name]['records'] and static['view_count']==len(roster),
                 'The independent signed static replay differs from its published endpoint.')
            qa[name]=static['records']; qa_proof['static'][name]=descriptor(static_path)
    views={}; color=None
    for suffix,(description,pairs) in PICKS[subject].items():
        key=VIEW+suffix
        world=roster[key]['world_camera']; crop=roster[key]['head_crop_xyxy']
        original=base['grid']['records'][key]
        proof={}; traces={}
        for name,model in models.items():
            archived=qa[name][key] if qa is not None else original if name=='r22' else None
            proof[name],traces[name]=endpoint_view(model,world,crop,pairs,archived)
        raw=preprocess(models[side]['raw_arrays'],world)
        x0,y0,*_=crop; world_export_max=0
        for points in pairs:
            for xy in points:
                item=traces[side][xy]; native=pixel(raw,(x0+xy[0],y0+xy[1]))
                world_export_max=max(world_export_max,int(np.abs(code(item['rgb'])-code(native['rgb'])).max()),
                                     int(np.abs(code(item['opacity'])-code(native['opacity']))))
        need(world_export_max<=1,'The fixed marked rays changed through the portable export.')
        pairs_by_baseline={name:[describe_pair(model,models[side],traces[name],traces[side],pair) for pair in pairs]
                           for name,model in [('r22',base),('r27',historical)]}
        views[key]=dict(description=description,source_world_camera=world,crop=crop,
            candidate_raw_to_portable_max_code=world_export_max,parity_by_arm=proof,pairs_by_baseline=pairs_by_baseline)
        if key==TINT_VIEW and tint is not None:
            masks=baseline_region_masks(tint)
            color=dict(registration=descriptor(tint_path),regions_by_arm={},parity_by_arm={})
            for name,model in models.items():
                archived=qa[name][key] if qa is not None else {'panel':tint['signed_baseline_panels'][name]} if name in ('r22','r27') else None
                stats,parity=region_pixels(model,world,crop,masks,archived)
                color['regions_by_arm'][name]=stats; color['parity_by_arm'][name]=parity
    result=dict(schema='shuolee-r28-paired-fixed-marked-face-rays/1',subject=subject,step=step,side=side,
        source_sha256=sha(__file__),verified_against_gpu_static=verify_qa,qa=qa_proof,
        baseline_r22=base['hashes'],endpoint_r27=models['r27']['hashes'],endpoint_r28=models[side]['hashes'],
        views=views,hair_color_fixed_regions=color,scope='Identical historical synthetic face rays and nearby controls. The fixed hair polygons share the same original alpha mask. These cameras have no real RGB target.')
    target=OUT/'ray-comparison'/f'{subject}-{step}-{"signed" if verify_qa else "cpu"}.json'
    save_json(target,result)
    print(dict(file=str(target),sha256=sha(target),verified_against_gpu_static=verify_qa),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject',choices=('016','006'))
    parser.add_argument('step',type=int,nargs='?',choices=(10000,18000))
    parser.add_argument('--verify-qa',action='store_true')
    parser.add_argument('--register-hair-rois',action='store_true')
    args=parser.parse_args()
    if args.register_hair_rois:
        path,plan=registered_tint()
        print(dict(file=str(path),sha256=sha(path),pixels={name:int(mask.sum()) for name,mask in baseline_region_masks(plan).items()}))
    elif args.step is None:
        parser.error('Use an actual complete R28 checkpoint step.')
    else:
        run(args.subject,args.step,args.verify_qa)
