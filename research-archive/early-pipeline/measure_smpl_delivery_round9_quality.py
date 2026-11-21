"""Compare exact R8/R9 native ear and static hotspot lossless panels on CPU."""

import argparse
import hashlib
import json
import re
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parent))
import measure_smpl_delivery_round8_quality as previous

prior = previous.prior


native, require, pin, read, sha = prior.native, prior.require, prior.pin, prior.read, prior.sha
R7, R8 = prior.R7, previous.R8
R8_SHA = '208df5371dff11026a294ec0facfea6f6aa4200d2b717583f29f2b56c21709bf'
EAR_SHA = '18da159115e9d72a32f858533bb21298e85fd7ac6faf0f9ba59c1ebe0f8e812e'
HOT_SHA = '40256da439b580121e34c537998f4594eb4b2eba93cff7d3fb6bf6362876da6e'
OLD_LABEL = f'round8_{previous.COMMIT[:7]}'


def endpoints(subject, commit):
    spec=read(pin(R8/'receipt.json',R8_SHA))['subjects'][subject]
    old_path=pin(spec['review'],spec['review_sha256']); old=read(old_path)
    old_run=pin(spec['result'],spec['result_sha256']); old_plan=read(pin(old_path.parent/'plan.json',old['plan_sha256']))
    require(old['state']=='round8_delivery_views_complete' and old['subject']==subject and old_plan['new_commit']==previous.COMMIT,
            'The published R8 baseline is incomplete.')
    if commit is None:
        return (old_path,old,old_plan,old_run),None
    root=prior.LAB/f'shuolee-smpl-delivery-round9-{subject}'
    path=root/'view-comparison/result.json'; review=read(path)
    plan=read(pin(path.parent/'plan.json',review['plan_sha256']))
    wrapper=read(root/'wrapper-terminal.json')
    require(review['state']=='round9_delivery_views_complete' and review['subject']==subject and review['optimizer_updates']==review['refits']==0
            and plan['new_commit']==wrapper['candidate_commit']==commit and wrapper['state']=='completed'
            and Path(plan['baseline_review'])==old_path and plan['baseline_review_sha256']==spec['review_sha256']
            and plan['new']['terminal_sha256']==wrapper['result_sha256'], 'Use the completed exact R9 same-camera review.')
    run=pin(root/'run/result.json',wrapper['result_sha256'])
    require(run.parent==Path(plan['new']['run']) and read(run)['final_snapshot']=='step-010000', 'Unexpected R9 reviewed terminal.')
    return (old_path,old,old_plan,old_run),(path,review,plan,run)


def ear(commit):
    (old_path,old,_,old_run),candidate=endpoints('016',commit)
    baseline=read(pin(R8/'ear-016-reevaluation.json',EAR_SHA)); audit=read(pin(native.ROUND4/'ear-016-source-audit.json',native.EAR_SHA))
    require(baseline['state']=='remeasured' and baseline['round8']['review_sha256']==sha(old_path),'R8 ear receipt differs.')
    coords=audit['roi']; identity=audit['identity']; crop=baseline['roi']['crop_xyxy']
    mask=prior.mask([coords['polygon_xy_native_integer_centers_inclusive']],coords['wh'])
    y,x=np.nonzero(mask); xy=np.column_stack((x,y)); local=xy-crop[:2]
    require(len(xy)==574 and hashlib.sha256(mask.tobytes()).hexdigest()==coords['roi_sha256']==baseline['roi']['sha256'], 'Ear polygon changed.')
    raw,matte=identity['source_rgb'],identity['score_mask']
    with Image.open(pin(raw['path'],raw['sha256'])) as rgb, Image.open(pin(matte['path'],matte['sha256'])) as fg:
        require(rgb.size==fg.size==tuple(coords['wh']) and np.all(np.asarray(fg.convert('L'))[y,x]==255),'Original source or matte support differs.')
        source=np.asarray(rgb.convert('RGB').crop(crop))
    rec=native.camera_record(old['new'],audit)
    _,mesh=native.posed_mesh(old_run,audit,rec)
    slot=rec['native_camera']['pose_index']
    owners=native.mesh_coverage(mesh['vertices_world'][slot],mesh['faces'],rec['native_camera'],xy); hit=owners>=0
    require(int((~hit).sum())==267 and np.unique(owners[hit]).tolist()==audit['mesh_support']['distinct_right_ear_owner_faces'],'Locked ear mesh support differs.')
    pane,strips=native.panel(old_path,old,source,crop,[previous.OLD_LABEL,OLD_LABEL]); old_metrics=native.metrics(strips[-1],source,local,hit,old_path,rec)
    require(old_metrics==baseline['round8']['metrics'] and sha(pane)==baseline['round8']['panel_sha256'],'R8 ear scores do not reproduce.')
    out=dict(schema='shuolee-smpl-round9-ear016-remeasurement-v1',state='baseline_verified',subject='016',frame=native.FRAME,
        roi=baseline['roi'],source_sha256=raw['sha256'],mask_sha256=matte['sha256'],native_camera=rec['native_camera'],
        round8=baseline['round8'],round8_ear_receipt_sha256=EAR_SHA,gpu_calls=0,optimizer_updates=0,quality_accepted=False)
    if candidate is None:
        return out
    path,new,_,run=candidate; record=native.camera_record(new['new'],audit)
    require(record['native_camera']==rec['native_camera'] and record['pose_sha256']==rec['pose_sha256'],'R9 ear camera or fit pose changed.')
    panel,rows=native.panel(path,new,source,crop,[OLD_LABEL,f'round9_{commit[:7]}'])
    require(np.array_equal(rows[0],strips[-1]),'The R9 comparison first row differs from pinned R8.')
    metrics=native.metrics(rows[1],source,local,hit,path,record)
    out.update(state='remeasured',**prior.verify_replay(old_run,run),round9=dict(panel=str(panel),panel_sha256=sha(panel),review_sha256=sha(path),metrics=metrics),
        delta_round9_minus_round8={region:{k:metrics[region][k]-old_metrics[region][k]
            for k in ('alpha_u8_lt_166','alpha_u8_lt_243','black_rgb_mae_01')} for region in ('all_roi','no_mesh','right_ear_owner_mesh')})
    return out


def split(row):
    black,white,alpha=np.split(row,3,axis=1)
    require(np.array_equal(alpha[...,0],alpha[...,1]) and np.array_equal(alpha[...,0],alpha[...,2]),'The direct alpha PNG is not grayscale.')
    return black,white,alpha[...,0]


def hotspot(commit):
    baseline=read(pin(R8/'hotspot-comparison.json',HOT_SHA))
    causal=read(pin(R7/'causal-diagnostic/receipt.json',prior.CAUSAL_SHA))
    require(baseline['state']=='remeasured' and baseline['round8_commit']==previous.COMMIT
            and baseline['r6_causal_receipt_sha256']==prior.CAUSAL_SHA and causal['state']=='completed', 'R8 hotspot and causal lineage differ.')
    out=dict(schema='shuolee-smpl-round9-static-hotspot-comparison-v1',state='baseline_verified' if commit is None else 'remeasured',
        r8_hotspot_receipt_sha256=HOT_SHA,r6_causal_receipt_sha256=prior.CAUSAL_SHA,round9_commit=commit,gpu_calls=0,optimizer_updates=0,
        quality_accepted=False,subjects={},scope='Same-camera synthetic fixed-polygon changes. No matching source RGB truth and no proof that artifacts have not relocated.')
    for subject, base in baseline['subjects'].items():
        (old_path,old,old_plan,old_run),candidate=endpoints(subject,commit)
        view=base['view']; old_rec=old['static_render_receipts']['round8'][view]
        crop=old_plan['static_views'][view]['crop_xyxy']; w,h=np.diff(np.asarray(crop).reshape(2,2),axis=0)[0]
        point_audit=causal['subjects'][subject]; diagnostic=read(pin(point_audit['path'],point_audit['sha256']))
        require(crop==base['crop_xyxy'] and old_rec['sha256']==base['round8_png_sha256']
                and all(base[key]==diagnostic[key] for key in ('hotspot_poly_crop_local','control_poly_crop_local')),
                'R8 static crop, causal regions or PNG identity differs.')
        original=prior.png(old_path.parent/old_rec['file'],old_rec['sha256']); before=split(original)
        hot=prior.mask(base['hotspot_poly_crop_local'],(w,h)); control=prior.mask(base['control_poly_crop_local'],(w,h))
        require(original.shape==(h,3*w,3) and int(hot.sum())==diagnostic['hotspot_pixels']
                and int(control.sum())==diagnostic['control_pixels'] and not (hot&control).any(),'Locked static polygons differ.')
        verified=prior.static_metrics(before,before,hot,control)
        require(all(value['round6']==base['metrics'][region]['round8'] for region,value in verified.items()), 'R8 hotspot metrics do not reproduce.')
        entry=dict(view=view,crop_xyxy=crop,hotspot_poly_crop_local=base['hotspot_poly_crop_local'],
            control_poly_crop_local=base['control_poly_crop_local'],round8_png_sha256=old_rec['sha256'])
        if candidate is not None:
            path,new,plan,run=candidate; record=new['static_render_receipts']['round9'][view]
            pane=prior.choose(new['panels'],view=view,scale=1.0)
            require(pane['columns']==['black','white','alpha'] and pane['rows']==[OLD_LABEL,f'round9_{commit[:7]}']
                    and pane['header_pixels']==0 and pane['crop_xyxy']==crop and record['camera']==old_rec['camera']
                    and record['source_rgb_compared'] is False and record['static_ply_sha256']==read(run)['sha256']
                    and plan['static_views'][view]==old_plan['static_views'][view], 'The matched static camera, crop, or PLY changed.')
            candidate_png=prior.png(path.parent/record['file'],record['sha256']); composite=prior.png(path.parent/pane['file'],pane['sha256'])
            require(composite.shape==(2*h,3*w,3) and np.array_equal(composite[:h],original) and np.array_equal(composite[h:],candidate_png),
                    'R9 matched PNG differs from direct R8/R9 rendering rows.')
            metrics=prior.static_metrics(before,split(candidate_png),hot,control)
            for value in metrics.values():
                baseline_stats,candidate_stats=value.pop('round6'),value.pop('round7')
                value.update(round8=baseline_stats,round9=candidate_stats)
            entry.update(**prior.verify_replay(old_run,run),round9_png_sha256=record['sha256'],matched_panel_sha256=pane['sha256'],metrics=metrics)
        out['subjects'][subject]=entry
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('part',choices=('ear','hotspot'))
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-baseline',action='store_true'); mode.add_argument('--commit')
    parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.commit and not re.fullmatch('[0-9a-f]{40}', args.commit):
        parser.error('Pin the full R9 commit.')
    data=(ear if args.part=='ear' else hotspot)(args.commit)
    text=json.dumps(data,indent=2,allow_nan=False)+'\n'
    if args.output:
        with args.output.open('x',encoding='utf-8') as stream:stream.write(text)
    print(text,end='')


if __name__=='__main__':main()
