"""CPU-only128 adapter of the frozen football source-only12+4 shortlist.

No roster admission, fitting, detector execution, camera optimization or mask
editing. Candidate contacts are display aids, not joint/material ground truth.
"""
import argparse
import ast
import hashlib
import inspect
import json
import math
from pathlib import Path
import textwrap

import select_officewoman_circle24_sources127 as base

N_TRAIN, N_HOLD, PER_SLOT = 12, 4, 2
HERE = Path(__file__).resolve().parent
ROOT = base.ROOT
SOURCE = Path('/opt/gs-slam-lab/shuolee-cross-subject-128/graywoman')
OUT = ROOT/'graywoman-half12-input128'
VIDEO_SHA = '965befa346fdb0b5e5140745ab5651ef6ed7f8a87ae32777e1d88017bc45e547'
BASE_SHA = 'a2a23bd1661c66160bce3dc594d5663653f56a29331657ee04805708f872b41d'
FOOTBALL_PLAN = ROOT/'football-half12-input126/plan.json'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


PROTOCOL = dict(
    training_count=12, heldout_count=4, heldout_reserve_count=8,
    heldout='Reserve nearest2 real-PTS frames to each of4 quarter-bin centers, ties greater native face Laplacian variance then greater raw17 body score then stable ID. Choose first source-supported candidate after review. All8 reserves excluded from training, including unused backups.',
    training='12equal-real-PTS bins. Exclude all8 reserves. Per-bin .6rank(-native face Laplacian variance)+.4rank(-raw17 body detector score), ties nearest time-bin center then stable frameID. Inspect only top2 per bin. No fit error is read.',
    rank_ties='Equal metric values share rank=(number strictly better)/(n-1); n=1 gives0. Same source-only dense ranking as frozen football126.',
    fallback='If a bin lacks a valid source, retain the coverage gap. Fill only from the already reviewed finite training pool, descending nearest real-PTS distance to selected set, ties frozen source-quality rank then frameID. Recompute distances after each addition. Never use heldout reserves.',
    reference='After source review only: among selected12 source-confirmed observed face/open-eye sources, .6rank(-native face sharpness)+.4rank(-raw17 body score), ties stable frameID. First frame is not preferred. No reference is selected by this draft.',
    visibility='Hands are behind lower back. Occluded wrists/fingers/elbows remain unknown; native detector guesses are not observations. No per-frame manual pose editing. Source-review templates are null, never implicitly true.',
    confidence='Future existing confidence normalization must use only frozen train12 scores, never heldout scores/coordinates. This helper does not normalize or fit.',
    coverage='Qualitative half-circle from source side/front/opposite-side review, no rear. Time bins are coverage proxies, not measured yaw bins or physical180degrees. No equal-angle claim.',
    mask='Long connected shoe shadows must be inspected in raw/mask contacts. Do not edit masks or silently remove foreground pixels here.',
    evidence_scope='Independent holdout frames of this new fit, not an unseen video or verified physical landmarks. Conditional H0 K is inherited unchanged; physical calibration remains unaccepted.',
    budget='At most24 training candidates and8 heldout candidates. No full-sequence fitting, added extraction, or expanded scan after this finite pool.')


def shortlist(features):
    fs=sorted(features,key=lambda r:(r['time_seconds'],r['frame']))
    assert len(fs)>=N_TRAIN+N_HOLD*PER_SLOT and len({r['frame'] for r in fs})==len(fs)
    start,end=fs[0]['time_seconds'],fs[-1]['time_seconds'];assert end>start
    held=[];reserved=set()
    for slot in range(N_HOLD):
        center=start+(end-start)*(slot+.5)/N_HOLD
        ordered=sorted((r for r in fs if r['frame'] not in reserved),key=lambda r:(abs(r['time_seconds']-center),-r['face_sharpness'],-r['raw_body_score'],r['frame']))[:PER_SLOT]
        reserved.update(r['frame'] for r in ordered)
        held.append(dict(slot=slot,center_seconds=center,candidate_order=[r['frame'] for r in ordered]))
    train=[]
    for slot in range(N_TRAIN):
        left=start+(end-start)*slot/N_TRAIN;right=start+(end-start)*(slot+1)/N_TRAIN;center=(left+right)/2
        available=[dict(r) for r in fs if r['frame'] not in reserved and left<=r['time_seconds'] and (r['time_seconds']<right or slot==N_TRAIN-1)]
        sharp=base.rank([-r['face_sharpness'] for r in available]);score=base.rank([-r['raw_body_score'] for r in available])
        for r,s,q in zip(available,sharp,score):r['source_quality_rank']=.6*s+.4*q
        ordered=sorted(available,key=lambda r:(r['source_quality_rank'],abs(r['time_seconds']-center),r['frame']))
        train.append(dict(slot=slot,time_interval=[left,right],candidate_order=[r['frame'] for r in ordered[:PER_SLOT]],all_ordered_candidates=ordered))
    return dict(training_candidate_slots=train,holdout_reservation=held)


def adapted_prepare():
    assert sha(base.__file__)==BASE_SHA,'Frozen CPU source extractor changed.'
    source=textwrap.dedent(inspect.getsource(base.prepare))
    replacements={
        "obs['scores'][i,5:17].sum()":"obs['scores'][i,:17].sum()",
        'raw_body12_score':'raw_body_score',
        "subject='officewoman',record_id='019'":"subject='graywoman',record_id='015'",
    }
    for old,new in replacements.items():
        assert source.count(old)==1,(old,source.count(old))
        source=source.replace(old,new)
    ast.parse(source)
    namespace=dict(base.__dict__)
    namespace.update(N_TRAIN=N_TRAIN,N_HOLD=N_HOLD,PER_SLOT=PER_SLOT,SOURCE=SOURCE,OUT=OUT,
                     VIDEO_SHA=VIDEO_SHA,PROTOCOL=PROTOCOL,shortlist=shortlist,__file__=__file__)
    exec(compile(source,str(__file__)+'::frozen_cpu_prepare','exec'),namespace)
    return namespace['prepare'],source


def audit():
    _,source=adapted_prepare()
    assert 'obs[\'scores\'][i,:17].sum()' in source
    assert "body_sharpness=float(lap[mask].var())" in source # retained unused diagnostic, never ranked
    features=[dict(frame='frame_%06d'%(i+1),time_seconds=i/3,face_sharpness=float(i%7),raw_body_score=float(i%11)) for i in range(64)]
    original=json.dumps(features,sort_keys=True);s=shortlist(features)
    assert original==json.dumps(features,sort_keys=True)
    reserved={f for x in s['holdout_reservation'] for f in x['candidate_order']}
    candidates={f for x in s['training_candidate_slots'] for f in x['candidate_order']}
    assert len(reserved)==8 and len(candidates)==24 and not reserved&candidates
    assert all(len(x['candidate_order'])==2 for x in s['training_candidate_slots'])
    assert base.rank([0,0,1])==[0,0,1]
    with_errors=[dict(r,old_fit_error=999-i) for i,r in enumerate(features)]
    # Additional arbitrary fields cannot change candidate IDs or ordering.
    t=shortlist(with_errors)
    assert [x['candidate_order'] for x in s['training_candidate_slots']]==[x['candidate_order'] for x in t['training_candidate_slots']]
    frozen=load(FOOTBALL_PLAN)
    assert frozen['algorithm']['training'].startswith('12equal-time bins')
    return dict(CPU_static_audit=True,helper_sha256=sha(__file__),base_helper_sha256=BASE_SHA,
                football_frozen_plan_sha256=sha(FOOTBALL_PLAN),prepared_function_sha256=hashlib.sha256(source.encode()).hexdigest(),
                synthetic_train_candidates=24,synthetic_holdout_reserves=8,fit_errors_unused=True,
                no_GPU=True,no_old_individual_state=True,final_roster_created=False)


def prepare():
    receipt=audit();fn,_=adapted_prepare();fn()
    path=OUT/'plan-before-source-review.json';plan=load(path)
    assert plan['source_frames']==64 and plan['training_frames']==[] and plan['heldout_frames']==[] and plan['reference_frame'] is None
    assert len(plan['source_review_template'])==32
    plan.update(state='source_only_finite_shortlist_awaiting_actual_review',source_wh=[1520,2704],
                preparation_audit=receipt,conditional_K=True,formal_calibration_accepted=False,
                source_person='New gray sweater older woman; hands behind lower back; hidden fingers/wrists unknown.',
                actual_pose_or_beta_initialized=False,source_color_tags_verified=False,
                no_second_undistortion=True,source_masks_modified=False,source_pixels_modified=False)
    for entry in plan['source_review_template'].values():
        entry.update(left_wrist_source_supported=None,right_wrist_source_supported=None,
                     left_elbow_source_supported=None,right_elbow_source_supported=None,
                     left_foot_source_supported=None,right_foot_source_supported=None,
                     shadow_leak_observed=None)
    path.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('GRAYWOMAN128_SHORTLIST_ONLY',len(plan['source_review_template']),sha(path))


def contacts():
    from PIL import Image
    plan_path=OUT/'plan-before-source-review.json';plan=load(plan_path)
    assert plan['training_frames']==[] and plan['heldout_frames']==[] and plan['reference_frame'] is None
    out=OUT/'source-review';assert not out.exists(),'Review contacts already exist; do not regenerate.'
    out.mkdir();records=[];rows={r['frame']:r for r in plan['source_rows']}
    for frame in sorted(plan['source_review_template']):
        r=rows[frame];assert sha(r['image'])==r['source_sha256'] and sha(r['mask'])==r['mask_sha256']
        im=Image.open(r['image']).convert('RGB');mask=Image.open(r['mask']).convert('L');assert im.size==mask.size==(1520,2704)
        bx=r['initial_box_xyxy'];x0=max(0,math.floor(bx[0])-24);y0=max(0,math.floor(bx[1])-24)
        x1=min(im.width,math.ceil(bx[2])+24);y1=min(im.height,math.ceil(bx[3])+24);height=y1-y0
        assert x1>x0 and height>0
        boxes=dict(source=[0,0,im.width,im.height],mask=[0,0,im.width,im.height],
                   head=[x0,y0,x1,min(y1,y0+math.ceil(.30*height))],
                   arms_wrists=[x0,y0+math.floor(.20*height),x1,min(y1,y0+math.ceil(.70*height))],
                   feet=[x0,y0+math.floor(.72*height),x1,y1])
        paths={}
        for name,box in boxes.items():
            p=out/(frame+'-'+name+'.png');(mask if name=='mask' else im).crop(box).save(p)
            paths[name]=dict(path=str(p),sha256=sha(p),crop_original_xyxy=box,native_integer_crop=True,no_resize=True,
                             proposal_not_visibility_truth=name not in ('source','mask'))
        records.append(dict(frame=frame,source_sha256=r['source_sha256'],source_mask_sha256=r['mask_sha256'],paths=paths))
    pages=[]
    for start in range(0,len(records),4):
        page=Image.new('RGB',(1800,1420),'#ececec')
        for index,r in enumerate(records[start:start+4]):
            x=(index%2)*900;y=(index//2)*710
            for name,ox,oy,w,h in [('source',0,0,240,700),('mask',245,0,240,700),('head',495,0,390,220),('arms_wrists',495,225,390,260),('feet',495,490,390,210)]:
                tile=Image.open(r['paths'][name]['path']).convert('RGB');tile.thumbnail((w,h),Image.Resampling.LANCZOS)
                page.paste(tile,(x+ox+(w-tile.width)//2,y+oy+(h-tile.height)//2))
        path=out/('contact-%02d.jpg'%(start//4));page.save(path,quality=94)
        pages.append(dict(path=str(path),sha256=sha(path),frames=[r['frame'] for r in records[start:start+4]],
                          display_only_resize=True,no_text_overlay=True,layout='4candidates row-major; each raw/mask/head/arms-wrists/feet'))
    result=dict(source_plan_sha256=sha(plan_path),source_rows=records,pages=pages,actual_visual_review=False,
                final_roster_created=False,source_images_modified=False,source_masks_modified=False,CPU_only=True,
                region_scope='Native integer detector-body-box proportion crops only; not source visibility or joint ground truth. Raw and mask full raster retained, including shadows.')
    (out/'source-crops.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('GRAYWOMAN128_SOURCE32_CONTACTS_READY',out,sha(out/'source-crops.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['protocol','audit','prepare','contacts']);args=p.parse_args()
    if args.mode=='protocol':print(json.dumps(PROTOCOL,ensure_ascii=False,indent=2))
    elif args.mode=='audit':print(json.dumps(audit(),ensure_ascii=False,indent=2))
    elif args.mode=='prepare':prepare()
    else:contacts()
