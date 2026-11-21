"""CPU review glue: same source/cameras1291000 versus fresh DC-box1301000.

The frozen129 reviewer and all source/fit arrays remain unchanged. Source RGB,
mask, learned mesh, GS and cumulative alpha use the existing native path.
"""
import argparse
import ast
import difflib
import hashlib
import os
from pathlib import Path
import types

import numpy as np
from PIL import Image
import review_leafwoman_half12_gaussian129 as previous
from review_full_sequence123 import joined
from run_official_gaussians123 import read,save,sha,now

H=Path(__file__).parent
OUT=previous.OUT.parent/'leafwoman-half12-dcbox1000-130'
BASELINE=previous.OUT
BASE_SHA='3750943adae25a08d57030546e4bfff17f2209b3dc814080a1b7103a25c56e32'
TRIAL_HELPER=H/'train_leafwoman_dcbox130.py'
TRIAL_SHA='0de0fb5ac54bb5446af896a765a7c98ce07d65572df066c2b2dfaffaf225a5af'
TRIAL_PLAN=H/'train_leafwoman_dcbox130.plan.json'
TRIAL_PLAN_SHA='498f3faa55c63c8e2b92ad4793fef91c6cac635b69c086812bdcf5ac164c445f'


def compiled_review():
    assert sha(previous.__file__)==BASE_SHA
    main,_,before=previous.compiled_review();code=before
    for old,new in [('高斯_持叶半圈12帧129_','高斯_持叶DCbox半圈12帧130_'),
                    ('LEAFWOMAN12_GAUSSIAN_REVIEW_READY','LEAFWOMAN_DCBOX130_GAUSSIAN_REVIEW_READY')]:
        assert code.count(old)==1;code=code.replace(old,new)
    ast.parse(code)
    namespace=dict(main.__globals__,OUT=OUT)
    exec(compile(code,str(__file__)+'::unchanged_CPU_source_review','exec'),namespace)
    return namespace['main'],before,code


# Only the input-output root is rebound. The exact fit24 ID/fullK validator
# and129 source context functions are reused without changing their code.
_scope=dict(previous.__dict__,OUT=OUT,compiled_review=compiled_review,__file__=__file__)
_fn=previous.validate_prepared_subset
validate_prepared_subset=types.FunctionType(_fn.__code__,_scope,_fn.__name__,_fn.__defaults__,_fn.__closure__)
_scope['validate_prepared_subset']=validate_prepared_subset
_fn=previous.review
source_review=types.FunctionType(_fn.__code__,_scope,_fn.__name__,_fn.__defaults__,_fn.__closure__)


def same_inputs():
    """Paths differ intentionally; source pixels/masks/fullK/ROI must not."""
    assert sha(TRIAL_HELPER)==TRIAL_SHA and sha(TRIAL_PLAN)==TRIAL_PLAN_SHA
    a=read(BASELINE/'data/training-inputs.json')['records'];b=read(OUT/'data/training-inputs.json')['records']
    sa=read(BASELINE/'source-native-inputs.json')['records'];sb=read(OUT/'source-native-inputs.json')['records']
    ma=read(BASELINE/'pixel-mapping.json')['records'];mb=read(OUT/'pixel-mapping.json')['records']
    assert len(a)==len(b)==len(sa)==len(sb)==len(ma)==len(mb)==previous.N
    for x,y,s,t,m,n in zip(a,b,sa,sb,ma,mb):
        for key in ('frame','wh','K','K_integer_centers','w2c','source_sha256','mask_sha256','validity_sha256','source_joint_group_supported','face_region_is_observed'):
            assert x[key]==y[key],(x['frame'],key)
            assert s[key]==t[key],(s['frame'],'source',key)
        for key in ('frame','crop_xyxy','scale','integer_center_transform','derived_wh'):
            assert m[key]==n[key],(x['frame'],'pixel_mapping',key)
    return a,b,ma,mb


def compare(step):
    baseline=BASELINE/'gaussian-run'/f'step-{step:06d}'
    snapshot=OUT/'gaussian-run'/f'step-{step:06d}'
    old=read(baseline/'review/result.json');new=read(snapshot/'review/result.json')
    old_records={r['frame']:r for r in old['records']}
    a,b,ma,mb=same_inputs();main,_,_=compiled_review();scope=main.__globals__
    prediction,alpha=scope['source_pixel_prediction'],scope['source_pixel_alpha']
    dest=snapshot/'review/comparison129';dest.mkdir(exist_ok=False)
    records=[]
    for row,other,mapping,mapping_new in zip(a,b,ma,mb):
        frame=row['frame'];rec=next(r for r in new['records'] if r['frame']==frame)
        source=read(OUT/'source-native-inputs.json')['records'][b.index(other)]
        raw=Image.open(source['image']).convert('RGB');mask=Image.open(source['mask']).convert('RGB')
        gauss0,gauss1=prediction(baseline,row,mapping),prediction(snapshot,other,mapping_new)
        av0,ar0=alpha(baseline,row,mapping);av1,ar1=alpha(snapshot,other,mapping_new)
        ad0=Image.fromarray(np.rint(np.clip(av0,0,1)*255).astype(np.uint8)).convert('RGB')
        ad1=Image.fromarray(np.rint(np.clip(av1,0,1)*255).astype(np.uint8)).convert('RGB')
        gauss0,ad0=previous.show_unrendered_as_gray(gauss0,ad0,mapping)
        gauss1,ad1=previous.show_unrendered_as_gray(gauss1,ad1,mapping_new)
        images=[]
        for entry in rec['images']:
            region=entry['region']
            if region=='full_frame':continue # Same-frame full-raster130 stays in the five-panel output.
            box=entry['crop_original_xyxy']
            old_entry=next(e for e in old_records[frame]['images'] if e['region']==region)
            assert old_entry['crop_original_xyxy']==box
            picture=joined([im.crop(box) for im in (raw,mask,gauss0,gauss1,ad0,ad1)])
            path=previous.ARCHIVE/f'高斯对照_持叶130_{step:06d}_{frame}_{region}_原图_mask_129RGB_130RGB_129alpha_130alpha.png'
            assert not path.exists();picture.save(path)
            images.append(dict(region=region,path=str(path),sha256=sha(path),crop_original_xyxy=box,
                panel_order=['source','source_mask','baseline129_1000_RGB','candidate130_1000_RGB','baseline129_1000_cumulative_alpha','candidate130_1000_cumulative_alpha']))
        records.append(dict(frame=frame,images=images,baseline_metrics=old_records[frame]['metrics'],candidate_metrics=rec['metrics'],
            baseline_RGB_sha256=sha(baseline/(frame+'_gaussian.png')),candidate_RGB_sha256=sha(snapshot/(frame+'_gaussian.png')),
            baseline_alpha_sha256=ar0['alpha_npy_sha256'],candidate_alpha_sha256=ar1['alpha_npy_sha256']))
    pages=[]
    for region,height in [('body',430),('arms',350),('face',240)]:
        entries=[(r['frame'],e) for r in records for e in r['images'] if e['region']==region]
        for start in range(0,len(entries),4):
            panels=[]
            for frame,entry in entries[start:start+4]:
                pic=Image.open(entry['path']).convert('RGB');scale=min(1,height/pic.height,1800/pic.width)
                panels.append(pic.resize((round(pic.width*scale),round(pic.height*scale)),Image.Resampling.LANCZOS))
            sheet=Image.new('RGB',(max(p.width for p in panels),sum(p.height for p in panels)),(25,25,25));y=0
            for pic in panels:sheet.paste(pic,(0,y));y+=pic.height
            path=dest/f'{region}-{start//4+1:02d}.png';sheet.save(path)
            pages.append(dict(region=region,path=str(path),sha256=sha(path),frames=[f for f,e in entries[start:start+4]]))
    save(dest/'result.json',dict(updated=now(),step=step,records=records,pages=pages,
        comparison='Same frozen129 source/fit/masks/fullK/ROI and1000 actual updates; fresh130 DC-box candidate versus actual1291000 baseline. No refit/reselection/renderer/opacity-policy change.',
        baseline_snapshot_result_sha256=sha(baseline/'result.json'),candidate_snapshot_result_sha256=sha(snapshot/'result.json'),
        baseline_review_result_sha256=sha(baseline/'review/result.json'),candidate_review_result_sha256=sha(snapshot/'review/result.json'),
        source_pixel_metrics_are_not_geometric_truth=True,ROI_exterior='Gray48 is not-rendered/unknown, not alpha0.',
        alpha_scope='Both show all-layer cumulative1-T, not front-surface-only coverage.',
        unknown_lower_body_or_back_not_recovered_by_this_comparison=True,actual_visual_review=False,quality_accepted=False))
    return dest/'result.json'


def review(step):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and step==1000
    assert read(OUT/'process-exit.json')['exit_code']==0
    dc=read(OUT/'dc-projection-1000-receipt.json')
    assert dc['original_Adam_steps']==dc['DC_projection_steps']==1000
    assert dc['complete_checkpoint_after_projection'] is True and dc['Adam_moments_preserved'] is True
    same_inputs();source_review(step)
    path=OUT/'gaussian-run/step-001000/review/result.json';data=read(path)
    data.update(candidate='leafwoman130_DCbox_fresh1000',baseline1291000_comparison_pending=False,
        only_new_scientific_constraint='Post-original-Adam SH0 DC projection to source-SDR RGB[0,1].',
        training_helper_sha256=TRIAL_SHA,trial_plan_sha256=TRIAL_PLAN_SHA,DC_projection_receipt_sha256=sha(OUT/'dc-projection-1000-receipt.json'),
        same_source_masks_fullK_pixel_mapping_verified=True,quality_accepted=False)
    save(path,data)
    comparison=compare(step)
    print('LEAFWOMAN_DCBOX130_CPU_SOURCE_AND_129_COMPARISON_READY',str(comparison),flush=True)


def audit():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and sha(previous.__file__)==BASE_SHA
    assert sha(TRIAL_HELPER)==TRIAL_SHA and sha(TRIAL_PLAN)==TRIAL_PLAN_SHA
    context=previous.input_context();_,before,code=compiled_review()
    assert validate_prepared_subset.__code__ is previous.validate_prepared_subset.__code__
    assert source_review.__code__ is previous.review.__code__
    prepared=all((OUT/p).exists() for p in ('data/training-inputs.json','source-native-inputs.json','pixel-mapping.json'))
    if prepared:same_inputs()
    save(Path(__file__).with_suffix('.audit.json'),dict(updated=now(),state='CPU_static130_review_ready',
        helper_sha256=sha(__file__),frozen129_reviewer_sha256=BASE_SHA,trial_helper_sha256=TRIAL_SHA,trial_plan_sha256=TRIAL_PLAN_SHA,
        frames=context['frames'],reference_frame=context['reference'],prepared_input_contract_checked=prepared,
        snapshot130_read=False,CPU_render_executed=False,GPU_started=False,source_fit_training_or129_evidence_modified=False,
        current_mesh_fit24_ID_fullK_unknown_feet_rules_unchanged=True,
        comparison_regions=['body','arms','face'],candidate_also_has_full_frame=True,
        native_record_panels_not_resized=True,contact_pages_resize_for_display_only=True,
        quality_accepted=False,interface_diff=''.join(difflib.unified_diff(before.splitlines(True),code.splitlines(True),fromfile='frozen129_CPUreview',tofile='same_CPUreview_new130_output_label'))))
    print('LEAFWOMAN_DCBOX130_CPU_REVIEW_STATIC_READY',sha(__file__),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['audit','review']);parser.add_argument('--step',type=int,default=1000);args=parser.parse_args()
    audit() if args.mode=='audit' else review(args.step)
