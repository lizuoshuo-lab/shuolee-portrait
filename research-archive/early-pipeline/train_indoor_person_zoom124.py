"""One reproducible person-ROI plus 2x sampling control, fresh GS, stop1000."""
import ast
import copy
import difflib
import fcntl
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
from PIL import Image
import torch
import cross_subject_sh0124 as gs
from cache_training_targets124 import adapt

SUBJECT='indoor-person-zoom2x124'
BASE=gs.ROOT/'indoor-selected-sh0-pipeline124'
OUT=gs.ROOT/SUBJECT
SCALE=2
LIMIT=1000


def prepare():
    assert not OUT.exists(),'Preserve candidates; no silent repeat'
    baseline=gs.read(BASE/'plan.json')
    for path,digest in baseline['source_hashes'].items():assert gs.sha(path)==digest,path
    assert gs.read(BASE/'gaussian-run/result.json')['state']=='bounded_control_completed'
    document=gs.read(BASE/'data/training-inputs.json');source_rows=document['records']
    assert len(source_rows)==6
    for folder in ['fit','data/rgb','data/mask','data/validity']:(OUT/folder).mkdir(parents=True)
    for name in ['fit/fitted.npz','fit/surface-cache.npz','selection.json','selection-rule.json','loss_utils_validity.py']:
        shutil.copyfile(BASE/name,OUT/name)
        assert gs.sha(BASE/name)==gs.sha(OUT/name)
    gs.save(OUT/'source-native-inputs.json',document)
    rows=[];mapping=[];checks=[];panels=[]
    for row in source_rows:
        old=copy.deepcopy(row);frame=row['frame']
        rgb=Image.open(row['image']).convert('RGB');mask=Image.open(row['mask']).convert('L')
        valid=Image.open(row['validity']).convert('L')
        assert rgb.size==mask.size==valid.size==tuple(row['wh'])
        ma=np.asarray(mask);ya,xa=np.nonzero(ma>0)
        bbox=[int(xa.min()),int(ya.min()),int(xa.max()+1),int(ya.max()+1)]
        pad=int(np.ceil(.10*max(bbox[2]-bbox[0],bbox[3]-bbox[1])))
        x0,y0=max(0,bbox[0]-pad),max(0,bbox[1]-pad)
        x1,y1=min(rgb.width,bbox[2]+pad),min(rgb.height,bbox[3]+pad)
        box=(x0,y0,x1,y1);wh=((x1-x0)*SCALE,(y1-y0)*SCALE)
        assert np.count_nonzero(ma[y0:y1,x0:x1])==np.count_nonzero(ma)
        resized=rgb.crop(box).resize(wh,Image.Resampling.BILINEAR)
        rm=mask.crop(box).resize(wh,Image.Resampling.NEAREST)
        rv=valid.crop(box).resize(wh,Image.Resampling.NEAREST)
        assert np.array_equal(np.asarray(rm)[::2,::2],ma[y0:y1,x0:x1])
        assert np.array_equal(np.asarray(rv)[::2,::2],np.asarray(valid)[y0:y1,x0:x1])
        paths={key:OUT/'data'/folder/(frame+'.png') for key,folder in [('image','rgb'),('mask','mask'),('validity','validity')]}
        for key,img in [('image',resized),('mask',rm),('validity',rv)]:img.save(paths[key])
        T=np.array([[SCALE,0,-SCALE*x0+(SCALE-1)/2],[0,SCALE,-SCALE*y0+(SCALE-1)/2],[0,0,1.]])
        Kint=T@np.asarray(row['K_integer_centers']);K=Kint.copy();K[:2,2]+=.5
        X=np.array([[.2,-.3,2.],[.7,.4,3.],[-.5,.1,1.4],[0,0,2.]])
        before=X@np.asarray(row['K_integer_centers']).T;before=before[:,:2]/before[:,2:]
        after=X@Kint.T;after=after[:,:2]/after[:,2:]
        mapped=np.c_[before,np.ones(len(before))]@T.T
        error=float(np.max(np.abs(after-mapped[:,:2])));assert error<1e-9
        row=copy.deepcopy(row);row.update(image=str(paths['image']),mask=str(paths['mask']),validity=str(paths['validity']),
            source_sha256=gs.sha(paths['image']),mask_sha256=gs.sha(paths['mask']),validity_sha256=gs.sha(paths['validity']),
            wh=list(wh),K_integer_centers=Kint.tolist(),K=K.tolist(),
            pixel_domain='Deterministic person-mask ROI plus2x bilinear RGB / nearest mask-validity; derived pixels, not new optical detail.',
            original_native_image=old['image'],original_native_wh=old['wh'])
        for key in ['initial_box_xyxy','face_box_xyxy']:
            if row.get(key) is not None:
                row[key]=((np.asarray(old[key])-np.array([x0,y0,x0,y0]))*SCALE).tolist()
        if row.get('other_person_boxes_xyxy'):
            row['other_person_boxes_xyxy']=((np.asarray(old['other_person_boxes_xyxy'])-np.array([x0,y0,x0,y0]))*SCALE).tolist()
        rows.append(row)
        mapping.append(dict(frame=frame,source=old,bbox=bbox,padding=pad,crop_xyxy=list(box),scale=SCALE,
            integer_center_transform=T.tolist(),derived_wh=list(wh),projection_affine_max_abs_px=error,
            original_foreground_fraction=float((ma>0).mean()),derived_foreground_fraction=float((np.asarray(rm)>0).mean()),
            no_foreground_pixels_removed=True,unknown_labels_preserved_by_nearest=True))
        thumb=resized.copy();thumb.thumbnail((300,430))
        mthumb=rm.resize(thumb.size,Image.Resampling.NEAREST).convert('RGB')
        person=Image.fromarray((np.asarray(resized).astype(np.float32)*(np.asarray(rm)[...,None]/255.)).astype(np.uint8));person.thumbnail((300,430))
        panel=Image.new('RGB',(900,460),(35,35,35));panel.paste(thumb,(0,20));panel.paste(mthumb,(300,20));panel.paste(person,(600,20))
        from PIL import ImageDraw
        ImageDraw.Draw(panel).text((5,3),frame+' original crop2x / mask / masked target',fill='white')
        panel.save(OUT/'data'/(frame+'-input-review.png'));panels.append(panel)
    sheet=Image.new('RGB',(1800,1380),(35,35,35))
    for i,panel in enumerate(panels):sheet.paste(panel,((i%2)*900,(i//2)*460))
    sheet.save(OUT/'input-contact.png')
    document['records']=rows;document['derived_from']=str(BASE/'data/training-inputs.json')
    gs.save(OUT/'data/training-inputs.json',document)
    gs.save(OUT/'pixel-mapping.json',dict(updated=gs.now(),records=mapping,all6_projection_checks=True,
        exact_calibration_claim=False,physical_K_still_conditional=True,source_fit_unchanged=True))
    receipt=copy.deepcopy(gs.read(BASE/'smplify-gaussian-interface-receipt.json'))
    receipt.update(updated=gs.now(),fit_sha256=gs.sha(OUT/'fit/fitted.npz'),
        training_inputs_sha256=gs.sha(OUT/'data/training-inputs.json'),surface_cache_sha256=gs.sha(OUT/'fit/surface-cache.npz'),
        source_receipt=str(BASE/'smplify-gaussian-interface-receipt.json'),
        camera_input_change='Only exact crop/resize affine;3D mesh, w2c and frame IDs unchanged.')
    gs.save(OUT/'smplify-gaussian-interface-receipt.json',receipt)
    original=(BASE/'adapter.py').read_text();changed=adapt(original)
    replacements={
        'budget=opt.iterations, seconds=':'budget=LIMIT, seconds=',
        'Cross-subject frozen-SH0 configuration; inspect1000, bounded18000; cached subject fit':'Person ROI plus2x sampling control; hard stop1000; source fit reused only for input-scale comparison',
        'Gaussian options and SH0 unchanged; only reviewed fitted-input source changed':'Only deterministic person ROI plus2x input sampling and corresponding fullK; original SH0 optimizer and density unchanged',
        'indoor inputs and conservative known-pixel L1/SSIM; original SH0 optimizer and density schedule':'Indoor ROI-plus2x inputs with preserved known/unknown labels; original SH0 optimizer and density schedule',
    }
    for old,new in replacements.items():assert changed.count(old)==1,old;changed=changed.replace(old,new)
    ast.parse(changed);(OUT/'adapter.py').write_text(changed)
    (OUT/'input-interface.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),changed.splitlines(True),fromfile='frozen-indoor',tofile='person-roi2x-only')))
    plan=copy.deepcopy(baseline)
    plan.update(updated=gs.now(),state='registered_input_review_pending',subject=SUBJECT,label='室内人物裁剪＋2倍采样对照',
        record_id='008',baseline_run=str(BASE),user_authorization='当前高斯问题是重建质量过低，破碎太多，可以考虑将人像放大训练，看看问题还存在吗',
        hypothesis='Person-focused field of view and2x sampling may reduce broken visible surfaces relative to whole-frame supervision where foreground is4.8%-11.6%.',
        causal_scope='Combined ROI crop plus2x resampling input profile; cannot attribute improvement to optical resolution or either component alone.',
        crop_rule='Nonzero original person-mask bbox; symmetric ceil(0.10*max(width,height)) margin; clamp to real image bounds; no generated padding.',
        sampling='RGB bilinear, mask/known-unknown validity nearest; exact half-pixel crop-scale K. No second undistortion or K optimization.',
        fixed='Same6 frame IDs, poses, shape, reference105, world gauge, source supervision, SH0, loss definitions/weights, LR and full original density schedule.',
        comparison_budget=LIMIT,snapshots=[LIMIT],from_scratch=True,source_fitted_pose_reused_for_control=True,
        new_pose_fitting=False,optimizer_state_reused=False,gaussian_parameters_reused=False,
        performance_interface='Previously bitwise-verified GPU RGB target cache only; optional SSIM cache NOT enabled.',
        no_new_optical_detail=True,occluded_regions_excluded_from_quality=True,unknown_is_not_blank=True,
        criteria='Actual all6 visible body/face/clothing and continuous novel views; compare1000 vs baseline1000 in original native pixel coordinates. Original mesh-fold/face/20mm/300k guards retained. Stop1000 for review, no automatic18k.',
        risks=['Cropping increases foreground weight relative to geometric priors; may amplify faulty masks or pose errors.',
               'Bilinear resizing adds no detail and can change edge/SSIM behavior.',
               'Unknown rear surfaces and invisible limbs have no new evidence.'],
        peak_allocated_limit_bytes=11*2**30,preflight_budget_seconds=120,
        quality_accepted=False,end_to_end_15min_accepted=False)
    paths=[Path(__file__),gs.HERE/'cache_training_targets124.py',gs.HERE/'native_target_cache124.py',BASE/'adapter.py',BASE/'plan.json',
        OUT/'adapter.py',OUT/'pixel-mapping.json',OUT/'fit/fitted.npz',OUT/'fit/surface-cache.npz',OUT/'data/training-inputs.json',
        OUT/'smplify-gaussian-interface-receipt.json',OUT/'selection.json',OUT/'selection-rule.json',OUT/'loss_utils_validity.py']
    for row in rows:paths.extend(Path(row[key]) for key in ['image','mask','validity'])
    plan['source_hashes'].update({str(p):gs.sha(p) for p in paths})
    gs.save(OUT/'plan.json',plan);gs.save(OUT/'progress.json',dict(updated=gs.now(),state=plan['state'],step=0,pid=None))
    print('PERSON_ZOOM_PREPARED',[(r['frame'],r['wh']) for r in rows],flush=True)


def preflight(revision='preflight'):
    assert not (OUT/(revision+'-launch.json')).exists()
    plan=gs.read(OUT/'plan.json')
    for path,digest in plan['source_hashes'].items():assert gs.sha(path)==digest,path
    assert gs.read(OUT/'validity-review.json')['training_admitted']
    lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gs.save(OUT/(revision+'-launch.json'),dict(updated=gs.now(),pid=os.getpid(),command=sys.argv))
    started=time.monotonic();function=ast.parse((OUT/'adapter.py').read_text()).body[0]
    cut=next(i for i,n in enumerate(function.body) if isinstance(n,ast.FunctionDef) and n.name=='status')
    body=function.body[:cut]
    assignments=[n for n in body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='run' for t in n.targets)]
    assert len(assignments)==1;assignments[0].value=ast.parse(f"OUT/{(revision+'-only')!r}",mode='eval').body
    function.body=[*body,ast.parse('return locals()').body[0]]
    namespace=dict(gs.__dict__,OUT=OUT,LIMIT=LIMIT,suspended_baseline=lambda:None)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),str(OUT/'adapter.py')+':preflight','exec'),namespace)
    ctx=namespace['train']();model=ctx['model'];model.create_from_pcd(None,1.);model.training_setup(ctx['opt'])
    initial=[p.detach().clone() for g in model.optimizer.param_groups for p in g['params']]
    dataset=ctx['Images']();results=[];torch.cuda.reset_peak_memory_stats()
    for i,row in enumerate(ctx['rows']):
        model.optimizer.zero_grad(set_to_none=True);cam=dataset[i];model.select_mesh_by_timestep(cam.timestep)
        output=ctx['render'](cam,model,ctx['pipe'],ctx['background'])['render'];target=cam.original_image
        assert tuple(output.shape)==tuple(target.shape)==(3,row['wh'][1],row['wh'][0])
        loss=.8*ctx['l1_loss'](output,target)+.2*(1-ctx['ssim'](output,target));loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(model.static_offset.grad).all()
        results.append(dict(frame=row['frame'],shape=list(output.shape),loss=float(loss),offset_grad_norm=float(model.static_offset.grad.norm()),
            projection_error_px=ctx['records'][i]['projection_error_px']))
        assert torch.cuda.max_memory_allocated()<plan['peak_allocated_limit_bytes']
        assert time.monotonic()-started<plan['preflight_budget_seconds']
    assert not model.optimizer.state
    assert all(torch.equal(p.detach(),v) for p,v in zip([p for g in model.optimizer.param_groups for p in g['params']],initial))
    gs.save(OUT/'validity-cuda-preflight.json',dict(updated=gs.now(),passed=True,results=results,optimizer_updates=0,
        initial_points=len(model._xyz),peak_allocated_bytes=torch.cuda.max_memory_allocated(),seconds=time.monotonic()-started,
        physical_calibration_accepted=False))
    print('PERSON_ZOOM_GPU_READY',torch.cuda.max_memory_allocated(),flush=True)


def preflight_v2():
    preflight('preflight-v2')


def train():
    assert not (OUT/'launch.json').exists() and not (OUT/'gaussian-run').exists()
    assert gs.read(OUT/'validity-cuda-preflight.json')['passed']
    lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gs.save(OUT/'launch.json',dict(updated=gs.now(),pid=os.getpid(),command=sys.argv,
        proc_start_identity=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21]))
    namespace=dict(gs.__dict__,OUT=OUT,LIMIT=LIMIT,SNAPSHOTS=[LIMIT],suspended_baseline=lambda:None)
    exec(compile((OUT/'adapter.py').read_text(),str(OUT/'adapter.py'),'exec'),namespace)
    try:namespace['train']()
    except gs.ControlComplete:pass


if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except BaseException as error:
        if OUT.exists():gs.save(OUT/(sys.argv[1]+'-failure.json'),dict(updated=gs.now(),error=repr(error),traceback=traceback.format_exc(),auto_retry=False))
        raise
