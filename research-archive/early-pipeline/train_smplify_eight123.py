"""One deterministic eight-image training subset; all59 poses remain reviewable.

Select without inspecting Gaussian losses. Reuse the existing SMPLify fits,
native observations and original pixels; change only the training image roster.
"""
from pathlib import Path
import argparse, ast, copy, difflib, shutil, traceback
import numpy as np
from PIL import Image
import train_smplify_direct_gaussian123 as direct
from run_official_gaussians123 import read,save,sha,now

BASE=direct.OUT
OUT=BASE.parent/'body-gaussian-smplify-eight'
BODY=np.array([0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11])


def source():
    _,_,old=direct.source();new=old
    def replace(a,b):
        nonlocal new
        assert new.count(a)==1,a
        new=new.replace(a,b,1)
    replace('    counts = np.zeros(59, dtype=int)', """    selected=read(OUT/'selection.json')['selected_frames']
    training_slots=[i for i,r in enumerate(rows) if r['frame'] in selected]
    assert len(training_slots)==8
    counts = np.zeros(59, dtype=int)""")
    replace('        def __len__(self): return len(rows)', '        def __len__(self): return len(training_slots)')
    replace('        def __getitem__(self, i):', '        def __getitem__(self, i):\n            i=training_slots[i]')
    replace('direct_smplify_route=True', 'direct_smplify_route=True, training_frames=8, review_frames=59')
    replace("stop_reason='Preregistered1000-step direct SMPLify-X comparison budget'", "stop_reason='Preregistered1000-step eight-image roster comparison'")
    ns=dict(direct.source()[0].__globals__);ns['OUT']=OUT
    exec(compile(new,'<eight-image-roster-original-Gaussian-training>','exec'),ns)
    return ns['train'],old,new


def rank(values):
    a=np.asarray(values,dtype=float)
    return np.array([(np.count_nonzero(a<v)+.5*np.count_nonzero(a==v))/len(a) for v in a])


def prepare():
    assert not OUT.exists(),'No duplicate subset experiment.'
    p=copy.deepcopy(read(BASE/'plan.json'))
    for f,h in p['source_hashes'].items():assert sha(f)==h,f
    OUT.mkdir()
    # Rule is written before reading/evaluating per-frame scores.
    rule=dict(updated=now(),selection_count=8,selection_is_heuristic=True,
        observations='Existing SMPLify-X2D fit observations only; no Gaussian results used in selection.',
        coverage='Eight equal-population bins of unwrapped fitted body yaw; best quality rank in each bin. Fitted yaw is a view proxy, not independent camera ground truth.',
        score='0.45 rank(body weighted reprojection RMS)+0.25 rank(face weighted reprojection RMS)+0.20 rank(-native face sharpness)+0.10 rank(-body observation confidence). Lower is better; tie by frame id.',
        no_manual_frame_replacement=True,no_new_fit_or_detector=True,no_threshold_sweep=True,
        uncertainty='Detector confidence is a relative weight, not visibility. Head and body may face differently. Occluded hands/back cannot be validated from this score.',
        preservation='Other51 images remain excluded from Gaussian RGB optimization and are all rendered for checks. Their SMPLify fits were computed previously, so they are not independent physical ground truth.',
        budget=1000,from_scratch=True,optimizer_empty=True,parameters='Original direct59 parameters unchanged',
        fifteen_minute_acceptance=False)
    save(OUT/'selection-rule.json',rule)
    import cv2
    cv2.setNumThreads(1)
    with np.load(BASE/'fit/fitted.npz',allow_pickle=False) as z:fit=dict(z)
    rows=read(BASE/'data/training-inputs.json')['records']
    assert len(rows)==59
    obs=fit['observed_native_keypoints'];projected=fit['joints_projected_native']
    residual=np.linalg.norm(projected-obs[:,:,:2],axis=-1)
    chronological=np.argsort([r['time_seconds'] for r in rows],kind='stable')
    rotations=fit['root_rotation']
    yaw0=np.arctan2(rotations[:,0,2],rotations[:,2,2]);yaw=np.empty(59)
    yaw[chronological]=np.unwrap(yaw0[chronological])
    records=[]
    for i,r in enumerate(rows):
        assert sha(r['image'])==r['source_sha256'] and sha(r['mask'])==r['mask_sha256']
        raw=np.asarray(Image.open(r['image']).convert('RGB'))
        box=np.asarray(r['face_box_xyxy']).astype(int)
        box[[0,2]]=np.clip(box[[0,2]],0,raw.shape[1]);box[[1,3]]=np.clip(box[[1,3]],0,raw.shape[0])
        x0,y0,x1,y1=box;assert x1>x0 and y1>y0
        gray=cv2.cvtColor(raw[y0:y1,x0:x1],cv2.COLOR_RGB2GRAY)
        sharpness=float(cv2.Laplacian(gray,cv2.CV_64F).var())
        metrics={}
        for label,indices in [('body',BODY),('face',np.arange(67,118))]:
            weights=obs[i,indices,2]**2
            assert float(weights.sum())>0
            metrics[label+'_weighted_rms_px']=float(np.sqrt(np.sum(weights*residual[i,indices]**2)/weights.sum()))
        records.append(dict(frame=r['frame'],slot=i,time_seconds=r['time_seconds'],yaw_proxy_deg=float(np.degrees(yaw[i])),
            face_sharpness=sharpness,body_confidence=float(obs[i,BODY,2].mean()),face_box_xyxy=box.tolist(),**metrics))
    score=.45*rank([r['body_weighted_rms_px'] for r in records])+.25*rank([r['face_weighted_rms_px'] for r in records])
    score+=.2*rank([-r['face_sharpness'] for r in records])+.1*rank([-r['body_confidence'] for r in records])
    for i,r in enumerate(records):r['quality_score']=float(score[i])
    order=sorted(range(59),key=lambda i:(yaw[i],rows[i]['frame']))
    selected=[]
    for b,group in enumerate(np.array_split(order,8)):
        best=min(map(int,group),key=lambda i:(score[i],rows[i]['frame']))
        selected.append(best)
        for i in group:records[int(i)]['coverage_bin']=b;records[int(i)]['selected']=int(i)==best
    names=[rows[i]['frame'] for i in sorted(selected,key=lambda i:rows[i]['time_seconds'])]
    save(OUT/'selection.json',dict(updated=now(),selected_frames=names,records=records,
        training_count=8,excluded_from_Gaussian_RGB_count=51,all_frame_evaluation=59,
        fitted_yaw_range_deg=[float(np.degrees(yaw.min())),float(np.degrees(yaw.max()))],
        selection_rule_sha256=sha(OUT/'selection-rule.json'),quality_accepted=False))
    for name in ['fit/fitted.npz','fit/surface-cache.npz','data/training-inputs.json','smplify-gaussian-interface-receipt.json']:
        target=OUT/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/name,target)
        assert sha(target)==sha(BASE/name)
    # Same59 geometry cache, K and reference33; only8 images are sampled by the loss.
    _,old,new=source();ast.parse(new)
    (OUT/'roster-only.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True))))
    (OUT/'eight-image-adapter.py').write_text(new)
    p.update(updated=now(),state='prepared',user_authorization=['不需要那么多帧，搞8个左右最好的帧训练'],
        baseline_run=str(BASE),training_frames=8,evaluation_frames=59,selected_frames=names,
        only_parameter_change='Training image roster59 -> deterministic8; same SMPLify meshes, Gaussian optimizer/loss/density settings, zero shared clothing offset, empty optimizer.',
        from_scratch=True,optimizer_state_reused=False,old_gaussian_checkpoint_loaded=False,
        no_automatic_new_experiment=True,quality_accepted=False,end_to_end_15min_accepted=False,
        admission='Review8 selected original/SMPLify images before launch. At1000 compare8 optimized images and51 excluded images separately against the frozen59-frame1000 control.')
    p['source_hashes'].update({str(f):sha(f) for f in [Path(__file__).resolve(),OUT/'selection.json',OUT/'selection-rule.json',
        OUT/'fit/fitted.npz',OUT/'fit/surface-cache.npz',OUT/'data/training-inputs.json',OUT/'smplify-gaussian-interface-receipt.json']})
    save(OUT/'plan.json',p);save(OUT/'progress.json',dict(updated=now(),state='prepared',pid=None,step=0,quality_accepted=False))
    # Existing native raw/SMPLify panels: no rendered Gaussian data influences selection.
    reviewed=read(BASE/'gaussian-run/step-001000/review/result.json')
    figures=[]
    for name in names:
        item=next(x for x in reviewed['records'] if x['frame']==name)
        f=next(x for x in item['images'] if x['region']=='body')
        im=Image.open(f['path']).convert('RGB');w=im.width//3
        im=im.crop((0,0,w*2,im.height));im.save(OUT/(name+'-selection.png'))
        scale=min(1.,300/im.height);im=im.resize((round(im.width*scale),round(im.height*scale)),Image.Resampling.LANCZOS);figures.append(im)
    w=max(i.width for i in figures);h=max(i.height for i in figures)
    contact=Image.new('RGB',(w*4,h*2),'white')
    for i,im in enumerate(figures):contact.paste(im,((i%4)*w,(i//4)*h))
    contact.save(OUT/'selected-eight.png')
    save(OUT/'previous-continuation-superseded.json',dict(updated=now(),previous_launcher='continue_smplify_direct123.py',
        previous_continuation_started=False,reason='Latest user changed the training roster to about8; no59-frame continuation launched.'))
    print('EIGHT_SELECTED',names,flush=True)
    print('YAW_PROXY_RANGE',float(np.degrees(yaw.min())),float(np.degrees(yaw.max())),flush=True)
    print('SELECTED_METRICS',[(r['frame'],r['body_weighted_rms_px'],r['face_weighted_rms_px']) for r in records if r['selected']],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','run','train']);ap.add_argument('--pid',type=int);args=ap.parse_args()
    try:
        if args.mode=='prepare':prepare()
        elif args.mode=='run':direct.control.OUT=OUT;direct.control.__file__=__file__;direct.control.run(args.pid)
        else:source()[0]()
    except BaseException as e:
        if OUT.exists():
            save(OUT/(args.mode+'-failure.json'),dict(updated=now(),error=repr(e),traceback=traceback.format_exc(),auto_retry=False))
            save(OUT/'progress.json',dict(updated=now(),state='failed_preserved',pid=None,error=repr(e),quality_accepted=False))
        raise
