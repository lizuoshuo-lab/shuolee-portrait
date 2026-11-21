"""One conditional shared-K diagnostic; original074 extrinsics remain fixed."""
import json, os, sys, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from audit_camera_conditioning117 import B, O, P, local, now, read, save, sha, fit_fresh, project

D=O/'D1-intrinsics-only'
SRC=B/'current/shared-point-bundle-convergence-101/input.npz'
MANIFEST=B/'current/avatar-repository-pair-114/smplx/data/training-inputs.json'
ARMS=['fixed074','shared_K_only']
LABELS=['face_box_lower','face_box_upper','head_height_outside_face_box','upper_body_height','lower_body_height','ceiling','floor']

def quant(v):
    a=np.asarray(v)
    assert np.isfinite(a).all()
    return dict(n=len(a),median=float(np.median(a)),p90=float(np.quantile(a,.9)),maximum=float(a.max())) if len(a) else dict(n=0)

def prepare():
    assert not D.exists(), 'No repeated candidate or overwrite'
    cond=read(O/'D1-conditioning/result.json')
    assert cond['checks']['jacobians_pass'] and not cond['checks']['invalid_point_fit_tracks']
    save(O/'D1-conditioning/review.json',dict(updated=now(),
        source_images_reviewed=['109/review/material-tracks-01.jpg','109/review/material-tracks-03.jpg'],
        source_findings=['Top-band images include low-texture ceiling and structural edges. Rounded-fixture contour tracks1779/1787shown in historicalsheet were already excluded, not reintroduced.',
            'Floor images have repeated carpet texture. Representative review cannot independently prove every physical correspondence.',
            'No mid-height observations were admitted;175frame coverage is not uniform object/depth coverage.'],
        conditioning='All fixed-extrinsic localKgroups have rank3, but pointmotion absorbs most fixedXYZ sensitivity. Ceiling/floor cy localdirections differ(-14.45px/+4.53px); directions are not calibratedK.',
        mapping='Original175K/E match114; +0.5 imagekeypoint conversion and -0.5 projection are consistent. No new pixelmapping implementation defect found.',
        decision='Allow one conditionalK-only diagnostic with unchangeddata andheldsplit, no source supplementation mixed into this comparison.',
        physical_calibration_accepted=False,root_cause_resolved=False))
    D.mkdir();(D/'review').mkdir()
    save(D/'preregistration.json',dict(created=now(),
        hypothesis='Does changing only one shared rectified pinhole intrinsic matrix improve independent background and human reprojection, conditional on original074 extrinsics?',
        sole_causal_variable='Shared f,cx,cy free in treatment. All175 original074 extrinsics fixed in both arms; XYZ nuisance points freshly optimized in both.',
        camera='SIMPLE_PINHOLE f common to fx/fy; zero skew, no new distortion. Same rectified1514x2694 images. Integer-center observations converted to COLMAPcorner by +0.5 once.',
        fitting='109 admitted originalID modulo5!=0:2127 static-background tracks and69751 observations. Whole557 held tracks and every human observation absent from both camera solves.',
        validation='Reuse frozen109 held split; outer temporal thirds triangulate each held track afresh under each arm, middle third evaluated. All101 human tracks held from calibration. Old validation is reused, not unseen physical calibration GT.',
        labels='Use110 corrected per-observation foreground-height bands while preserving facebox lower/upper labels; these are height bands, not anatomical segmentation.',
        extra_check='116 reviewed231 three-view tracks remain secondary. Fresh endpoint DLT, held middle; no point or observation selection by residual.',
        initialization='Fresh DLT plus point-only robust refinement from original2D/074K/E. No old XYZ, calibrated candidate, human or Gaussian checkpoint loaded.',
        gauge='Every original074 extrinsic fixed; original arbitrary scene scale inherited. No new pose gauge or metric-scale claim.',
        solver='Ceres SoftLOneLoss scale2px per2D residual block;100iterations,240seconds perarm,4threads; functiontol1e-6,gradienttol1e-10,paramtol1e-8. One pair, no restart or sweep.',
        gates=dict(background_median_factor=.9,background_and_each_region_p90_increase_max_px=.2,
            face_lower_median_factor=.8,face_lower_and_upper_p90_increase_max_px=0,
            other_human_height_p90_increase_max_px=.5,shared_f_relative_change_max=.05,
            principal_change_max_image_fraction=.01,converged=True,all_positive=True),
        limitation='K is conditional on estimated extrinsics; source-positive-depth selection and imperfect texture identity remain. Negative result cannot prove K correct or micromotion causal. Rectification distortion is not tested.',
        visual_frames=[23,50,75,100,125,150],visual='Same unwarped original image and identical crop per arm. Cyan observed point; red projected held point. Frame selection frozen before solving.',
        stop='OneKcandidate. If gates fail, do not start nextR/t candidate or Gaussian. Review correspondence next, without silently relaxing thresholds.',
        source_hashes={str(p):sha(p) for p in [P/'admitted-tracks.npz',SRC,MANIFEST]},code_sha256=sha(Path(__file__))))
    q=read(O/'queue-state.json');q.update(updated=now(),state='D1_K_candidate_registered',stage='D1_K',active_pid=None,
        next_action='Run solve_shared_intrinsics117.py run once in existinggvhmr environment. No other camera job or training.');save(O/'queue-state.json',q)
    print('K-only diagnostic registered',flush=True)

def solve(arm,a,xyz):
    sys.path.insert(0,'/opt/gs-slam-lab/shuolee-bundle-audit/pyceres-2.3')
    import pyceres,pycolmap
    keep=a['original_ids']%5!=0;ob=a['observations'];off=a['offsets'];E=a['w2c'];K=a['K']
    pi=np.repeat(np.arange(len(off)-1),np.diff(off));sel=keep[pi];pi=pi[sel];ob=ob[sel];fr=ob[:,0].astype(int)
    rec=pycolmap.Reconstruction()
    rec.add_camera(pycolmap.Camera(model='SIMPLE_PINHOLE',width=1514,height=2694,params=[K[0,0],K[0,2],K[1,2]],camera_id=1))
    ki=np.full(len(ob),-1,int)
    for i in range(175):
        ix=np.flatnonzero(fr==i);assert len(ix)>=20;ki[ix]=np.arange(len(ix))
        rec.add_image(pycolmap.Image(name=f'raw_{i+1:06d}.png',keypoints=ob[ix,1:]+.5,cam_from_world=pycolmap.Rigid3d(E[i,:3]),camera_id=1,id=i+1));rec.register_image(i+1)
    point_ids=np.flatnonzero(keep);offsets=np.r_[0,np.cumsum(np.diff(off)[keep])];pids=[]
    for j,lo,hi in zip(point_ids,offsets[:-1],offsets[1:]):
        ix=np.arange(lo,hi);assert (pi[ix]==j).all()
        tr=pycolmap.Track([pycolmap.TrackElement(int(fr[k])+1,int(ki[k])) for k in ix])
        pids.append(rec.add_point3D(xyz[j].copy(),tr))
    config=pycolmap.BundleAdjustmentConfig()
    for i in range(175):config.add_image(i+1);config.set_constant_cam_pose(i+1)
    if arm=='fixed074':config.set_constant_cam_intrinsics(1)
    opt=pycolmap.BundleAdjustmentOptions();opt.refine_focal_length=arm!='fixed074';opt.refine_principal_point=arm!='fixed074'
    opt.refine_extra_params=False;opt.refine_extrinsics=False;opt.loss_function_type=pycolmap.LossFunctionType.SOFT_L1;opt.loss_function_scale=2.
    opt.solver_options.function_tolerance=1e-6;opt.solver_options.gradient_tolerance=1e-10;opt.solver_options.parameter_tolerance=1e-8
    opt.solver_options.max_num_iterations=100;opt.solver_options.max_solver_time_in_seconds=240;opt.solver_options.num_threads=4;opt.solver_options.minimizer_progress_to_stdout=True
    native=pycolmap.BundleAdjuster(opt,config);loss=pyceres.SoftLOneLoss(2.);native.set_up_problem(rec,loss);problem=native.problem
    so=native.set_up_solver_options(problem,opt.solver_options);assert problem.num_residual_blocks()==69751
    save(D/(arm+'-start.json'),dict(created=now(),pid=os.getpid(),fit_tracks=len(point_ids),fit_observations=len(ob),human_or_held_observations=0,options=so.summary()))
    start=time.monotonic();summary=pyceres.SolverSummary();pyceres.solve(so,problem,summary)
    (D/(arm+'-solver.txt')).write_text(summary.FullReport(),encoding='utf-8')
    outE=np.tile(np.eye(4),(175,1,1))
    for i in range(175):outE[i,:3]=rec.images[i+1].cam_from_world.matrix()
    assert np.max(abs(outE-E))<1e-10
    f,cx,cy=rec.cameras[1].params;newK=np.array([[f,0,cx],[0,f,cy],[0,0,1.]])
    if arm=='fixed074':assert np.array_equal(newK,K)
    outX=np.array([rec.points3D[j].xyz for j in pids]);positive=True;errs=[]
    for m,j in enumerate(point_ids):
        o=a['observations'][off[j]:off[j+1]];pred,v=project(outX[m],E[o[:,0].astype(int)],newK)
        positive=positive and bool((v[:,2]>0).all());errs.extend(np.linalg.norm(pred-o[:,1:],axis=1))
    result=dict(updated=now(),arm=arm,seconds=time.monotonic()-start,K=newK.tolist(),termination=str(summary.termination_type),
        usable=summary.IsSolutionUsable(),message=summary.message,fit_error=quant(errs),fit_positive=positive,
        actual_extrinsic_max_change=float(np.max(abs(outE-E))),relative_f_change=float(f/K[0,0]-1),principal_shift_px=[float(cx-K[0,2]),float(cy-K[1,2])],source_cameras_changed=False,gaussian_training=False)
    np.savez_compressed(D/(arm+'.npz'),K=newK,w2c=outE,xyz=outX,original_track_ids=a['original_ids'][keep])
    save(D/(arm+'-result.json'),result);print(json.dumps(result),flush=True)
    return newK,result

def evaluate(a,matrices,native):
    b=np.load(SRC);rows=read(MANIFEST)['records'];bounds=[]
    for r in rows:
        mask=np.array(Image.open(local(r['mask'])).convert('L'))>127;yy=np.flatnonzero(mask.any(1));bounds.append((int(yy[0]),int(yy[-1])))
    def label(g,frame,y):
        if g in (1,2):return LABELS[g-1]
        if g==0:return 'background_not_used'
        lo,hi=bounds[frame];v=(y-lo)/max(hi-lo,1)
        return LABELS[2] if v<.24 else LABELS[3] if v<.62 else LABELS[4]
    validation=[];off=a['offsets'];fit_original=set(a['original_ids'][a['original_ids']%5!=0].tolist())
    for j,(lo,hi) in enumerate(zip(off[:-1],off[1:])):
        if a['original_ids'][j]%5:continue
        ob=a['observations'][lo:hi];h=np.zeros(len(ob),bool);h[len(ob)//3:2*len(ob)//3]=True
        validation.append(dict(id=int(a['original_ids'][j]),obs=ob,held=h,labels=[LABELS[5+int(a['regions'][j])]]*len(ob)))
    for j,(lo,hi) in enumerate(zip(b['offsets'][:-1],b['offsets'][1:])):
        g=b['groups'][lo:hi]
        if not (g>0).any():continue
        assert int(b['original_track_ids'][j]) not in fit_original
        ob=b['observations'][lo:hi]
        validation.append(dict(id=int(b['original_track_ids'][j]),obs=ob,held=b['held'][lo:hi],labels=[label(int(v),int(o[0]),o[2]) for v,o in zip(g,ob)]))
    outputs={};detail={};E=a['w2c']
    for arm,K in matrices.items():
        errs={k:[] for k in LABELS};meds={k:[] for k in LABELS};records=[];invalid=[]
        for t in validation:
            o=t['obs'];h=t['held'];fr=o[:,0].astype(int);X,ok=fit_fresh(o[~h,1:],E[fr[~h]],K);pred,v=project(X,E[fr],K)
            ee=np.linalg.norm(pred-o[:,1:],axis=1);labels=np.array(t['labels'])
            if not ok or not np.isfinite(ee).all() or not(v[:,2]>0).all():invalid.append(t['id'])
            for name in LABELS:
                sel=h&(labels==name)
                if sel.any():errs[name].extend(ee[sel]);meds[name].append([t['id'],float(np.median(ee[sel]))])
            records.append(dict(id=t['id'],frames=(fr[h]+1).tolist(),observed=o[h,1:].tolist(),projected=pred[h].tolist(),labels=labels[h].tolist(),error=ee[h].tolist(),valid=t['id'] not in invalid))
        outputs[arm]={k:dict(**quant(errs[k]),tracks=len(meds[k])) for k in LABELS}
        outputs[arm]['background']=dict(**quant(errs['ceiling']+errs['floor']),tracks=len(meds['ceiling'])+len(meds['floor']))
        outputs[arm]['invalid_tracks']=invalid;detail[arm]=dict(records=records,medians=meds)
    rng=np.random.default_rng(117);uncertainty={}
    for name in LABELS:
        x=np.asarray(detail[ARMS[0]]['medians'][name]);y=np.asarray(detail[ARMS[1]]['medians'][name]);assert np.array_equal(x[:,0],y[:,0])
        delta=y[:,1]-x[:,1];ci=np.quantile(np.median(rng.choice(delta,(1000,len(delta)),replace=True),axis=1),[.025,.975])
        uncertainty[name]=dict(tracks=len(delta),median_paired_track_delta_px=float(np.median(delta)),bootstrap95=ci.tolist())
    ctrl=outputs[ARMS[0]];p=outputs[ARMS[1]];n=native[ARMS[1]]
    gates=dict(background_median=p['background']['median']<=.9*ctrl['background']['median'],
        background_tails=all(p[g]['p90']<=ctrl[g]['p90']+.2 for g in ['background','ceiling','floor']),
        lower_face_median=p['face_box_lower']['median']<=.8*ctrl['face_box_lower']['median'],
        face_tails=all(p[g]['p90']<=ctrl[g]['p90'] for g in LABELS[:2]),
        other_body_tails=all(p[g]['p90']<=ctrl[g]['p90']+.5 for g in LABELS[2:5]),
        small_K_change=abs(n['relative_f_change'])<=.05 and np.all(abs(np.array(n['principal_shift_px']))<=.01*np.array([1514,2694])),
        converged=all(v['termination']=='TerminationType.CONVERGENCE' for v in native.values()),
        positive=all(not outputs[k]['invalid_tracks'] and native[k]['fit_positive'] for k in ARMS),
        all175extrinsics_unchanged=all(v['actual_extrinsic_max_change']<1e-10 for v in native.values()))
    result=dict(updated=now(),arms=outputs,paired_track_uncertainty=uncertainty,gates=gates,numerical_gates_pass=bool(all(gates.values())),
        quality_accepted=False,source_cameras_changed=False,gaussian_training_started=False,root_cause_resolved=False,
        note='Conditional shared-K test, reused held measurements. Track bootstrap does not make nearby texture tracks independent. Visual/material review still required.')
    save(D/'evaluation.json',result);save(D/'evaluation-tracks.json',detail)
    return result,detail,rows

def secondary(matrices,E):
    import cv2
    J=B/'current/portrait-alignment-open-source-116/hsfm/joint-diagnostic-reserved';v=np.load(J/'frozen-validation.npz')
    out={};raw={}
    for arm,K in matrices.items():
        KK=K.copy();KK[:2,2]-=.5;err=[];bad=[]
        for j,(fr,xy) in enumerate(zip(v['frames'],v['xy'])):
            f=np.asarray(fr,int)-1;PP=np.array([KK@E[i,:3] for i in f]);xh=cv2.triangulatePoints(PP[0],PP[2],xy[0].reshape(2,1),xy[2].reshape(2,1))[:,0];xh/=xh[3]
            if not np.isfinite(xh).all() or any((E[i]@xh)[2]<=0 for i in f):bad.append(j)
            p=PP[1]@xh;err.append(float(np.linalg.norm(p[:2]/p[2]-xy[1])))
        raw[arm]=np.asarray(err);out[arm]={name:quant(np.array(err)[v['groups']==g]) for g,name in enumerate(['background','face_lower','face_upper','torso','legs_shoes'])};out[arm]['invalid']=bad
    prior=np.load(J/'native-fit-cached/evaluation/original074-frozen-errors.npz')['errors']
    assert np.max(abs(raw['fixed074']-prior))<1e-7
    save(D/'secondary116.json',dict(updated=now(),arms=out,baseline_reproduced=True,scope='Reused231reviewedtracks; descriptive secondary check; no fitting or parameter selection.'))

def visuals(detail,rows):
    font=ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf',18)
    lookup={}
    for arm,dd in detail.items():
        for r in dd['records']:
            for f,obs,pred,label in zip(r['frames'],r['observed'],r['projected'],r['labels']):lookup.setdefault((arm,f),[]).append((obs,pred,label))
    for f in read(D/'preregistration.json')['visual_frames']:
        src=Image.open(local(rows[f-1]['image'])).convert('RGB');assert src.size==(1514,2694)
        mask=np.array(Image.open(local(rows[f-1]['mask'])).convert('L'))>127;yy,xx=np.where(mask)
        face=[o for o,p,g in lookup.get(('fixed074',f),[]) if g in LABELS[:2]]
        if face:
            center=np.mean(face,axis=0);cx,cy=center
        else:cx=(xx.min()+xx.max())/2;cy=yy.min()+.12*(yy.max()-yy.min())
        x0=int(np.clip(cx-250,0,1514-500));y0=int(np.clip(cy-250,0,2694-500));box=(x0,y0,x0+500,y0+500)
        board=Image.new('RGB',(1512,1450),'white');draw=ImageDraw.Draw(board)
        for col,arm in enumerate(['original']+ARMS):
            im=src.copy();dr=ImageDraw.Draw(im)
            for obs,pred,g in lookup.get(('fixed074' if arm=='original' else arm,f),[]):
                x,y=obs;dr.ellipse((x-3,y-3,x+3,y+3),outline=(0,220,230),width=2)
                if arm!='original':
                    u,v=pred;dr.line((x,y,u,v),fill=(255,40,60),width=2);dr.line((u-4,v,u+4,v),fill=(255,40,60),width=2);dr.line((u,v-4,u,v+4),fill=(255,40,60),width=2)
            draw.text((col*504+3,5),f'{f:06d} | {arm}',fill='black',font=font)
            board.paste(im.resize((504,897),Image.Resampling.LANCZOS),(col*504,30));board.paste(im.crop(box),(col*504,945))
        draw.text((4,925),'Same original image/crop | cyan: observed | red: held projection | no image warp',fill='black',font=font)
        board.save(D/f'review/frame-{f:06d}.jpg',quality=95)

def run():
    assert (D/'preregistration.json').exists() and not (D/'start.json').exists()
    pr=read(D/'preregistration.json');assert all(sha(local(p))==h for p,h in pr['source_hashes'].items())
    assert sha(Path(__file__))==pr['code_sha256']
    save(D/'start.json',dict(created=now(),pid=os.getpid(),argv=sys.argv))
    q=read(O/'queue-state.json');q.update(updated=now(),state='D1_K_running',active_pid=os.getpid(),active_log='D1-intrinsics-only/stdout.log');save(O/'queue-state.json',q)
    t=time.monotonic()
    with np.load(P/'admitted-tracks.npz') as z:a={k:z[k] for k in z.files}
    assert a['K'][0,0]==a['K'][1,1]
    xyz=[];invalid=[]
    for j,(lo,hi) in enumerate(zip(a['offsets'][:-1],a['offsets'][1:])):
        o=a['observations'][lo:hi];E=a['w2c'][o[:,0].astype(int)];x,ok=fit_fresh(o[:,1:],E,a['K']);_,v=project(x,E,a['K']);xyz.append(x)
        if not ok or not np.isfinite(x).all() or not (v[:,2]>0).all():invalid.append(j)
    assert not invalid
    native={};matrices={}
    for arm in ARMS:matrices[arm],native[arm]=solve(arm,a,np.array(xyz))
    result,detail,rows=evaluate(a,matrices,native);secondary(matrices,a['w2c']);visuals(detail,rows)
    result.update(seconds=time.monotonic()-t,sources_unchanged=all(sha(local(p))==h for p,h in pr['source_hashes'].items()),state='completed_visual_review_pending')
    assert result['sources_unchanged'];save(D/'result.json',result)
    q=read(O/'queue-state.json');q.update(updated=now(),state='D1_K_review_pending',active_pid=None,stage='D1_review',
        next_action='Actually inspect fixed source overlays andheldstats. Do not promote candidate or runR/t/Gaussian ifgatesfail. Publish diagnostic only; correspondence audit remains separate.',result='D1-intrinsics-only/result.json');save(O/'queue-state.json',q)
    print(json.dumps(dict(state=result['state'],gates=result['gates'],seconds=result['seconds'],K=native['shared_K_only']['K'])),flush=True)

if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ['prepare','run']
    try:globals()[sys.argv[1]]()
    except Exception as e:
        if sys.argv[1]=='run' and (D/'start.json').exists():
            import traceback
            save(D/'failure.json',dict(updated=now(),pid=os.getpid(),error=repr(e),traceback=traceback.format_exc(),source_cameras_changed=False,training_started=False))
            q=read(O/'queue-state.json');q.update(updated=now(),state='D1_K_failed_inspect_evidence',active_pid=None,next_action='Inspect D1-intrinsics-only/failure.json; do not duplicate or restart an already completed arm.');save(O/'queue-state.json',q)
        raise
