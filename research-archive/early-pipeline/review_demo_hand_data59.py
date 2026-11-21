"""Read completed frames of the approved uniform ablation, without touching fitting.

Every completed frame is included. Crops and evaluation observations are frozen
from the baseline. This script never imports torch or launches another fit.
"""
from pathlib import Path
import json
import os
import sys
import time
import numpy as np
from PIL import Image
from review_full_sequence123 import mesh_renderer, joined, read, sha

H = Path(__file__).resolve().parent
B = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/demo-vhap-initialization/fullbody-smplifyx')
DATA = Path('/opt/gs-slam-lab/shuolee-local-surface-123/demo-vhap-fps3')
RUN = B/'hand-data-control59'
OUT = RUN/'comparison'
RECORD = Path('/mnt/d/00/记录/demo')
GROUPS = dict(body15=[0,16,15,18,17,5,2,6,3,7,4,13,10,14,11],
              face51=list(range(67,118)), feet6=list(range(19,25)),
              left_hand21=list(range(25,46)), right_hand21=list(range(46,67)))


def save(path, data):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(tmp, path)


def stats(values):
    x = np.asarray(values)
    return dict(count=int(x.size), mean=float(x.mean()), median=float(np.median(x)),
                p90=float(np.quantile(x,.9)), maximum=float(x.max())) if x.size else dict(count=0)


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}


def project(d):
    j=d['joints_camera'][0].astype(np.float64)
    assert np.isfinite(j).all() and (j[:,2]>0).all()
    q=j@d['K_integer_centers'].T
    uv=q[:,:2]/q[:,2:]
    # Saved CUDA projection uses float32; this read-only check uses float64.
    # Bound arithmetic roundoff separately from detector/physical pixel error.
    saved=d['joints_projected_native'][0]
    roundoff=16*np.finfo(saved.dtype).eps*max(1.,float(np.abs(saved).max()))
    assert np.max(np.abs(uv-saved))<=roundoff
    return uv


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import cv2
    cv2.setNumThreads(1)
    plan=read(B/'hand-data-control-plan.json')
    assert plan['execution_authorized'] and plan['chronological_roster']==[f'frame_{i:06d}' for i in range(1,60)]
    assert sha(H/'proposals/demo-fullbody-hand-data/run_hand_data_control.py')==plan['wrapper_sha256']
    baseline_receipt=read(B/'full-sequence/fit/official-fit-result.json')
    baseline_path=B/'full-sequence/fit/fitted.npz'
    assert sha(baseline_path)==baseline_receipt['fit_sha256']
    baseline=load(baseline_path)
    slots={str(n):i for i,n in enumerate(baseline['frame_ids'])}
    rows={r['frame']:r for r in read(DATA/'data/inputs.json')['records']}
    assert len(rows)==59 and set(rows)==set(slots)==set(plan['chronological_roster'])
    render, render_source=mesh_renderer()
    OUT.mkdir(exist_ok=True)
    evidence=[];began=time.monotonic()
    for name in plan['chronological_roster']:
        frame=RUN/'frames'/name
        if not (frame/'result.json').exists(): continue
        fit_report=read(frame/'result.json')
        assert fit_report['state']=='complete' and fit_report['frame_id']==name
        control=fit_report['single_variable_control']
        assert control['candidate']==[0,0,0,0,0] and control['source_sha256']==plan['wrapper_sha256']
        source=frame/'winner-camera-space.npz'; digest=sha(source)
        review_path=OUT/(name+'.json')
        if review_path.exists():
            previous=read(review_path)
            assert previous['candidate_sha256']==digest
            assert all(sha(x['record_path'])==x['sha256'] for x in previous['images'])
            evidence.append(previous); continue
        current=load(source); old_path=B/'full-sequence/frames'/name/'winner-camera-space.npz'; old=load(old_path)
        row=rows[name]; i=slots[name]; target=baseline['observed_native_keypoints'][i]
        for key in ['K_integer_centers','w2c','source_w2c','betas','faces','frame_id','source_frame','image_sha256','mask_sha256']:
            assert np.array_equal(old[key],current[key]), (name,key)
        assert sha(row['image'])==row['source_sha256'] and sha(row['mask'])==row['mask_sha256']
        assert np.array_equal(current['K_integer_centers'],row['K_integer_centers'])
        old_uv,new_uv=project(old),project(current)
        errors=[np.linalg.norm(uv-target[:,:2],axis=1) for uv in (old_uv,new_uv)]
        confidence=target[:,2]; active=confidence>0; active[[1,9,12]]=False
        groups={}
        for group,indices in GROUPS.items():
            ids=np.asarray(indices); ids=ids[active[ids]]
            groups[group]=dict(baseline=stats(errors[0][ids]),candidate=stats(errors[1][ids]))
            if len(ids): groups[group]['median_delta_px']=float(np.median(errors[1][ids])-np.median(errors[0][ids]))
        old_review=read(B/'full-sequence/review'/(name+'_saved-review.json'))
        # Independent arithmetic check against the existing baseline review.
        assert np.allclose(errors[0],[p['error_px'] for p in old_review['per_joint']],rtol=0,atol=1e-7)
        raw=Image.open(row['image']).convert('RGB')
        assert list(raw.size)==row['wh']
        overlays=[render(raw,np.array([0,0,*raw.size]),d) for d in (old,current)]
        images=[]
        for region in ('body','face','left_hand','right_hand','feet'):
            item=next(x for x in old_review['files'] if x['region']==region)
            box=item['crop_integer_xyxy']
            panels=[im.crop(box) for im in [raw,*overlays]]
            dest=RECORD/f'SMPL-X_手部数据统一对照_{name}_{region}_原图_原版_关闭手部2D.png'
            assert not dest.exists(), 'Preserve existing record image: '+str(dest)
            joined(panels).save(dest)
            with Image.open(dest) as check:
                assert np.array_equal(np.asarray(check)[:,:panels[0].width],np.asarray(panels[0]))
            images.append(dict(region=region,record_path=str(dest),sha256=sha(dest),crop_integer_xyxy=box,
                               panel_wh=list(panels[0].size),panels=['raw','baseline','hand_data_off'],resampling=False,text_added=False))
        per_joint=[dict(index=j,confidence=float(confidence[j]),evaluation_active=bool(active[j]),
                        candidate_image_term_active=bool(active[j] and not 25<=j<67),
                        baseline_px=float(errors[0][j]),candidate_px=float(errors[1][j]),delta_px=float(errors[1][j]-errors[0][j])) for j in range(118)]
        result=dict(frame_id=name,source_frame=row['source_frame_index'],candidate_sha256=digest,baseline_sha256=sha(old_path),
                    fit_seconds=fit_report['elapsed_seconds'],groups=groups,per_joint=per_joint,images=images,
                    left_elbow_baseline_px=float(errors[0][6]),left_elbow_candidate_px=float(errors[1][6]),
                    source_image_sha256=row['source_sha256'],renderer_sha256=sha(render_source),quality_accepted=False)
        assert sha(source)==digest
        save(review_path,result); evidence.append(result)
        print(json.dumps(dict(frame=name,reviewed=len(evidence),elbow=[float(errors[0][6]),float(errors[1][6])])),flush=True)
    progress=read(RUN/'progress.json')
    complete=progress['state']=='complete' and len(evidence)==59
    if complete:
        final_receipt=read(RUN/'fit/official-fit-result.json')
        assert final_receipt['control']['all_frames_included'] and final_receipt['frames']==59
        assert final_receipt['fit_sha256']==sha(RUN/'fit/fitted.npz')
        final=load(RUN/'fit/fitted.npz')
        assert np.array_equal(final['observed_native_keypoints'],baseline['observed_native_keypoints'])
        assert np.array_equal(final['frame_ids'],baseline['frame_ids'])
    summary=dict(state='complete_visual_review_pending' if complete else 'partial_completed_frames_only',
                 frames=len(evidence),expected_frames=59,whole_sequence_complete=complete,progress_snapshot=progress,
                 elapsed_review_seconds=time.monotonic()-began,baseline_fit_seconds=baseline_receipt['elapsed_seconds'],
                 candidate_fit_seconds=progress.get('elapsed_seconds'),per_frame=evidence,
                 elbow_over50_baseline=[r['frame_id'] for r in evidence if r['left_elbow_baseline_px']>50],
                 elbow_over50_candidate=[r['frame_id'] for r in evidence if r['left_elbow_candidate_px']>50],
                 regression_reporting='Every frame retained, including both hands. No frame or joint removed by residual.',
                 detector_residual_not_independent_ground_truth=True,actual_visual_review=False,quality_accepted=False,
                 gaussian_training_started=False,source_sha256=sha(__file__),plan_sha256=sha(B/'hand-data-control-plan.json'))
    assert not any(k=='torch' or k.startswith('torch.') for k in sys.modules)
    save(OUT/'result.json',summary)
    print(json.dumps({k:summary[k] for k in ('state','frames','elbow_over50_baseline','elbow_over50_candidate')}),flush=True)


if __name__=='__main__':main()
