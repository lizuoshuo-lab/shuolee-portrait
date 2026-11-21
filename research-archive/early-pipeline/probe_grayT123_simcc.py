"""Isolated decoder control. Never writes observations, cameras or geometry.

The optional log-response refinement follows MMPose v1.3.2 SimCC DARK.
This is an independently arranged diagnostic implementation, not a claim that
enabling DARK is correct for this subject or that semantic points are truth.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, time
import numpy as np
import cv2
from PIL import Image, ImageDraw

H = Path('/mnt/c/Users/Administrator/Documents/New project/shuolee-slide-update')
REC = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
OUT = REC/'grayT-simcc-decoder-review'
ROOT = Path('/opt/gs-slam-lab/shuolee-local-surface-123/grayT')
ASSETS = Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets')
SIZE = np.array([288,384])
KERNELS = (37,43)

def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d): p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def stats(a):
    a=np.asarray(a,float).reshape(-1)
    return dict(n=len(a),median=float(np.median(a)),p90=float(np.quantile(a,.9)),maximum=float(a.max())) if len(a) else dict(n=0)

def refine_axis(responses,kernel):
    """Return candidate bin locations and explicit failure flags; no fallback."""
    assert responses.ndim==3 and kernel in KERNELS
    n,k,length=responses.shape
    result=np.full((n,k),np.nan,np.float64); flags=[]
    # Preserve input response bytes. MMPose blurs zero-padded response, restores
    # its peak amplitude, then makes a finite-difference log-peak correction.
    for bi in range(n):
        for ji in range(k):
            raw=responses[bi,ji]
            reason=[]
            if not np.isfinite(raw).all() or np.max(raw)<=0:
                flags.append(dict(batch=bi,joint=ji,valid=False,reasons=['nonpositive_or_nonfinite_response']))
                continue
            p=int(np.argmax(raw)); peak=float(raw[p]); border=(kernel-1)//2
            if p<2 or p>length-3: reason.append('boundary_peak')
            maxima=np.flatnonzero((raw[1:-1]>=raw[:-2])&(raw[1:-1]>raw[2:]))+1
            competing=[int(i) for i in maxima if abs(int(i)-p)>border and raw[i]>=.98*peak]
            if competing: reason.append('comparable_separated_peak')
            padded=np.pad(raw.astype(np.float32),(border,border),mode='constant')[None]
            blurred=cv2.GaussianBlur(padded,(kernel,1),0)[0,border:-border]
            if not np.isfinite(blurred).all() or blurred.max()<=0:
                reason.append('invalid_blur'); flags.append(dict(batch=bi,joint=ji,valid=False,reasons=reason)); continue
            blurred*=peak/float(blurred.max())
            log=np.pad(np.log(np.clip(blurred,1e-3,50.)),(2,2),mode='edge')
            q=p+2
            first=.5*float(log[q+1]-log[q-1])
            second=1e-9+.25*float(log[q+2]-2*log[q]+log[q-2])
            if second>=-1e-7: reason.append('nonconcave_or_flat_peak')
            offset=first/second if abs(second)>1e-12 else float('nan')
            if not np.isfinite(offset) or abs(offset)>2: reason.append('unstable_local_offset')
            candidate=p-offset
            if np.isfinite(candidate): result[bi,ji]=candidate
            flags.append(dict(batch=bi,joint=ji,valid=not reason,reasons=reason,
                argmax_bin=p,continuous_bin=float(candidate) if np.isfinite(candidate) else None,
                offset_bins=float(offset) if np.isfinite(offset) else None,curvature=second,competing_peaks=competing))
    return result,flags

def controls():
    from rtmlib.tools.pose_estimation.pre_processings import get_warp_matrix
    target=OUT/'controls.json'
    if target.exists(): raise RuntimeError('Controls already ran; inspect saved result.')
    read(OUT/'plan.json')
    checks=[]
    for length,sigma,kernel in [(576,6.,37),(768,6.93,43)]:
        centers=np.array([40.12,80.37,160.71,length*.5+.23,length-40.44])
        raw=np.exp(-.5*((np.arange(length)[None]-centers[:,None])/sigma)**2).astype(np.float32)[None]
        before=raw.tobytes(); decoded,flags=refine_axis(raw,kernel)
        error=np.abs(decoded[0]-centers)
        assert before==raw.tobytes()
        assert all(f['valid'] for f in flags) and error.max()<.02
        checks.append(dict(name=f'known_continuous_peaks_{length}',passed=True,max_error_bins=float(error.max()),
            argmax_max_error_bins=float(np.abs(raw.argmax(-1)[0]-centers).max())))
        # Counterexamples remain present and explicitly fail, never deleted.
        mix=np.exp(-.5*((np.arange(length)-128.)/sigma)**2)+np.exp(-.5*((np.arange(length)-240.)/sigma)**2)
        cases=[np.zeros(length),np.ones(length),mix,np.exp(-.5*((np.arange(length)-.1)/sigma)**2)]
        cases.append(np.full(length,np.nan))
        _,fl=refine_axis(np.asarray(cases,np.float32)[None],kernel)
        assert all(not f['valid'] for f in fl)
        checks.append(dict(name=f'invalid_responses_{length}',passed=True,flags=fl))
    maxerr=0.;mirrorerr=0.
    for center,scale in [(np.array([814.31,1422.97]),np.array([1626.53,2168.7066666667])),
                         (np.array([11.4,38.2]),np.array([300.,400.]))]:
        uv=np.array([[0.,0.],[287.,383.],[142.23,181.79],[50.37,86.92]])
        inv=cv2.invertAffineTransform(get_warp_matrix(center,scale,0,tuple(SIZE)))
        independent=cv2.transform(uv[None].astype(np.float64),inv)[0]
        formula=uv/SIZE*scale+center-scale/2
        maxerr=max(maxerr,float(np.max(np.abs(independent-formula))))
        W=1520; reflected=uv.copy(); reflected[:,0]=SIZE[0]-uv[:,0]
        c2=center.copy();c2[0]=W-1-center[0]
        back=reflected/SIZE*scale+c2-scale/2;back[:,0]=W-1-back[:,0]
        mirrorerr=max(mirrorerr,float(np.max(np.abs(back-formula))))
    assert maxerr<.001 and mirrorerr<1e-9
    checks.append(dict(name='native_affine_and_flip_no_half_pixel_invention',passed=True,max_inverse_affine_px=maxerr,max_flip_identity_px=mirrorerr))
    save(target,dict(created=now(),checks=checks,all_passed=True,source_sha256=sha(Path(__file__)),
        limits='Synthetic controls validate calculation only; not real landmark accuracy or physical identity.'))
    print(json.dumps(read(target),ensure_ascii=False),flush=True)

def capture():
    import onnxruntime as ort
    from rtmlib import RTMPose
    from rtmlib.visualization.skeleton.coco133 import coco133
    from rtmlib.tools.pose_estimation.pre_processings import get_warp_matrix
    plan=read(OUT/'plan.json');control=read(OUT/'controls.json')
    assert control['all_passed'] and control['source_sha256']==sha(Path(__file__))
    if (OUT/'capture-started.json').exists():raise RuntimeError('Inference already started. Inspect process/result; no automatic repeat.')
    model=ASSETS/'rtmw/rtmw-l-384x288.onnx';norm=read(ASSETS/'rtmw/normalization.json')
    assert sha(model)==plan['model_sha256']
    obs_path=ROOT/'data/observations.npz';input_path=ROOT/'data/training-inputs.json'
    assert sha(obs_path)==plan['observations_sha256'] and sha(input_path)==plan['training_inputs_sha256']
    original=np.load(obs_path); rows=read(input_path)['records']
    class Pose(RTMPose):
        def __init__(self):
            opt=ort.SessionOptions();opt.intra_op_num_threads=4;opt.inter_op_num_threads=1
            self.session=ort.InferenceSession(str(model),sess_options=opt,providers=['CPUExecutionProvider'])
            self.onnx_model=str(model);self.model_input_size=tuple(SIZE);self.mean=np.array(norm['mean']);self.std=np.array(norm['std']);self.backend='onnxruntime';self.device='cpu';self.to_openpose=False
    pose=Pose();assert pose.session.get_inputs()[0].shape[-3:]==[3,384,288]
    names=[coco133['keypoint_info'][i]['name'] for i in range(133)]
    flip=np.array([names.index(coco133['keypoint_info'][i]['swap']) if coco133['keypoint_info'][i]['swap'] else i for i in range(133)])
    import os
    save(OUT/'capture-started.json',dict(created=now(),pid=os.getpid(),budget_seconds=plan['real_frame_wall_seconds'],frames=plan['frames'],source_sha256=sha(Path(__file__))))
    started=time.monotonic();records=[];tiles=[];images=[];arrays={};maxbase=0.;maxscore=0.;maxaffine=0.
    for frame in plan['frames']:
        i=frame-1;row=rows[i];assert row['frame']==str(original['frame_ids'][i])
        path=Path(row['image']);assert sha(path)==row['source_sha256']
        rgb=np.array(Image.open(path).convert('RGB'));W=rgb.shape[1];box=original['boxes'][i].copy()
        fbox=np.array([W-1-box[2],box[1],W-1-box[0],box[3]])
        old=[];new=[];scores=[];valid=[];axisdetails=[]
        for branch,(im,b) in enumerate([(rgb,box),(rgb[:,::-1].copy(),fbox)]):
            crop,center,scale=pose.preprocess(im,b)
            outputs=pose.inference(crop)
            assert [x.shape for x in outputs]==[(1,133,576),(1,133,768)]
            prev,score=pose.postprocess(outputs,center,scale)
            axes=[];flags=[]
            for axis,(raw,kernel) in enumerate(zip(outputs,KERNELS)):
                raw=raw.copy();arrays[f'f{frame:03d}_b{branch}_axis{axis}']=raw
                refined,ff=refine_axis(raw,kernel);axes.append(refined);flags.append(ff)
            candidate=np.stack(axes,axis=-1)/2/SIZE*scale+center-scale/2
            finevalid=np.array([f['valid'] for f in flags[0]])&np.array([f['valid'] for f in flags[1]])
            rawuv=np.stack([outputs[0].argmax(-1),outputs[1].argmax(-1)],axis=-1)/2
            inv=cv2.invertAffineTransform(get_warp_matrix(center,scale,0,tuple(SIZE)))
            independent=cv2.transform(rawuv.astype(np.float64),inv)
            maxaffine=max(maxaffine,float(np.max(np.abs(independent-prev))))
            if branch:
                prev=prev[:,flip];candidate=candidate[:,flip];score=score[:,flip];finevalid=finevalid[flip]
                prev[:,:,0]=W-1-prev[:,:,0];candidate[:,:,0]=W-1-candidate[:,:,0]
            old.append(prev[0]);new.append(candidate[0]);scores.append(score[0]);valid.append(finevalid)
            axisdetails.append(dict(branch=branch,center=center.tolist(),scale=scale.tolist(),flags=flags))
        old=np.mean(old,axis=0);new=np.mean(new,axis=0);score=np.minimum(*scores);good=valid[0]&valid[1]
        parity=np.linalg.norm(old-original['keypoints'][i],axis=-1);scoreerr=np.max(np.abs(score-original['scores'][i]))
        maxbase=max(maxbase,float(parity.max()));maxscore=max(maxscore,float(scoreerr))
        arrays[f'f{frame:03d}_argmax_native']=old;arrays[f'f{frame:03d}_continuous_native']=new
        arrays[f'f{frame:03d}_candidate_valid']=good;arrays[f'f{frame:03d}_scores_unchanged']=score
        rec=dict(frame=frame,old_observation_max_parity_px=float(parity.max()),old_score_max_difference=float(scoreerr),
            image_sha256=row['source_sha256'],box=box.tolist(),all_keypoints_continuous_valid=int(good.sum()),
            nose_valid=good[50:59].tolist(),nose_old=old[50:59].tolist(),nose_candidate=new[50:59].tolist(),
            nose_shift_native_px=stats(np.linalg.norm(new[50:59]-old[50:59],axis=-1)),details=axisdetails)
        records.append(rec)
        if parity.max()>.001 or scoreerr>1e-5:
            np.savez_compressed(OUT/'raw-responses-partial.npz',**arrays)
            save(OUT/'result-failed.json',dict(created=now(),reason='Original decoder replay differs; no further inference.',records=records))
            raise RuntimeError('Baseline parity failed, stop.')
        # Original-resolution crops. No text burnt onto pixels; manifest gives columns.
        xy=old[50:59].mean(0);x0,y0=np.rint(xy).astype(int)-80;rect=(x0,y0,x0+160,y0+160)
        rawim=Image.fromarray(rgb).crop(rect);a=rawim.copy();bim=rawim.copy()
        for target,pts,color in [(a,old,(0,240,255)),(bim,new,(255,160,0))]:
            d=ImageDraw.Draw(target)
            for j in range(50,59):
                if not np.isfinite(pts[j]).all():continue
                x,y=pts[j]-[x0,y0];d.ellipse((x-1.5,y-1.5,x+1.5,y+1.5),outline=color,width=1)
        tile=Image.new('RGB',(480,160));tile.paste(rawim,(0,0));tile.paste(a,(160,0));tile.paste(bim,(320,0))
        name=f'frame_{frame:03d}_raw_argmax_continuous_native.png';tile.save(OUT/name);tiles.append(tile)
        images.append(dict(frame=row['frame'],file=name,region='原图／原argmax青点／连续解码候选橙点；均为原生160px裁片',columns=['raw','argmax','candidate'],crop=list(rect)))
        save(OUT/'progress.json',dict(updated=now(),done=len(records),total=len(plan['frames']),stage='capturing_raw_responses'))
        print(f'frame {frame}: old parity {parity.max():.8f}px, nose candidate shift {rec["nose_shift_native_px"]}',flush=True)
        if time.monotonic()-started>plan['real_frame_wall_seconds']:raise RuntimeError('Fixed inference wall budget exceeded')
    assert sha(obs_path)==plan['observations_sha256'] and sha(input_path)==plan['training_inputs_sha256']
    assert maxaffine<.001
    np.savez_compressed(OUT/'raw-responses.npz',**arrays)
    sheet=Image.new('RGB',(960,480),(245,245,245))
    for i,t in enumerate(tiles):sheet.paste(t,((i%2)*480,(i//2)*160))
    sheet.save(OUT/'all6_native_decoder_control.png')
    shifts=np.concatenate([np.linalg.norm(np.array(r['nose_candidate'])-np.array(r['nose_old']),axis=-1) for r in records])
    result=dict(created=now(),seconds=time.monotonic()-started,frames=plan['frames'],inferences=2*len(records),
        original_decoder_max_parity_px=maxbase,original_score_max_difference=maxscore,independent_inverse_affine_max_px=maxaffine,
        nose_candidate_shift_native_px=stats(shifts),records=records,raw_responses_sha256=sha(OUT/'raw-responses.npz'),
        decoder_control_passed=True,physical_accuracy_validated=False,anatomical_visibility_validated=False,
        new_observations_installed=False,geometry_or_training_executed=False,quality_accepted=False)
    save(OUT/'result.json',result);save(OUT/'index.json',dict(images=images,image_labels_burned_in=False))
    save(OUT/'progress.json',dict(updated=now(),stage='completed_pending_actual_review',done=len(records),total=len(records)))
    print(json.dumps({k:v for k,v in result.items() if k!='records'},ensure_ascii=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['controls','capture']);a=ap.parse_args()
    controls() if a.action=='controls' else capture()
