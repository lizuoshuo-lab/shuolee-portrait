"""One existing-SFD native-frame visibility-support control; no fitting."""
import argparse
import fcntl
import os
import sys
import time
import traceback
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import audit_sofa_knee_objective124 as a
from run_official_gaussians123 import read, save, sha, now

OUT=a.ROOT/'sfd-face-presence124'
DEPS=Path('/opt/gs-slam-lab/smpl-x-alignment-20261029/face-diagnostic-deps')
OLD=a.ROOT.parent.parent/'smplx-native-face-observations-046/protocol.json'
MODEL_NAME='s3fd-619a316812.pth'


def prepare():
    assert not OUT.exists()
    model=read(OLD)['weights'][MODEL_NAME]
    assert Path(model['path']).is_file() and sha(model['path'])==model['sha256']
    previous=read(a.ROOT/'face-presence-audit/result.json')
    assert previous['actual_visual_review'] and not previous['input_gate_admitted']
    schedule=[]
    paths=[Path(__file__),OLD,Path(model['path']),a.ROOT/'observation-confidence-audit/result.json',
           a.ROOT/'face-presence-audit/result.json']
    paths += [DEPS/('face_alignment/detection/'+n) for n in
              ('core.py','sfd/sfd_detector.py','sfd/net_s3fd.py','sfd/detect.py','sfd/bbox.py')]
    for subject in ('sofa','indoor'):
        src=a.ROOT/(subject+'-input')
        paths += [src/'selection.json',src/'data/inputs.json',src/'data/observations.npz']
        selection=read(src/'selection.json')['fit_frames']
        for row in read(src/'data/inputs.json')['records']:
            if row['frame'] not in selection:continue
            paths.append(Path(row['image']))
            schedule.append(dict(subject=subject,**row))
    assert len(schedule)==24
    OUT.mkdir()
    save(OUT/'plan.json',dict(updated=now(),
        hypothesis='Existing learned SFD on the complete native RGB frame may supply useful independent face-presence support where relative whole-body scores and ROI Haar are unreliable. It is not a landmark visibility oracle.',
        cause_evidence='Current saved indoor53/61 native images show posterior head with nonzero face51 weights; the original whole-body score adapter does not encode absence. This control changes detector/context only, without yet changing any fitting input.',
        schedule=schedule,model=model,
        detector='Original face_alignment SFDDetector with explicit existing path; original default threshold0.5 and NMS0.3. One unscaled full-image forward per frame. No flip, crop or threshold search.',
        association='Retain every detected box. A candidate is associated only when its center is within the existing tracked-person box. Exactly one supplies presence support; zero or multiple remain unknown. No face51 template, best-score pick or posterior mesh is used.',
        budget=dict(frames=24,network_forwards=24,max_wall_seconds=300,single_gpu=True,new_download_bytes=0,fit_calls=0,gaussian_steps=0),
        validation='Actual native crops for every scheduled face, with positive visible-face and posterior-head controls. Inspect failures and association. A detection never validates every landmark; a miss means unsupported, not invisible.',
        stop='One fixed observation control. Do not modify old labels, confidence, solver, model, masks or viewer model. Do not restart/resize after OOM, sweep detector thresholds, reuse cropped old detections or auto-fit. Preserve a failure and review before any integration.',
        frozen={str(p):sha(p) for p in paths}))
    print('SFD_NATIVE_PRESENCE_REGISTERED',flush=True)


def run():
    plan=read(OUT/'plan.json')
    assert not (OUT/'progress.json').exists()
    for p,digest in plan['frozen'].items():assert sha(p)==digest,p
    save(OUT/'progress.json',dict(updated=now(),state='loading',pid=os.getpid(),done=0,total=24))
    started=time.monotonic()
    try:
        import torch
        torch.set_num_threads(4)
        sys.path.insert(0,str(DEPS))
        from face_alignment.detection.sfd.sfd_detector import SFDDetector
        with open('/tmp/shuolee-native122-gpu.lock','a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            detector=SFDDetector(device='cuda',path_to_detector=plan['model']['path'],verbose=False)
            assert detector.fiter_threshold==.5
            records=[];panels=[]
            with torch.inference_mode():
                for row in plan['schedule']:
                    assert time.monotonic()-started<300
                    assert sha(row['image'])==row['source_sha256']
                    raw=Image.open(row['image']).convert('RGB');im=np.asarray(raw)
                    assert raw.size==tuple(row['wh'])
                    boxes=detector.detect_from_image(im)
                    assert len(boxes)==0 or np.asarray(boxes).shape[1]==5
                    person=np.asarray(row['initial_box_xyxy']);candidates=[];detections=[]
                    for i,b in enumerate(boxes):
                        b=np.asarray(b,dtype=float);center=(b[:2]+b[2:4])/2
                        associated=bool(np.all(center>=person[:2]) and np.all(center<person[2:]))
                        detections.append(dict(xyxy=b[:4].tolist(),score=float(b[4]),associated=associated))
                        if associated:candidates.append(i)
                    # Review region depends on the original head ROI and person torso,
                    # not on whether this detector succeeded. Full image was inferred.
                    face=np.asarray(row['face_box_xyxy']);margin=max(float(max(face[2:]-face[:2])),128.)
                    lo=np.floor(face[:2]-margin).astype(int);hi=np.ceil(face[2:]+margin).astype(int)
                    lo=np.maximum(lo,[0,0]);hi=np.minimum(hi,raw.size);crop=tuple(np.r_[lo,hi].tolist())
                    drawn=raw.copy();draw=ImageDraw.Draw(drawn)
                    for box in detections:draw.rectangle(box['xyxy'],outline=(0,230,90) if box['associated'] else (230,150,0),width=2)
                    left,right=raw.crop(crop),drawn.crop(crop)
                    pair=Image.new('RGB',(2*left.width,left.height));pair.paste(left,(0,0));pair.paste(right,(left.width,0))
                    number='006' if row['subject']=='sofa' else '008'
                    filename=f'特征点_124_SFD_{row["frame"]}_原图_整图人脸检测.png'
                    record_path=Path('/mnt/d/00/记录')/number/filename
                    assert not record_path.exists();pair.save(record_path)
                    pair.thumbnail((760,400));panels.append(pair)
                    records.append(dict(subject=row['subject'],frame=row['frame'],detections=detections,
                        associated_indices=candidates,status='single_face_support' if len(candidates)==1 else 'unknown_no_unique_support',
                        image=str(record_path),crop_integer=crop,native_wh=row['wh'],
                        no_external_resize_or_landmark_change=True))
                    save(OUT/'progress.json',dict(updated=now(),state='observing',pid=os.getpid(),done=len(records),total=24))
                    print('SFD_FRAME',row['subject'],row['frame'],len(boxes),len(candidates),flush=True)
            torch.cuda.synchronize()
            peak=int(torch.cuda.max_memory_allocated())
        pages=[]
        for start in range(0,len(panels),6):
            page=Image.new('RGB',(1520,1200),'black')
            for i,panel in enumerate(panels[start:start+6]):page.paste(panel,((i%2)*760,(i//2)*400))
            path=OUT/f'page-{start//6+1:02d}.png';page.save(path);pages.append(str(path))
        for p,digest in plan['frozen'].items():assert sha(p)==digest,p
        save(OUT/'result.json',dict(updated=now(),state='completed_actual_review_pending',records=records,pages=pages,
            seconds=time.monotonic()-started,peak_cuda_allocated_bytes=peak,network_forwards=24,
            source_hashes_verified=True,input_modified=False,new_download_bytes=0,fit_calls=0,
            actual_visual_review=False,gate_admitted=False,quality_accepted=False,gaussian_admitted=False))
        save(OUT/'progress.json',dict(updated=now(),state='completed',pid=None,done=24,total=24))
        print('SFD_NATIVE_PRESENCE_COMPLETE',flush=True)
    except BaseException as e:
        save(OUT/'failure.json',dict(updated=now(),error=repr(e),traceback=traceback.format_exc()))
        save(OUT/'progress.json',dict(updated=now(),state='failed',pid=None))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);args=parser.parse_args()
    prepare() if args.action=='prepare' else run()
