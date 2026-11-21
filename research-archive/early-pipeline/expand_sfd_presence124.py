"""Same frozen SFD detector on the remaining sofa/indoor frames, cached24 retained."""
import argparse
import fcntl
import os
import sys
import time
import traceback
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import check_sfd_presence124 as previous
from run_official_gaussians123 import read,save,sha,now

a=previous.a
OUT=a.ROOT/'sfd-face-presence124/full167'


def prepare():
    assert not OUT.exists()
    review=read(previous.OUT/'visual-review.json')
    assert review['actual_visual_review'] and review['allow_same_detector_roster_expansion']
    old=read(previous.OUT/'plan.json');result=read(previous.OUT/'result.json')
    assert result['network_forwards']==24 and result['source_hashes_verified']
    schedule=[];frozen=old['frozen'].copy()
    for subject in ('sofa','indoor'):
        rows=read(a.ROOT/(subject+'-input/data/inputs.json'))['records']
        schedule += [dict(subject=subject,**r) for r in rows]
        for row in rows:frozen[row['image']]=sha(row['image'])
    assert len(schedule)==167
    for path in [Path(__file__),previous.OUT/'plan.json',previous.OUT/'result.json',previous.OUT/'visual-review.json']:
        frozen[str(path)]=sha(path)
    OUT.mkdir()
    save(OUT/'plan.json',dict(updated=now(),model=old['model'],schedule=schedule,
        hypothesis='Validate the unchanged learned face-presence support across the complete registered sofa56/indoor111 roster before any pose input integration.',
        detector=old['detector'],association=old['association'],
        budget=dict(total_frames=167,verified_cache_reuse=24,new_network_forwards=143,max_wall_seconds=300,new_downloads=0,fit_calls=0),
        rule='Reuse exactly the24 existing full-native SFD records by subject,frame,input hash; infer each of the143 remaining frames once. Never use old ROI Haar or older-person SFD detections as fallback.',
        stop='Preserve any resource/input/runtime failure; no resizing, score scan, extra pass or fit. Review all presence panels and uncertain native cases; support is not per-landmark visibility or pose admission.',frozen=frozen))
    print('SFD_FULL167_REGISTERED',flush=True)


def run():
    plan=read(OUT/'plan.json');assert not (OUT/'progress.json').exists()
    for path,digest in plan['frozen'].items():assert sha(path)==digest,path
    save(OUT/'progress.json',dict(updated=now(),state='loading',pid=os.getpid(),done=0,total=167))
    started=time.monotonic();new_calls=0
    try:
        import torch
        torch.set_num_threads(4);sys.path.insert(0,str(previous.DEPS))
        from face_alignment.detection.sfd.sfd_detector import SFDDetector
        cache={(r['subject'],r['frame']):r for r in read(previous.OUT/'result.json')['records']}
        records=[];panels=[]
        with open('/tmp/shuolee-native122-gpu.lock','a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            detector=SFDDetector(device='cuda',path_to_detector=plan['model']['path'],verbose=False)
            assert detector.fiter_threshold==.5
            with torch.inference_mode():
                for row in plan['schedule']:
                    assert time.monotonic()-started<300
                    assert sha(row['image'])==row['source_sha256']
                    key=(row['subject'],row['frame']);raw=Image.open(row['image']).convert('RGB')
                    if key in cache:
                        rec=dict(cache[key],cached=True)
                        pair=Image.open(rec['image']).convert('RGB')
                    else:
                        detections=[];candidates=[];person=np.asarray(row['initial_box_xyxy'])
                        boxes=detector.detect_from_image(np.asarray(raw));new_calls+=1
                        for j,box in enumerate(boxes):
                            b=np.asarray(box,dtype=float);center=(b[:2]+b[2:4])/2
                            associated=bool(np.all(center>=person[:2]) and np.all(center<person[2:]))
                            detections.append(dict(xyxy=b[:4].tolist(),score=float(b[4]),associated=associated))
                            if associated:candidates.append(j)
                        face=np.asarray(row['face_box_xyxy']);margin=max(float(max(face[2:]-face[:2])),128.)
                        lo=np.maximum(np.floor(face[:2]-margin).astype(int),[0,0])
                        hi=np.minimum(np.ceil(face[2:]+margin).astype(int),raw.size);crop=tuple(np.r_[lo,hi].tolist())
                        drawn=raw.copy();draw=ImageDraw.Draw(drawn)
                        for box in detections:draw.rectangle(box['xyxy'],outline=(0,230,90) if box['associated'] else (230,150,0),width=2)
                        l,r=raw.crop(crop),drawn.crop(crop)
                        pair=Image.new('RGB',(l.width*2,l.height));pair.paste(l,(0,0));pair.paste(r,(l.width,0))
                        number='006' if row['subject']=='sofa' else '008'
                        path=Path('/mnt/d/00/记录')/number/f'特征点_124_SFD_{row["frame"]}_原图_整图人脸检测.png'
                        assert not path.exists();pair.save(path)
                        rec=dict(subject=row['subject'],frame=row['frame'],detections=detections,
                            associated_indices=candidates,status='single_face_support' if len(candidates)==1 else 'unknown_no_unique_support',
                            image=str(path),crop_integer=crop,native_wh=row['wh'],cached=False,no_external_resize_or_landmark_change=True)
                    rec['source_sha256']=row['source_sha256'];records.append(rec)
                    pair.thumbnail((600,260));panels.append(pair)
                    save(OUT/'progress.json',dict(updated=now(),state='observing',pid=os.getpid(),done=len(records),total=167,new_calls=new_calls))
                    if len(records)%12==0:print('SFD_ROSTER',len(records),new_calls,flush=True)
            peak=int(torch.cuda.max_memory_allocated())
        assert new_calls==143 and sum(r['cached'] for r in records)==24
        pages=[]
        for start in range(0,len(panels),12):
            page=Image.new('RGB',(1200,1560),'black')
            for j,p in enumerate(panels[start:start+12]):page.paste(p,((j%2)*600,(j//2)*260))
            path=OUT/f'page-{start//12+1:02d}.png';page.save(path)
            pages.append(dict(path=str(path),frames=[dict(subject=r['subject'],frame=r['frame']) for r in records[start:start+12]]))
        for path,digest in plan['frozen'].items():assert sha(path)==digest,path
        save(OUT/'result.json',dict(updated=now(),state='complete_actual_review_pending',records=records,pages=pages,
            seconds=time.monotonic()-started,new_network_forwards=new_calls,cached_frames=24,
            peak_cuda_allocated_bytes=peak,source_hashes_verified=True,actual_visual_review=False,
            gate_admitted=False,fit_calls=0,gaussian_admitted=False,quality_accepted=False))
        save(OUT/'progress.json',dict(updated=now(),state='completed',pid=None,done=167,total=167,new_calls=new_calls))
        print('SFD_FULL167_COMPLETE',flush=True)
    except BaseException as e:
        save(OUT/'failure.json',dict(updated=now(),error=repr(e),traceback=traceback.format_exc(),new_calls=new_calls))
        save(OUT/'progress.json',dict(updated=now(),state='failed',pid=None,new_calls=new_calls));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);args=p.parse_args()
    prepare() if args.action=='prepare' else run()
