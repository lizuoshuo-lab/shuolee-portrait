"""Bounded native-video feature audit. Never remaps/resizes images or estimates K.

Decoded MP4 pixels are not asserted to be raw fisheye pixels. Detection and
subpixel corner refinement happen before any geometric image operation here.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, sys, subprocess, traceback
import cv2
import numpy as np
sys.path.insert(0,'/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118/src')
from shuolee_portrait.material_tracks import mutual_matches, cycle_components

BASE=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/raw-domain-121')
DATA=Path('/mnt/d/00/工作/数据集')
WORK=Path('/opt/gs-slam-lab/shuolee-raw-domain-121')
SUBJECTS=[('grass','2f1aac19-8ae1_1791971027322.mp4'),('sofa','819080de-8b00.mp4'),('portrait','online4.mp4'),('indoor','df776c03-91a4_1792572380952.mp4')]
def now():return datetime.now(timezone.utc).isoformat()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def state(**kw):save(BASE/'feature-progress.json',dict(updated=now(),pid=os.getpid(),training_started=False,**kw))

def run_subject(subject,filename):
    out=WORK/subject
    if out.exists():raise FileExistsError(f'Fresh output required; inspect existing job: {out}')
    out.mkdir(parents=True);(out/'frames').mkdir();(out/'features').mkdir();(out/'pairs').mkdir()
    video=DATA/filename;digest=sha(video)
    meta=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_streams','-show_frames','-show_entries','stream=width,height,nb_frames:frame=best_effort_timestamp_time','-of','json',str(video)]))
    frames=meta['frames'];stream=meta['streams'][0];wh=[stream['width'],stream['height']]
    indices=sorted(set(np.linspace(0,len(frames)-1,24).round().astype(int).tolist()))
    cap=cv2.VideoCapture(str(video));cap.set(cv2.CAP_PROP_ORIENTATION_AUTO,0)
    detector=cv2.SIFT_create(nfeatures=2500,contrastThreshold=.02)
    records=[];features=[];wanted=set(indices)
    for index in range(len(frames)):
        if not cap.grab():raise RuntimeError(f'Frame decode ended early at {index}')
        if index not in wanted:continue
        ok,bgr=cap.retrieve()
        if not ok or list(bgr.shape[1::-1])!=wh:raise RuntimeError('Native frame dimension/decode mismatch')
        k=len(records);name=f'{index:06d}.png';image=out/'frames'/name
        if not cv2.imwrite(str(image),bgr):raise RuntimeError('Cannot save native frame')
        gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
        keys,desc=detector.detectAndCompute(gray,None)
        xy=np.asarray([p.pt for p in keys],np.float64).reshape(-1,2)
        desc=np.empty((0,128),np.float32) if desc is None else desc
        # Separate native-domain corner audit, not automatically a 3D track.
        corners=cv2.goodFeaturesToTrack(gray,512,.01,8,blockSize=5)
        if corners is None:corners=np.empty((0,1,2),np.float32)
        before=corners.copy()
        if len(corners):cv2.cornerSubPix(gray,corners,(5,5),(-1,-1),(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_MAX_ITER,30,.01))
        edge=np.minimum.reduce([xy[:,0]/wh[0],xy[:,1]/wh[1],(wh[0]-1-xy[:,0])/wh[0],(wh[1]-1-xy[:,1])/wh[1]])<.1
        np.savez_compressed(out/'features'/f'{k:04d}.npz',xy=xy,descriptor=desc,
            sift_size=np.asarray([p.size for p in keys]),sift_response=np.asarray([p.response for p in keys]),
            edge_band=edge,corners_before=before.reshape(-1,2),corners_native_subpixel=corners.reshape(-1,2))
        rec=dict(frame=f'{index:06d}',raw_source_frame_index=index,pts_seconds=float(frames[index]['best_effort_timestamp_time']),
            image=str(image),source_sha256=sha(image),wh=wh,original_video_sha256=digest,
            pixel_domain='native_encoded_video_integer_centers',raw_fisheye_confirmed=False,
            additional_geometric_resampling=False,sift_points=len(keys),edge_band_points=int(edge.sum()),
            subpixel_corners=len(corners),decoded_color='OpenCV BGR8; codec color conversion, no geometric resampling')
        records.append(rec);features.append((xy,desc));state(state='detecting_native_features',subject=subject,frames_done=len(records),frames_total=len(indices))
        print(subject,index,'native_features',len(keys),flush=True)
    cap.release()
    if len(records)!=len(indices):raise RuntimeError('Missing requested native frames')
    triplets=[(i,i+1,i+2) for i in range(len(records)-2)]
    pairs=sorted({p for a,b,c in triplets for p in [(a,b),(b,c),(a,c)]})
    maps={};pair_info=[]
    for a,b in pairs:
        match=mutual_matches(features[a][1],features[b][1],.75)
        np.savez_compressed(out/'pairs'/f'{a:04d}_{b:04d}.npz',**match)
        maps[(a,b)]={int(i):int(j) for i,j in match['matches']}
        pair_info.append(dict(a=a,b=b,frames=[records[a]['frame'],records[b]['frame']],mutual_matches=len(match['matches'])))
        state(state='matching_native_features',subject=subject,pairs_done=len(pair_info),pairs_total=len(pairs))
    components,rejected,counts=cycle_components(maps,triplets)
    tracks=[]
    for nodes in components:
        token='|'.join(f'{records[a]["frame"]}:{b}' for a,b in nodes)
        ident=hashlib.sha256((digest+token).encode()).hexdigest()
        tracks.append(dict(track_id=ident,split='heldout' if int(ident[:8],16)%5==0 else 'fit',
            identity_basis='appearance_cycle_candidate_unreviewed',region='unknown',
            observations=[dict(frame=records[a]['frame'],uv=features[a][0][b].tolist(),sift_index=b,image_sha256=records[a]['source_sha256']) for a,b in nodes]))
    save(out/'observations.json',dict(schema='shuolee-native-observations-v1',records=records,tracks=tracks,pairs=pair_info,
        pixel_domain='native_encoded_video_integer_centers',detection_before_geometric_resampling=True,
        inverse_warped_detector_points=False,K_applied=False,poses_estimated=False,
        raw_fisheye_confirmed=False,physical_material_identity_verified=False))
    if sha(video)!=digest:raise RuntimeError('Source video changed while running')
    result=dict(subject=subject,video=str(video),source_sha256=digest,actual_native_wh=wh,frames=len(records),
        sift_features=sum(x['sift_points'] for x in records),edge_band_features=sum(x['edge_band_points'] for x in records),
        mutual_matches=sum(x['mutual_matches'] for x in pair_info),cycle_tracks=len(tracks),
        heldout_tracks=sum(x['split']=='heldout' for x in tracks),cycles=counts,
        frames_covered_by_cycle_tracks=len({o['frame'] for t in tracks for o in t['observations']}),
        output=str(out),requires_material_region_review=True,raw_fisheye_confirmed=False,
        detection_before_geometric_resampling=True,K_applied=False,poses_estimated=False,training_started=False)
    save(BASE/f'{subject}-features.json',result)
    save(out/'result.json',result)
    return result

def main():
    BASE.mkdir(parents=True,exist_ok=True);WORK.mkdir(parents=True,exist_ok=True)
    lock=open(WORK/'worker.lock','a+')
    import fcntl
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state(state='starting',implementation_sha256=sha(Path(__file__)))
    cv2.setNumThreads(4);results=[]
    try:
        for subject,video in SUBJECTS:results.append(run_subject(subject,video))
        save(BASE/'native-feature-result.json',dict(updated=now(),subjects=results,
            all_features_in_native_encoded_domain=True,raw_fisheye_domain_verified=False,
            calibration_accepted=False,training_started=False))
        state(state='native_features_complete_pending_domain_and_material_review',subjects_complete=len(results))
    except Exception as e:
        state(state='failed',error=f'{type(e).__name__}: {e}')
        traceback.print_exc();raise
if __name__=='__main__':main()
