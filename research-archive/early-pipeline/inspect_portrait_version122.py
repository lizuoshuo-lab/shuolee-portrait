"""N0: fixed dimension-derived pixel mapping between two encodings, not camera fitting."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,subprocess,time
import cv2,numpy as np

R=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
I=R/'video-update-20261027';P=I/'portrait-version-plan.json';O=I/'portrait-version-mapping'
OLD=Path('/opt/gs-slam-lab/shuolee-raw-domain-121/portrait')
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text())
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def register():
    if P.exists():raise RuntimeError('Already registered')
    v=read(I/'result.json')['videos'][1];old=read(OLD/'observations.json')
    assert v['id']=='clip02' and (v['width'],v['height'],v['frames'])==(1520,2704,557)
    frames=[0,266,556];rows={r['raw_source_frame_index']:r for r in old['records']}
    save(P,dict(stage='N0',registered=now(),new_video=v,old_video_sha256=old['records'][0]['original_video_sha256'],
        old_frames=[rows[f] for f in frames],native_new_frame_indices=frames,
        hypothesis='New clip is a higher-resolution encoding of the old portrait orbit. Test only the fixed edge-aligned resize mapping implied by dimensions (old1080x1920 to new1520x2704), including the half-pixel origin. Do not fit an affine, K, pose or 3D point.',
        mapping_old_integer_to_new_integer=[[1520/1080,0,(1520/1080-1)/2],[0,2704/1920,(2704/1920-1)/2],[0,0,1]],
        detection='OpenCV SIFT nfeatures3000 on each native unresized image. Mutual descriptor ratio0.75. Compare all accepted appearance matches to fixed mapping without residual filtering. Exact sameframe identity remains a tested hypothesis.',
        budget=dict(attempts=1,CPU_wall_seconds=120,GPU=0,pairs=3),
        limits='Report count, median/P90/max and every >1newpixel residual. No physical face acceptance follows from same-frame encoding consistency. Training images/old observations/K unchanged.',
        source_sha256=sha(Path(__file__))))
    print('N0_FIXED_MAPPING_REGISTERED')
def run():
    plan=read(P);assert plan['source_sha256']==sha(Path(__file__))
    if O.exists():raise RuntimeError('Do not repeat completed N0')
    O.mkdir();start=time.monotonic();v=plan['new_video'];newpath=Path(v['path']);assert sha(newpath)==v['sha256']
    framepaths={0:I/'clip02/sample_01.png',556:I/'clip02/sample_03.png'}
    cmd=['/usr/bin/ffmpeg','-nostdin','-hide_banner','-loglevel','error','-noautorotate','-i',str(newpath),'-vf',"select='eq(n,266)'",'-vsync','0','-frames:v','1',str(O/'new_frame000266.png')]
    subprocess.run(cmd,check=True,capture_output=True,timeout=120);framepaths[266]=O/'new_frame000266.png'
    A=np.asarray(plan['mapping_old_integer_to_new_integer']);sift=cv2.SIFT_create(nfeatures=3000);matcher=cv2.BFMatcher();results=[]
    for row in plan['old_frames']:
        f=row['raw_source_frame_index'];p=Path(row['image']);assert sha(p)==row['source_sha256']
        old=cv2.imread(str(p),cv2.IMREAD_GRAYSCALE);new=cv2.imread(str(framepaths[f]),cv2.IMREAD_GRAYSCALE)
        assert old.shape==(1920,1080) and new.shape==(2704,1520)
        ko,do=sift.detectAndCompute(old,None);kn,dn=sift.detectAndCompute(new,None)
        def ratio(a,b):
            return {m[0].queryIdx:m[0].trainIdx for m in matcher.knnMatch(a,b,k=2) if len(m)==2 and m[0].distance<.75*m[1].distance}
        ab,ba=ratio(do,dn),ratio(dn,do);pairs=sorted((i,j) for i,j in ab.items() if ba.get(j)==i)
        uv0=np.array([ko[i].pt for i,j in pairs]);uv1=np.array([kn[j].pt for i,j in pairs])
        expected=np.c_[uv0,np.ones(len(pairs))]@A.T;res=uv1-expected[:,:2];error=np.linalg.norm(res,axis=1)
        np.savez_compressed(O/f'frame{f:06d}_raw_matches.npz',old_uv=uv0,new_uv=uv1,expected_new_uv=expected[:,:2],residual=res,error=error)
        result=dict(frame=f,matches=len(pairs),median_new_px=float(np.median(error)),p90_new_px=float(np.percentile(error,90)),max_new_px=float(error.max()),
            strictly_below_1px=int((error<1).sum()),at_least_1px=int((error>=1).sum()),new_image_sha256=sha(framepaths[f]),old_image_sha256=sha(p),
            missing_match_selection='Only mutual descriptor ratio, no geometric residual filtering',matrix_fitted=False)
        results.append(result);print(json.dumps(result),flush=True)
        if time.monotonic()-start>120:raise RuntimeError('N0 budget exceeded')
    assert sha(newpath)==v['sha256']
    save(O/'result.json',dict(stage='N0',updated=now(),state='completed_fixed_resize_hypothesis_check',pairs=results,matrix=A.tolist(),
        elapsed_seconds=time.monotonic()-start,K_estimated=False,poses_estimated=False,training_images_resampled=False,
        physical_calibration_accepted=False,face_subpixel_accepted=False,old_tracks_reused_as_training=False,
        interpretation='Same-frame appearance matches are compared to a matrix fixed before matching. This checks encoded-image mapping, not camera calibration, cross-time physical correspondence, person stationarity, or real face accuracy.'))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--register',action='store_true');a=ap.parse_args()
    register() if a.register else run()
