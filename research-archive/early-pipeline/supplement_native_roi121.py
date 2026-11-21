"""One preregistered native crop feature supplement; detector ROIs are not correspondences."""
from pathlib import Path
from datetime import datetime,timezone
import os,sys,json,hashlib,time,traceback
os.environ['CUDA_VISIBLE_DEVICES']=''
import cv2
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0,'/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118/src')
from shuolee_portrait import runtime as rt
from shuolee_portrait.observations import load_detector,face_region_box
from shuolee_portrait.material_tracks import mutual_matches,cycle_components
rt.ASSETS=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets')
ROOT=Path('/opt/gs-slam-lab/shuolee-raw-domain-121')
R=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/raw-domain-121')
def now():return datetime.now(timezone.utc).isoformat()
def save(p,d):rt.save(p,d)
def state(**kw):save(R/'roi-supplement-progress.json',dict(updated=now(),pid=os.getpid(),training_started=False,**kw))
def model_pose():
    import onnxruntime as ort
    from rtmlib import RTMPose
    model=rt.ASSETS/'rtmw/rtmw-l-384x288.onnx'
    entry=next(x for x in rt.read(rt.PROJECT/'configs/server-assets.json')['files'] if x['path']=='rtmw/rtmw-l-384x288.onnx')
    if rt.sha(model)!=entry['sha256']:raise RuntimeError('Official ROI model hash mismatch')
    norm=rt.read(rt.ASSETS/'rtmw/normalization.json')
    class Pose(RTMPose):
        def __init__(self):
            opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1
            self.session=ort.InferenceSession(str(model),sess_options=opts,providers=['CPUExecutionProvider'])
            self.onnx_model=str(model);self.model_input_size=(288,384);self.mean=np.array(norm['mean']);self.std=np.array(norm['std']);self.backend='onnxruntime';self.device='cpu';self.to_openpose=False
    return Pose()
def run(subject,detector,pose):
    start=time.monotonic();base=ROOT/subject;out=base/'roi-supplement'
    if out.exists():raise RuntimeError('Supplement already exists: inspect progress; do not repeat')
    out.mkdir();(out/'features').mkdir();(out/'pairs').mkdir()
    old=rt.read(base/'observations.json');rows=old['records'];features=[];rois=[]
    sift_body=cv2.SIFT_create(nfeatures=1000,contrastThreshold=.02)
    sift_face=cv2.SIFT_create(nfeatures=500,contrastThreshold=.02)
    for i,row in enumerate(rows):
        if time.monotonic()-start>900:raise RuntimeError('Supplement per-person 900-second budget exceeded')
        bgr=cv2.imread(row['image']);h,w=bgr.shape[:2];assert [w,h]==row['wh']
        det=detector.predict(bgr,classes=[0],conf=.45,verbose=False,device='cpu')[0];boxes=det.boxes.xyxy.cpu().numpy()
        if len(boxes)==0:raise RuntimeError('No unambiguous person ROI; no interpolated ROI')
        areas=np.prod(boxes[:,2:]-boxes[:,:2],axis=1);order=np.argsort(-areas)
        if len(order)>1 and areas[order[1]]>.65*areas[order[0]]:raise RuntimeError('Ambiguous person ROI')
        bb=boxes[order[0]];margin=.04*max(bb[2:]-bb[:2]);body=np.clip(bb+[-margin,-margin,margin,margin],[0,0,0,0],[w,h,w,h])
        kp,sc=pose(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),bboxes=body[None]);face=face_region_box(kp[0],sc[0],(w,h))
        original=np.load(base/'features'/f'{i:04d}.npz');xy=list(original['xy']);ds=list(original['descriptor']);resp=list(original['sift_response']);source=['original_whole_frame']*len(xy)
        roirow=dict(frame=row['frame'],body_box=body.tolist(),face_box=face,roi_model_only_not_material_identity=True,crop_features={})
        for name,box,detector_local in [('body',body,sift_body),('face',face,sift_face)]:
            if box is None:roirow['crop_features'][name]=0;continue
            box=np.asarray(box);pad=12 if name=='face' else 0
            x0,y0=np.maximum(0,np.floor(box[:2]-pad).astype(int));x1,y1=np.minimum([w,h],np.ceil(box[2:]+pad).astype(int))
            native=cv2.cvtColor(bgr[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY)
            keys,desc=detector_local.detectAndCompute(native,None);roirow['crop_features'][name]=len(keys)
            for k,d in zip(keys,desc if desc is not None else []):xy.append([k.pt[0]+x0,k.pt[1]+y0]);ds.append(d);resp.append(k.response);source.append(name)
        xy=np.asarray(xy);ds=np.asarray(ds,np.float32);resp=np.asarray(resp)
        # Suppress one-pixel variants in each frame before any matching or split.
        tree=cKDTree(xy);keep=[];blocked=set()
        for k in sorted(range(len(xy)),key=lambda k:(-resp[k],k)):
            if k in blocked:continue
            keep.append(k);blocked.update(tree.query_ball_point(xy[k],1.0))
        keep=np.asarray(sorted(keep));xy=xy[keep];ds=ds[keep]
        np.savez_compressed(out/'features'/f'{i:04d}.npz',xy=xy,descriptor=ds,source=np.asarray(source)[keep],sift_response=resp[keep])
        features.append((xy,ds));rois.append(roirow);state(state='native_roi_features',subject=subject,frames_done=i+1,frames_total=len(rows));print(subject,'native_roi',i+1,roirow['crop_features'],flush=True)
    triplets=[(i,i+1,i+2) for i in range(len(rows)-2)];pairs=sorted({p for a,b,c in triplets for p in [(a,b),(b,c),(a,c)]});maps={};pi=[]
    for a,b in pairs:
        match=mutual_matches(features[a][1],features[b][1],.75);np.savez_compressed(out/'pairs'/f'{a:04d}_{b:04d}.npz',**match)
        maps[(a,b)]={int(i):int(j) for i,j in match['matches']};pi.append(dict(a=a,b=b,frames=[rows[a]['frame'],rows[b]['frame']],mutual_matches=len(match['matches'])))
    components,rejected,counts=cycle_components(maps,triplets)
    # Retain original held-out observations, and conservatively isolate new tracks nearby.
    oldfit={r['frame']:[] for r in rows};oldheld={r['frame']:[] for r in rows}
    for t in old['tracks']:
        for o in t['observations']:(oldfit if t['split']=='fit' else oldheld)[o['frame']].append(o['uv'])
    trees={kind:{f:cKDTree(np.array(p).reshape(-1,2)) if p else None for f,p in d.items()} for kind,d in [('fit',oldfit),('heldout',oldheld)]}
    tracks=[];quarantined=0
    for nodes in components:
        ident=hashlib.sha256((old['records'][0]['original_video_sha256']+'|'+ '|'.join(f'{rows[a]["frame"]}:{b}' for a,b in nodes)).encode()).hexdigest()
        votes=set()
        for a,b in nodes:
            frame=rows[a]['frame'];uv=features[a][0][b]
            for kind in ['fit','heldout']:
                tree=trees[kind][frame]
                if tree is not None and tree.query(uv)[0]<=3.:votes.add(kind)
        if votes=={'fit','heldout'}:split='quarantine';quarantined+=1
        elif votes:split=next(iter(votes))
        else:split='heldout' if int(ident[:8],16)%5==0 else 'fit'
        tracks.append(dict(track_id=ident,split=split,region='unknown',identity_basis='native_ROI_appearance_cycles_unreviewed',
            observations=[dict(frame=rows[a]['frame'],uv=features[a][0][b].tolist(),sift_index=b,image_sha256=rows[a]['source_sha256']) for a,b in nodes]))
    save(out/'observations.json',dict(schema='shuolee-native-observations-v1',records=rows,tracks=tracks,pairs=pi,rois=rois,
        pixel_domain='native_already_processed_export_integer_centers',K_applied=False,poses_estimated=False,physical_identity_verified=False,
        feature_images_resized=False,descriptor_crop_to_full_image='translation by integer crop origin only',
        split_policy='Per-frame1px duplicate suppression; preserve original fit/held membership via3px spatial exclusion. Mixed members quarantined; new isolated identities hashed20%heldout before geometry.'))
    result=dict(subject=subject,updated=now(),tracks=len(tracks),fit=sum(t['split']=='fit' for t in tracks),heldout=sum(t['split']=='heldout' for t in tracks),quarantine=quarantined,cycles=counts,
        face_ROIs=sum(r['face_box'] is not None for r in rois),output=str(out),training_started=False,physical_identity_verified=False)
    save(out/'result.json',result);save(R/f'{subject}-roi-supplement.json',result);return result
def main():
    import fcntl,torch
    lock=open(ROOT/'roi-supplement.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4);cv2.setNumThreads(4)
    state(state='loading_existing_official_ROI_models',script_sha256=rt.sha(Path(__file__)))
    detector=load_detector();pose=model_pose();results=[]
    for subject in ['grass','sofa','portrait','indoor']:results.append(run(subject,detector,pose))
    save(R/'roi-supplement-result.json',dict(updated=now(),subjects=results,training_started=False));state(state='complete_pending_material_review')
if __name__=='__main__':
    try:main()
    except Exception as e:state(state='failed',error=repr(e));save(R/'roi-supplement-failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
