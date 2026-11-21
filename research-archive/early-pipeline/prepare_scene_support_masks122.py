"""N11 preparation: inspect background lost by a coarse person rectangle."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
from datetime import datetime, timezone
import sys,json,hashlib,time,traceback
import numpy as np
from PIL import Image,ImageDraw

R=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
B=R/'video-update-20261027/background-geometry';P=B/'scene-support-mask-plan.json';O=B/'scene-support-masks'
PROJECT=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118');ASSETS=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets')
FRAMES=[230,246,262,278,294,310,326]
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def register():
    assert not P.exists() and not O.exists()
    files=[B/f'frame_{f:06d}.png' for f in FRAMES]+[B/'offplane-support/visual-review.json',B/'material-quarantine/result.json',
        ASSETS/'sam2/sam2.1_hiera_tiny.pt',ASSETS/'detection/yolov8n.pt',PROJECT/'src/shuolee_portrait/observations.py',PROJECT/'src/shuolee_portrait/runtime.py']
    save(P,dict(stage='N11_preparation',registered=now(),source_sha256=sha(Path(__file__)),
        question='The broad rectangle excludes actual background near the person. Can an independently segmented person silhouette expose usable scene material at different depths?',
        reason='N9 actual image review establishes broad rectangle overexclusion and weak conditional offplane support. N10 one proven material mismatch removal does not improve face3.11 to3.14px. Do not combine another mismatch cleanup with K or camera fitting.',
        operation='Local pinned YOLO person box and SAM2.1 tiny mask on all7 original images. Expand detector box4percent of its maximumdimension, use boxcenter positive prompt; dilate mask by32nativepixels. Retain native image unchanged.',
        review='Inspect every fullframe overlay before any new feature extraction. Save original silhouette and expanded exclusion separately. Recovered background is only a candidate; water/reflections/vegetation do not automatically become static.',
        stop='No person, ambiguous similar-sized people, maskscore<0.5, area outside0.5to95percent or180seconds -> preserve failure, do not change threshold or manually paint masks.',
        budget=dict(attempts=1,frames=7,CPU_seconds=180,GPU=0,new_features=0,new_cameras=0,training_steps=0),
        input_hashes={str(f):sha(f) for f in files}))
    print('N11_MASK_PLAN_REGISTERED',flush=True)

def run():
    assert not O.exists();plan=read(P);assert plan['source_sha256']==sha(Path(__file__))
    for f,h in plan['input_hashes'].items():assert sha(Path(f))==h
    O.mkdir();start=time.monotonic();save(O/'progress.json',dict(state='mask_observation_cpu',pid=os.getpid(),updated=now()))
    sys.path.insert(0,str(PROJECT/'src'))
    from shuolee_portrait import runtime as rt
    rt.configure(ASSETS)
    from shuolee_portrait.observations import load_detector
    import torch,cv2
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    torch.set_num_threads(4);cv2.setNumThreads(2)
    manifest=read(PROJECT/'configs/server-assets.json')['files']
    for name in ['sam2/sam2.1_hiera_tiny.pt','detection/yolov8n.pt']:
        assert sha(ASSETS/name)==next(r['sha256'] for r in manifest if r['path']==name)
    detector=load_detector().to('cpu')
    net=build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',str(ASSETS/'sam2/sam2.1_hiera_tiny.pt'),device='cpu',apply_postprocessing=False)
    segmenter=SAM2ImagePredictor(net,max_hole_area=0,max_sprinkle_area=0)
    rows=[];overview=Image.new('RGB',(304*7,541))
    with torch.inference_mode():
        for i,f in enumerate(FRAMES):
            assert time.monotonic()-start<180
            rgb=np.array(Image.open(B/f'frame_{f:06d}.png').convert('RGB'));H,W=rgb.shape[:2]
            det=detector.predict(rgb[:,:,::-1].copy(),classes=[0],conf=.45,verbose=False,device='cpu')[0]
            boxes=det.boxes.xyxy.cpu().numpy();assert len(boxes)>0
            area=np.prod(boxes[:,2:]-boxes[:,:2],axis=1);order=np.argsort(-area)
            assert len(order)==1 or area[order[1]]<.65*area[order[0]]
            box=boxes[order[0]];pad=.04*max(box[2:]-box[:2]);box=np.clip(box+[-pad,-pad,pad,pad],[0,0,0,0],[W,H,W,H])
            segmenter.set_image(rgb);masks,scores,_=segmenter.predict(box=box,point_coords=((box[:2]+box[2:])/2)[None],point_labels=np.ones(1,dtype=np.int32),multimask_output=False)
            m=masks[0].astype('uint8');fraction=float(m.mean());assert .005<fraction<.95 and scores[0]>=.5
            exclusion=cv2.dilate(m,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(65,65)),iterations=1).astype(bool)
            old=np.zeros((H,W),bool);old[900:2400,350:1100]=True
            recovered=old&~exclusion
            Image.fromarray(m*255).save(O/f'person_{f:06d}.png');Image.fromarray(exclusion.astype('uint8')*255).save(O/f'exclusion_{f:06d}.png')
            Image.fromarray(recovered.astype('uint8')*255).save(O/f'recovered_{f:06d}.png')
            overlay=rgb.astype(float);overlay[recovered]=.65*overlay[recovered]+.35*np.array([0,220,255]);overlay[m>0]=.85*overlay[m>0]+.15*np.array([255,70,50])
            im=Image.fromarray(overlay.astype('uint8'));d=ImageDraw.Draw(im);d.rectangle((350,900,1100,2400),outline=(255,180,0),width=3)
            im.save(O/f'overlay_{f:06d}.png');overview.paste(im.resize((304,541),Image.Resampling.LANCZOS),(304*i,0))
            rows.append(dict(frame=f,box=box.tolist(),sam_score=float(scores[0]),person_pixels=int(m.sum()),exclusion_pixels=int(exclusion.sum()),
                old_rectangle_pixels=int(old.sum()),recovered_background_pixels=int(recovered.sum()),source_sha256=sha(B/f'frame_{f:06d}.png'),
                mask_sha256=sha(O/f'person_{f:06d}.png'),exclusion_sha256=sha(O/f'exclusion_{f:06d}.png')))
            save(O/'progress.json',dict(state='mask_observation_cpu',done=i+1,total=7,pid=os.getpid(),updated=now(),elapsed_seconds=time.monotonic()-start));print('MASK',f,int(recovered.sum()),flush=True)
    overview.save(O/'all7_support_overlays.png')
    save(O/'result.json',dict(stage='N11_preparation',state='complete_pending_actual_mask_review',updated=now(),elapsed_seconds=time.monotonic()-start,rows=rows,
        camera_or_K_changed=False,features_extracted=False,background_static_certified=False,training_started=False))
    for f,h in plan['input_hashes'].items():assert sha(Path(f))==h
    save(O/'progress.json',dict(state='complete_pending_review',pid=None,updated=now()))
    print('N11_MASKS_READY',flush=True)

if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except BaseException as e:
        if O.exists():save(O/'failure.json',dict(updated=now(),error=repr(e),traceback=traceback.format_exc()))
        raise
