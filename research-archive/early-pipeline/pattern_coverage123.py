"""One frozen coverage pass, no matching, calibration, segmentation or fitting."""
from pathlib import Path
from datetime import datetime,timezone
import os,sys,json,time,hashlib,traceback,subprocess,fcntl
os.environ.update(PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',YOLO_AUTOINSTALL='false')
TOP=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=TOP/'shuolee/current/local-clothed-surface-123';PRE=REC/'pattern-preflight'
OUT=PRE/'coverage';ROOT=Path('/opt/gs-slam-lab/shuolee-local-surface-123/preflight/pattern')
PROJECT=ROOT.parents[1]/'runtime';ASSETS=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets')
sys.path.insert(0,str(PROJECT/'src'))
from shuolee_portrait import runtime as rt
rt.ASSETS=ASSETS
from shuolee_portrait.observations import load_detector,face_region_box
read=rt.read;save=rt.save;sha=rt.sha
def now():return datetime.now(timezone.utc).isoformat()
def main():
    import numpy as np,torch,onnxruntime as ort,cv2
    from PIL import Image
    from rtmlib import RTMPose
    from rtmlib.visualization.skeleton.coco133 import coco133
    torch.set_num_threads(4);cv2.setNumThreads(4)
    assert not OUT.exists(),'Existing coverage output, inspect rather than repeat.'
    plan=read(PRE/'coverage-plan.json');manifest=read(ROOT/'decoded.json')
    assert sha(ROOT/'decoded.json')==plan['manifest_sha256'] and len(manifest['records'])==99
    assert manifest['video_sha256']==plan['input_sha256']
    assets=read(PROJECT/'configs/server-assets.json')['files']
    paths=['detection/yolov8n.pt','rtmw/rtmw-l-384x288.onnx','rtmw/normalization.json']
    hashes={}
    for name in paths:
        p=ASSETS/name;entry=next(x for x in assets if x['path']==name)
        assert p.is_file() and sha(p)==entry['sha256'],name
        hashes[name]=entry['sha256']
    lock=open('/tmp/shuolee-native122-gpu.lock','a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    running=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
    assert not running,'Another GPU compute task is active: '+running
    OUT.mkdir();(OUT/'responses').mkdir()
    registered=dict(registered=now(),parent_plan_sha256=sha(PRE/'coverage-plan.json'),script_sha256=sha(Path(__file__)),model_sha256=hashes,
        previous_goal_turn='progress: local matching branch rejected with evidence, all99patternframes decoded and reviewed in5samples',
        detector=dict(model='YOLOv8n',imgsz=640,conf=.45,iou=.7,classes=[0],max_det=300,rect=True,augment=False,device=0),
        subject_selection='Largest first detection, require nextarea<=.65largest; subsequent highestIoU>=.1 with runnerupgap>=.05. If ambiguous, preserve every detection, store unavailable pose and do not update identity.',
        pose=dict(model='officialRTMW-L',input_wh=[288,384],color='RGB',normalization='verifiedofficialJSON',padding=1.25,body_margin=.04,backend='ONNXRuntimeCPU',decoder='existingargmax,SimCCratio2',forward_passes_per_resolved_frame=1,flipped_inference=False),
        visibility='No automatic anatomical visibility label. Store image boundary tests, confidence and rawSimCC. ROI is not a material correspondence or visible-surface truth.',
        budget=dict(frames=99,max_wall_seconds=1200,new_downloads=0,newcamera=0,SMPLX=0,training=0))
    save(OUT/'execution-plan.json',registered)
    started=time.monotonic()
    save(OUT/'progress.json',dict(updated=now(),state='loading_existing_models',pid=os.getpid(),command=sys.argv,done=0,total=99,training=False))
    detector=load_detector();norm=read(ASSETS/'rtmw/normalization.json')
    class Pose(RTMPose):
        def __init__(self):
            opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1
            self.session=ort.InferenceSession(str(ASSETS/'rtmw/rtmw-l-384x288.onnx'),sess_options=opts,providers=['CPUExecutionProvider'])
            self.onnx_model=str(ASSETS/'rtmw/rtmw-l-384x288.onnx');self.model_input_size=(288,384)
            self.mean=np.array(norm['mean']);self.std=np.array(norm['std']);self.backend='onnxruntime';self.device='cpu';self.to_openpose=False
    pose=Pose();assert pose.session.get_inputs()[0].shape[-2:]==[384,288]
    names=[coco133['keypoint_info'][i]['name'] for i in range(133)]
    regions=dict(body=list(range(17)),feet=list(range(17,23)),face=list(range(23,91)),left_hand=list(range(91,112)),right_hand=list(range(112,133)))
    rows=[];previous=None;allxy=[];allscore=[];allinside=[]
    with torch.inference_mode():
        for slot,r in enumerate(manifest['records']):
            assert time.monotonic()-started<1200,'Coverage wall-clock budget exhausted'
            path=ROOT/'raw'/r['name'];assert sha(path)==r['source_sha256']
            rgb=np.array(Image.open(path).convert('RGB'));h,w=rgb.shape[:2];assert [w,h]==r['wh']==[2704,1520]
            pred=detector.predict(rgb[:,:,::-1].copy(),classes=[0],conf=.45,imgsz=640,iou=.7,max_det=300,rect=True,augment=False,verbose=False,device=0)[0]
            boxes=pred.boxes.xyxy.cpu().numpy();conf=pred.boxes.conf.cpu().numpy();chosen=None;reason=None
            if len(boxes)==0:reason='no_person'
            else:
                areas=np.prod(boxes[:,2:]-boxes[:,:2],axis=1)
                if previous is None:
                    order=np.argsort(-areas)
                    if len(order)>1 and areas[order[1]]>.65*areas[order[0]]:reason='ambiguous_first_person'
                    else:chosen=int(order[0])
                else:
                    inter=np.maximum(np.minimum(boxes[:,2:],previous[2:])-np.maximum(boxes[:,:2],previous[:2]),0).prod(1)
                    iou=inter/(areas+np.prod(previous[2:]-previous[:2])-inter+1e-8);order=np.argsort(-iou)
                    if iou[order[0]]<.1 or len(order)>1 and iou[order[0]]-iou[order[1]]<.05:reason='ambiguous_subject_continuity'
                    else:chosen=int(order[0])
            row=dict(slot=slot,frame=r['frame'],source_frame_index=r['source_frame_index'],time_seconds=r['time_seconds'],image=str(path),source_sha256=r['source_sha256'],wh=[w,h],detections_xyxy=boxes.tolist(),detection_scores=conf.tolist(),chosen=chosen,selection_issue=reason,anatomical_visibility='unknown_requires_source_review')
            xy=np.full((133,2),np.nan);scores=np.full(133,np.nan);inside=np.zeros(133,bool)
            if chosen is not None:
                previous=boxes[chosen].copy();margin=.04*max(previous[2:]-previous[:2])
                box=np.clip(previous+[-margin,-margin,margin,margin],[0,0,0,0],[w,h,w,h])
                inp,center,scale=pose.preprocess(rgb,box);outputs=pose.inference(inp)
                kp,sc=pose.postprocess(outputs,center,scale);xy,scores=kp[0],sc[0]
                assert xy.shape==(133,2) and scores.shape==(133,) and np.isfinite(xy).all() and np.isfinite(scores).all()
                independent=np.stack([outputs[0].argmax(-1),outputs[1].argmax(-1)],-1)[0]/2/np.array([288,384])*scale+center-scale/2
                valid=scores>0
                parity=float(np.max(np.abs(independent[valid]-xy[valid]))) if valid.any() else 0.
                assert parity<.001,'Native coordinate mapping differs'
                inside=(xy[:,0]>=0)&(xy[:,0]<=w-1)&(xy[:,1]>=0)&(xy[:,1]<=h-1)
                np.savez_compressed(OUT/'responses'/f'{slot:03d}.npz',simcc_x=outputs[0],simcc_y=outputs[1],center=center,scale=scale,box=box,keypoints=xy,scores=scores)
                row.update(body_box=box.tolist(),face_box=face_region_box(xy,scores,(w,h)),native_inverse_affine_parity_px=parity,
                    regions={k:dict(total=len(ids),out_of_image=int((~inside[ids]).sum()),score_min=float(scores[ids].min()),score_median=float(np.median(scores[ids])),score_max=float(scores[ids].max())) for k,ids in regions.items()})
            allxy.append(xy);allscore.append(scores);allinside.append(inside);rows.append(row)
            save(OUT/'observations-partial.json',dict(rows=rows,keypoint_names=names,regions=regions,physical_identity_verified=False))
            save(OUT/'progress.json',dict(updated=now(),state='coverage',pid=os.getpid(),command=sys.argv,done=slot+1,total=99,training=False))
            print(f'coverage {slot+1}/99 selection={reason or "resolved"}',flush=True)
    np.savez_compressed(OUT/'observations.npz',keypoints=np.array(allxy),scores=np.array(allscore),in_image=np.array(allinside),frame_ids=np.array([x['frame'] for x in rows]),source_indices=np.array([x['source_frame_index'] for x in rows]),keypoint_names=np.array(names))
    save(OUT/'observations.json',dict(rows=rows,keypoint_names=names,regions=regions,native_coordinates=True,physical_identity_verified=False,occlusion_is_not_classified_by_confidence=True))
    usable=[x for x in rows if x['chosen'] is not None]
    result=dict(updated=now(),state='complete_actual_visibility_review_pending',frames=99,resolved_subject_frames=len(usable),unresolved=[x['source_frame_index'] for x in rows if x['chosen'] is None],seconds=time.monotonic()-started,
        all_source_frames_preserved=True,native_inverse_affine_parity_max_px=max([x['native_inverse_affine_parity_px'] for x in usable],default=None),
        region_out_of_image={k:sum(x['regions'][k]['out_of_image'] for x in usable) for k in regions},
        frames_with_region_out_of_image={k:sum(x['regions'][k]['out_of_image']>0 for x in usable) for k in regions},
        physical_visibility_validated=False,physical_face_correspondences=0,cameras_estimated=False,SMPLX_fitted=False,training_started=False,quality_accepted=False)
    save(OUT/'result.json',result);save(OUT/'progress.json',dict(updated=now(),state='complete',pid=None,done=99,total=99,training=False))
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':
    try:main()
    except Exception as exc:
        if OUT.exists():save(OUT/'failure.json',dict(updated=now(),error=repr(exc),traceback=traceback.format_exc(),training_started=False))
        raise
