"""One native five-frame furniture observation, reusing the existing policy.

No fit, camera, original person mask, loss weight or old experiment is changed.
Candidate unknown pixels must be inspected before any training integration.
"""
import fcntl
import os
import signal
import sys
import time
import traceback
from pathlib import Path
import numpy as np
from PIL import Image
from run_official_gaussians123 import read, save, sha, now, PROJECT, ASSETS

ROOT = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/cross-subject-sh0-124')
SRC = ROOT/'indoor-input'
OUT = ROOT/'indoor-visibility124'
H = Path(__file__).resolve().parent
RECORD = Path('/mnt/d/00/记录/008')


def prepare():
    assert not OUT.exists(), 'Read the existing run; never duplicate or overwrite it.'
    rows = read(SRC/'data/inputs.json')['records']; assert len(rows) == 111
    slots = np.linspace(0,len(rows)-1,5).astype(int).tolist()
    schedule = [rows[i] for i in slots]
    sources = [Path(__file__), H/'prepare_capture118_external_occluders.py',
        SRC/'data/inputs.json', SRC/'data/observations.npz',
        PROJECT/'src/shuolee_portrait/observations.py', PROJECT/'configs/server-assets.json']
    manifest = read(PROJECT/'configs/server-assets.json')['files']
    for rel in ['detection/yolov8n.pt','sam2/sam2.1_hiera_tiny.pt']:
        p=ASSETS/rel; assert sha(p)==next(x['sha256'] for x in manifest if x['path']==rel)
        sources.append(p)
    for r in schedule:
        assert sha(r['image'])==r['source_sha256'] and sha(r['mask'])==r['mask_sha256']
    OUT.mkdir(); (OUT/'masks').mkdir()
    save(OUT/'plan.json',dict(updated=now(),state='registered',subject='indoor',
        causal_variable='Observation support for visible furniture; no person fitting or loss update.',
        schedule=schedule,slots=slots,selection='Five equal chronological positions, fixed before inference.',
        policy=dict(classes=[56,57,60],detector_confidence=.25,sam_score=.5,erosion_kernel=5,
            lower_box_fraction_start=.45,horizontal_box_margin=.1,remove_person_positive=True),
        policy_source=str(H/'prepare_capture118_external_occluders.py'),
        native_input_wh=[2704,1520],model_internal_resize='Official detector/SAM preprocessing only.',
        unknown_definition='Furniture candidate pixels near lower body, excluding person foreground. Not depth truth.',
        budget=dict(native_frames=5,seconds=240,GPU=1,fitting_updates=0,Gaussian_updates=0,new_download_bytes=0),
        original_masks_unchanged=True,old118_run_not_restarted=True,
        sources={str(p):sha(p) for p in sources},quality_accepted=False,training_admitted=False))
    print('NATIVE_FURNITURE_OBSERVATION_REGISTERED', [r['frame'] for r in schedule], flush=True)


def run():
    plan=read(OUT/'plan.json')
    assert not (OUT/'progress.json').exists(), 'Do not relaunch this registered observation.'
    for p,d in plan['sources'].items(): assert sha(p)==d,p
    lock=open('/tmp/shuolee-native122-gpu.lock','a+'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.monotonic()
    save(OUT/'launch.json',dict(updated=now(),pid=os.getpid(),command=sys.argv,
        proc_start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],single_gpu_lock=True))
    save(OUT/'progress.json',dict(updated=now(),state='loading_existing_weights',pid=os.getpid(),done=0,total=5))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('240s registered observation budget')))
    signal.alarm(240)
    import torch, cv2
    torch.set_num_threads(4); cv2.setNumThreads(1)
    sys.path.insert(0,str(PROJECT/'src'))
    from shuolee_portrait import runtime as rt
    rt.configure(ASSETS)
    from shuolee_portrait.observations import load_detector
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    detector=load_detector().to('cuda')
    assert {detector.names[i] for i in plan['policy']['classes']}=={'chair','couch','dining table'}
    network=build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',str(ASSETS/'sam2/sam2.1_hiera_tiny.pt'),device='cuda',apply_postprocessing=False)
    segmenter=SAM2ImagePredictor(network,max_hole_area=0,max_sprinkle_area=0)
    records=[]
    with torch.inference_mode():
        for i,r in enumerate(plan['schedule']):
            assert sha(r['image'])==r['source_sha256'] and sha(r['mask'])==r['mask_sha256']
            rgb=np.array(Image.open(r['image']).convert('RGB')); person=np.array(Image.open(r['mask']).convert('L'))>127
            height,width=person.shape;assert [width,height]==plan['native_input_wh']
            prediction=detector.predict(rgb[:,:,::-1].copy(),classes=[56,57,60],conf=.25,verbose=False,device='cuda')[0]
            boxes=prediction.boxes.xyxy.cpu().numpy();classes=prediction.boxes.cls.cpu().numpy();scores=prediction.boxes.conf.cpu().numpy()
            furniture=np.zeros_like(person);objects=[]
            if len(boxes):
                segmenter.set_image(rgb)
                for box,cls,score in zip(boxes,classes,scores):
                    mm,ss,_=segmenter.predict(box=box,multimask_output=False)
                    include=float(ss[0])>=.5
                    if include:furniture|=mm[0].astype(bool)
                    objects.append(dict(label=detector.names[int(cls)],box=box.tolist(),detector_confidence=float(score),sam_score=float(ss[0]),included=include))
            furniture=cv2.erode(furniture.astype('uint8'),np.ones((5,5),np.uint8)).astype(bool)&~person
            b=np.array(r['initial_box_xyxy']);bw,bh=b[2]-b[0],b[3]-b[1]
            roi=np.rint(np.clip([b[0]-.1*bw,b[1]+.45*bh,b[2]+.1*bw,b[3]],[0,0,0,0],[width,height,width,height])).astype(int)
            region=np.zeros_like(person);region[roi[1]:roi[3],roi[0]:roi[2]]=True
            unknown=furniture&region;assert not np.any(unknown&person)
            mask_path=OUT/'masks'/(r['frame']+'.png');Image.fromarray(unknown.astype('uint8')*255).save(mask_path)
            # Four columns, same integer crop: original, person, unknown, unknown overlay.
            crop=np.rint(np.clip([b[0]-.15*bw,b[1]-.05*bh,b[2]+.15*bw,b[3]],[0,0,0,0],[width,height,width,height])).astype(int)
            overlay=rgb.copy();overlay[unknown]=(.45*rgb[unknown]+.55*np.array([255,145,0])).astype('uint8')
            parts=[Image.fromarray(x).crop(tuple(crop)).convert('RGB') for x in [rgb,person.astype('uint8')*255,unknown.astype('uint8')*255,overlay]]
            tile=Image.new('RGB',(parts[0].width*4,parts[0].height))
            for k,part in enumerate(parts):tile.paste(part,(k*part.width,0))
            image_path=RECORD/f'分析_室内遮挡124_{r["frame"]}_原图_人物mask_候选未知区_叠加.png'
            assert not image_path.exists();tile.save(image_path)
            records.append(dict(frame=r['frame'],source_image=r['image'],source_sha256=r['source_sha256'],
                person_mask=r['mask'],person_mask_sha256=r['mask_sha256'],unknown_mask=str(mask_path),
                unknown_sha256=sha(mask_path),unknown_pixels=int(unknown.sum()),positive_person_removed=0,
                roi_xyxy=roi.tolist(),objects=objects,evidence_image=str(image_path),evidence_sha256=sha(image_path),crop_xyxy=crop.tolist()))
            save(OUT/'progress.json',dict(updated=now(),state='observing',pid=os.getpid(),done=i+1,total=5,seconds=time.monotonic()-started))
            print('OBSERVED',r['frame'],int(unknown.sum()),flush=True)
    for p,d in plan['sources'].items():assert sha(p)==d,p
    signal.alarm(0)
    save(OUT/'result.json',dict(updated=now(),state='native_observation_review_pending',records=records,
        seconds=time.monotonic()-started,original_inputs_unchanged=True,new_fit_calls=0,new_Gaussian_steps=0,
        actual_visual_review=False,unknown_is_not_depth_truth=True,training_admitted=False))
    save(OUT/'progress.json',dict(updated=now(),state='completed_review_pending',pid=None,done=5,total=5))
    print('NATIVE_FURNITURE_OBSERVATION_COMPLETE',flush=True)


if __name__=='__main__':
    try:prepare() if sys.argv[1]=='prepare' else run()
    except BaseException as e:
        if OUT.exists():save(OUT/'failure.json',dict(updated=now(),error=repr(e),traceback=traceback.format_exc(),auto_retry=False))
        raise
