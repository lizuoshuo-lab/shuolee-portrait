"""One frozen semantic-mask pass, CPU only. No feature or camera operations."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='4'
os.environ['OPENBLAS_NUM_THREADS']='2'
from pathlib import Path
import sys,time,json,signal,hashlib,traceback
import numpy as np
from PIL import Image
import match_video10_material123 as base
import match_video10_temporal_bridge123 as bridge

R=base.R; B=base.O; O=bridge.O; N=R/'video10-semantic-role'
read=base.read;save=base.save;sha=base.sha;now=base.now

def run():
    p=read(N/'plan.json');assert not (N/'execution-admission.json').exists(),'Already run: inspect evidence instead of repeating.'
    assert p['budget']['mask_runs']==1 and p['budget']['frames']==33
    frames=p['frames'];regions={r['frame']:r for r in read(O/'regions.json')['rows']};bp=read(B/'plan.json')
    for i,f in enumerate(base.FRAMES):
        regions[f]=dict(frame=f,body=bp['regions']['body_boxes'][i],head=bp['regions']['head_boxes'][i],exclude_background=bp['regions']['background_exclusions'][i])
    frozen=list((B/'run').glob('*.npz'))+list((O/'run').glob('*.npz'))
    frozen += [O/'run/observations.json',O/'run/families.json',O/'regions.json',B/'plan.json',O/'plan.json']
    frozen += [bridge.imgpath(f) for f in frames]
    assert len(frames)==33 and len(set(frames))==33
    hashes={str(q):sha(q) for q in frozen}
    weight=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets/sam2/sam2.1_hiera_tiny.pt')
    assert sha(weight)==p['weight_sha256']
    save(N/'execution-admission.json',dict(updated=now(),script_sha256=sha(Path(__file__)),plan_sha256=sha(N/'plan.json'),frozen_hashes=hashes,regions=regions,weight_sha256=sha(weight),budget_seconds=900))
    (N/'masks').mkdir();(N/'review').mkdir();start=time.monotonic();records=[]
    def deadline(*_):raise TimeoutError('Registered900second CPU mask budget exhausted')
    signal.signal(signal.SIGALRM,deadline);signal.alarm(900)
    def state(s,**kw):save(N/'progress.json',dict(updated=now(),state=s,pid=os.getpid(),elapsed=time.monotonic()-start,frames_done=len(records),training_started=False,**kw))
    try:
        state('loading_frozen_local_sam2_cpu')
        sys.path.insert(0,p['source_path'])
        import torch,cv2
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        torch.set_num_threads(4);cv2.setNumThreads(4)
        net=build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',str(weight),device='cpu',apply_postprocessing=False)
        predictor=SAM2ImagePredictor(net,max_hole_area=0,max_sprinkle_area=0)
        for f in frames:
            rgb=np.asarray(Image.open(bridge.imgpath(f)).convert('RGB'));assert rgb.shape==(1520,2704,3)
            box=np.asarray(regions[f]['body'],np.float32)
            with torch.inference_mode():
                predictor.set_image(rgb)
                masks,scores,_=predictor.predict(box=box,multimask_output=False)
            mask=masks[0].astype(bool);assert mask.shape==rgb.shape[:2]
            out=N/'masks'/f'person_{f:06d}.png';Image.fromarray(mask.astype('uint8')*255).save(out)
            overlay=rgb.astype(np.float32);overlay[mask]=.55*overlay[mask]+.45*np.array([0,210,210])
            edge=cv2.morphologyEx(mask.astype('uint8'),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8)).astype(bool)
            overlay[edge]=[0,255,255];Image.fromarray(overlay.astype('uint8')).resize((1352,760)).save(N/'review'/f'overlay_{f:06d}.jpg',quality=94)
            ys,xs=np.where(mask);bounds=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)] if len(xs) else None
            records.append(dict(frame=f,source=str(bridge.imgpath(f)),source_sha256=hashes[str(bridge.imgpath(f))],body_box=box.tolist(),mask=str(out),mask_sha256=sha(out),mask_pixels=int(mask.sum()),mask_bounds=bounds,score=float(scores[0]),reviewed=False))
            state('segmenting_native_frames');print('MASK',f,int(mask.sum()),float(scores[0]),flush=True)
        pages=[]
        for j in range(11):
            page=Image.new('RGB',(1352,2280),'white');group=records[j*3:j*3+3]
            for k,r in enumerate(group):page.paste(Image.open(N/'review'/f'overlay_{r["frame"]:06d}.jpg'),(0,k*760))
            path=N/'review'/f'page_{j:02d}.jpg';page.save(path,quality=94)
            pages.append(dict(page=j,frames=[r['frame'] for r in group],path=str(path),sha256=sha(path)))
        assert all(sha(Path(q))==h for q,h in hashes.items()),'Frozen inputs changed'
        signal.alarm(0)
        result=dict(updated=now(),state='masks_complete_review_required',frames=len(records),seconds=time.monotonic()-start,records=records,pages=pages,input_hashes_unchanged=True,
          original_feature_coordinates_changed=False,semantic_labels_applied=False,physical_identity_verified=False,training_started=False)
        save(N/'result.json',result);state('masks_complete_review_required',pid_final=None);print(json.dumps({k:result[k] for k in ['state','frames','seconds']}),flush=True)
        v=read(N/'progress.json');v['pid']=None;save(N/'progress.json',v)
    except BaseException as e:
        save(N/'failure.json',dict(updated=now(),error=repr(e),traceback=traceback.format_exc(),frames_done=len(records),records=records));state('mask_generation_failed');raise

if __name__=='__main__':run()
