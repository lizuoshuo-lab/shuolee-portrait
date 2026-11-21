"""One native-PNG SAM2 video-mask candidate; upstream network/tracker are unmodified."""
import argparse,fcntl,os,signal,sys,time,traceback
from pathlib import Path
import numpy as np
from PIL import Image
from run_official_gaussians123 import read,save,sha,now,PROJECT,ASSETS
from train_demo_body_upstream123 import DEMO

BASE=DEMO/'body-gaussian-smplify-eight-dc-reviewedfit'
OUT=BASE/'sam2-temporal-mask-control'
DATA=BASE/'data/training-inputs.json'
SAM=PROJECT/'third_party/sam2/sam2'
WEIGHT=ASSETS/'sam2/sam2.1_hiera_tiny.pt'
HERE=Path(__file__).resolve()
RECORD=Path('/mnt/d/00/记录/demo')

def files():
    return [HERE,DATA,BASE/'selection.json',WEIGHT,SAM/'build_sam.py',SAM/'sam2_video_predictor.py',
            SAM/'utils/misc.py',SAM/'modeling/sam2_base.py',SAM/'configs/sam2.1/sam2.1_hiera_t.yaml']

def verify():
    plan=read(OUT/'plan.json')
    for p,h in plan['files'].items():assert sha(p)==h,p
    rows=sorted(read(DATA)['records'],key=lambda r:r['time_seconds'])
    assert [r['frame'] for r in rows]==plan['all_frame_ids']
    for row in rows:
        assert sha(row['image'])==row['source_sha256'],row['frame']
        assert sha(row['mask'])==row['mask_sha256'],row['frame']
    return plan,rows

def prepare():
    assert not OUT.exists(),'Never overwrite or repeat this candidate'
    source_rows=read(DATA)['records']
    rows=sorted(source_rows,key=lambda r:r['time_seconds']);selected=read(BASE/'selection.json')['selected_frames']
    assert len(rows)==59 and len(selected)==8 and len(set(selected))==8
    assert [r['frame'] for r in rows]==sorted(r['frame'] for r in rows)
    assert np.all(np.diff([r['time_seconds'] for r in rows])>0)
    source_files=files()
    assert all(p.is_file() for p in source_files)
    for row in rows:
        assert sha(row['image'])==row['source_sha256'] and sha(row['mask'])==row['mask_sha256']
    seed=min(selected);seed_row=next(r for r in rows if r['frame']==seed)
    OUT.mkdir()
    save(OUT/'plan.json',dict(updated=now(),state='registered_before_inference',
      hypothesis='The per-frame mask predictor changes person/held-device semantics and cuts visible sleeves. Original SAM2 video object memory from one existing complete person mask may preserve semantic identity across the sequence.',
      only_change='Per-image segmentation -> original SAM2VideoPredictor add_new_mask and forward/backward propagation; one frozen mask seed. No new points, detectors, model downloads, morphology, pose/camera or Gaussian updates.',
      seed_rule='Earliest chronological member of the already frozen8 training frames. This is a reproducible mask seed, not SMPL body/shape initialization or result-driven frame selection.',
      seed_frame=seed,seed_index=[r['frame'] for r in rows].index(seed),seed_sha256=seed_row['mask_sha256'],
      all_frame_ids=[r['frame'] for r in rows],evaluation_frames=selected,
      input_manifest_order=[r['frame'] for r in source_rows],
      explicit_slot_mapping=[dict(video_slot=i,frame=r['frame'],source_time=r['time_seconds'],
           manifest_slot=next(j for j,x in enumerate(source_rows) if x['frame']==r['frame'])) for i,r in enumerate(rows)],
      loader='In-process PNG loader adapter uses original _load_img_as_tensor and original mean/std. No JPEG re-encoding, input resize/crop edit, coordinate change or repository source modification.',
      settings=dict(device='cuda',dtype='float32',matmul_tf32=False,apply_postprocessing=False,
        offload_video_to_cpu=True,offload_state_to_cpu=False,output_logit_threshold=0.0),
      budget=dict(candidates=1,frames=59,forward_once=True,reverse_once=True,total_seconds=240,GPU_memory_GiB=12),
      stop='Stop on occupied GPU, changed inputs, nonfinite logits, missing frame/object identity, time or memory budget. No repeat, seed scan or parameter sweep. Review all8 before adoption; if candidate passes, review remaining frames before any all59 claim.',
      acceptance='Visible face/hat/skin/clothing/shoes must be preserved and held equipment consistently excluded. A segmentation model is not occlusion ground truth: device-over-body pixels remain unknown, not known background or inferred visible person.',
      risk='Memory can propagate initial errors or switch to equipment. Low-resolution model features cannot guarantee subpixel boundaries. The initial mask is a prior automatic estimate, not hand-labeled ground truth.',
      quality_accepted=False,training_started=False,old_masks_overwritten=False,
      files={str(p):sha(p) for p in source_files}))
    print('TEMPORAL_MASK_PLAN_REGISTERED',seed,flush=True)

def run():
    plan,rows=verify()
    assert not (OUT/'progress.json').exists(),'This candidate may run only once'
    from run_offsetlr_idle123 import gpu_pids
    locks=[open(x,'a+') for x in ('/tmp/shuolee-native122-gpu.lock','/tmp/shuolee-body-export123.lock')]
    for h in locks:fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not gpu_pids(),'GPU occupied'
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('240-second candidate budget')))
    signal.alarm(plan['budget']['total_seconds'])
    started=time.monotonic()
    save(OUT/'progress.json',dict(updated=now(),state='loading',pid=os.getpid(),frames=0,training_updates=0))
    import torch
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    sys.path.insert(0,str(PROJECT/'src'))
    from shuolee_portrait import runtime as rt
    rt.configure(ASSETS)
    import sam2.sam2_video_predictor as video
    from sam2.build_sam import build_sam2_video_predictor
    from sam2.utils.misc import _load_img_as_tensor
    predictor=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_t.yaml',str(WEIGHT),
                                       device='cuda',apply_postprocessing=False)
    original_loader=video.load_video_frames
    def load_pngs(video_path,image_size,offload_video_to_cpu,img_mean=(.485,.456,.406),
                  img_std=(.229,.224,.225),async_loading_frames=False,compute_device=None):
        assert video_path==str(DATA) and offload_video_to_cpu and not async_loading_frames
        images=torch.empty((len(rows),3,image_size,image_size),dtype=torch.float32)
        for i,row in enumerate(rows):
            tensor,h,w=_load_img_as_tensor(row['image'],image_size)
            assert [w,h]==row['wh']==[2704,1520] and torch.isfinite(tensor).all()
            images[i]=tensor
        images.sub_(torch.tensor(img_mean,dtype=torch.float32)[:,None,None])
        images.div_(torch.tensor(img_std,dtype=torch.float32)[:,None,None])
        save(OUT/'loader-result.json',dict(updated=now(),frames=len(rows),frame_ids=[r['frame'] for r in rows],
          original_native_wh=[w,h],model_tensor_shape=list(images.shape),PNG_no_reencoding=True,
          original_loader_helper_sha256=sha(SAM/'utils/misc.py'),normalized_tensor_finite=bool(torch.isfinite(images).all()),
          scope='Official predictor internally resizes to1024square; native images and output pixel identity remain unchanged. Not a feature-localization accuracy claim.'))
        return images,h,w
    seed_index=plan['seed_index'];seed=np.array(Image.open(rows[seed_index]['mask']).convert('L'))>127
    (OUT/'masks').mkdir();records={};emissions=0
    with torch.inference_mode():
        video.load_video_frames=load_pngs
        try:state=predictor.init_state(str(DATA),offload_video_to_cpu=True,offload_state_to_cpu=False)
        finally:video.load_video_frames=original_loader
        idx,ids,seed_logits=predictor.add_new_mask(state,frame_idx=seed_index,obj_id=1,mask=seed)
        assert idx==seed_index and ids==[1] and torch.isfinite(seed_logits).all()
        save(OUT/'seed-result.json',dict(frame=plan['seed_frame'],mask_sha256=plan['seed_sha256'],
          output_changed_pixels=int(np.count_nonzero((seed_logits[0,0].cpu().numpy()>0)!=seed)),
          scope='Official mask downsample/upsample can alter boundaries. Input seed is preserved; no bitwise output parity asserted.'))
        for reverse in (False,True):
            for idx,ids,logits in predictor.propagate_in_video(state,start_frame_idx=seed_index,reverse=reverse):
                assert ids==[1] and logits.shape==(1,1,1520,2704) and torch.isfinite(logits).all()
                row=rows[idx];M=logits[0,0].cpu().numpy()>0;frame=row['frame'];emissions+=1
                path=OUT/'masks'/(frame+'.png')
                if frame in records:
                    assert idx==seed_index and np.array_equal(M,np.array(Image.open(path))>127)
                    continue
                assert .005<float(M.mean())<.95,'Mask absent or implausible area'
                Image.fromarray(M.astype(np.uint8)*255).save(path)
                old=np.array(Image.open(row['mask']).convert('L'))>127
                records[frame]=dict(frame=frame,slot=idx,reverse=reverse,mask=str(path),sha256=sha(path),
                    mask_fraction=float(M.mean()),original_mask_iou=float((M&old).sum()/(M|old).sum()),
                    added_pixels=int((M&~old).sum()),removed_pixels=int((old&~M).sum()))
                save(OUT/'progress.json',dict(updated=now(),state='propagating',pid=os.getpid(),frames=len(records),
                  last_frame=frame,reverse=reverse,seconds=time.monotonic()-started,training_updates=0))
                assert time.monotonic()-started<240
                assert torch.cuda.max_memory_allocated()<12*1024**3
                assert not set(gpu_pids())-{os.getpid()}
    assert len(records)==59
    verify();signal.alarm(0)
    save(OUT/'result.json',dict(updated=now(),state='completed_pending_visual_review',
       records=[records[r['frame']] for r in rows],seconds=time.monotonic()-started,
       emission_count=emissions,peak_GPU_GiB=torch.cuda.max_memory_allocated()/1024**3,
       original_repo_modified=False,training_started=False,quality_accepted=False,old_masks_overwritten=False))
    save(OUT/'progress.json',dict(updated=now(),state='completed',pid=None,frames=59,training_updates=0))
    print('TEMPORAL_MASK_COMPLETE',len(records),flush=True)

def sheets():
    result=read(OUT/'result.json');plan,rows=verify()
    crops={r['frame']:r['box'] for r in read(BASE/'eight-view-surface-audit/cpu-result.json')['records']}
    masks={r['frame']:r for r in result['records']};pages=[]
    for row in rows:
        f=row['frame']
        if f not in plan['evaluation_frames']:continue
        raw=np.array(Image.open(row['image']).convert('RGB'))
        old=np.array(Image.open(row['mask']).convert('L'))>127
        new=np.array(Image.open(masks[f]['mask']).convert('L'))>127
        b=crops[f];w=b[2]-b[0];h=b[3]-b[1]
        images=[Image.fromarray(raw),Image.fromarray(old.astype('uint8')*255).convert('RGB'),
          Image.fromarray(new.astype('uint8')*255).convert('RGB'),Image.fromarray(raw*old[:,:,None]),Image.fromarray(raw*new[:,:,None])]
        sheet=Image.new('RGB',(5*w,h))
        for k,im in enumerate(images):sheet.paste(im.crop(b),(k*w,0))
        path=RECORD/('mask_时序传播_'+f+'_原图_原mask_时序mask_原人像_时序人像.png');sheet.save(path)
        pages.append(dict(frame=f,path=str(path),sha256=sha(path),native_crop=b,resampling=False,text_added=False))
    save(OUT/'review-images.json',pages)
    print('TEMPORAL_MASK_REVIEW_IMAGES',len(pages),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run','sheets']);args=parser.parse_args()
    try:globals()[args.mode]()
    except BaseException as e:
        if OUT.exists():
            save(OUT/'failure.json',dict(updated=now(),mode=args.mode,error=repr(e),traceback=traceback.format_exc(),retry=False))
            if args.mode=='run':save(OUT/'progress.json',dict(updated=now(),state='failed_preserved',pid=None))
        raise
