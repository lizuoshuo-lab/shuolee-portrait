"""One fixed8 SAM2 prompt control, no detector/pose rerun or training-mask replacement."""
import argparse,fcntl,sys,time,traceback
from pathlib import Path
import numpy as np
from PIL import Image
from run_official_gaussians123 import read,save,sha,now,PROJECT,ASSETS
from train_demo_body_upstream123 import DEMO

BASE=DEMO/'body-gaussian-smplify-eight-dc-reviewedfit'
AUDIT=BASE/'eight-view-surface-audit'
OUT=BASE/'sam2-limb-prompt-control'
OBS=Path('/opt/gs-slam-lab/shuolee-local-surface-123/demo-vhap-fps3/data/observations.npz')
HERE=Path(__file__).resolve()


def prepare():
    assert not OUT.exists()
    assert read(AUDIT/'gpu-result.json')['parameter_tensors_bitwise_unchanged']
    files=[HERE,OBS,BASE/'selection.json',BASE/'data/training-inputs.json',AUDIT/'cpu-result.json',AUDIT/'gpu-result.json',ASSETS/'sam2/sam2.1_hiera_tiny.pt']
    OUT.mkdir()
    save(OUT/'plan.json',dict(updated=now(),state='registered_before_segmentation',frames=read(BASE/'selection.json')['selected_frames'],
        hypothesis='One torso point with a person box omits visible sleeve pixels in frame37. Uniform confident-limb points may recover them without hand-edited masks; held-object ambiguity may remain.',
        only_change='SAM2 prompt positive points: existing torso center plus high-confidence COCO body joints5..16 and midpoints of their eight arm/leg segments. Same native images, box, official tiny weights, no postprocess, multimask_output=False.',
        confidence_min=.5,segments=[[5,7],[7,9],[6,8],[8,10],[11,13],[13,15],[12,14],[14,16]],
        budget=dict(frames=8,once_only=True,runtime_seconds=120,new_weights=False,Gaussian_training_updates=0),
        stop='Stop on source change or baseline mask replay discrepancy, nonfinite outputs or low SAM score. Keep one candidate for every selected frame; no score-based cherry-picking or prompt retries. Review source/mask/candidate across all8 before any admission.',
        risk='Occluded or inaccurate2D joints can include equipment/background. Person segmentation does not identify the unseen body behind an occluder. Confidence is not ground truth.',
        quality_accepted=False,old_masks_overwritten=False,files={str(p):sha(p) for p in files}))
    print('SAM2_LIMB_PROMPT_REGISTERED',flush=True)


def run():
    assert not (OUT/'result.json').exists()
    p=read(OUT/'plan.json')
    amended=(OUT/'paired-control-amendment.json').exists()
    if amended:
        amendment=read(OUT/'paired-control-amendment.json')
        assert sha(HERE)==amendment['script_sha256']
        assert not (OUT/'paired-progress.json').exists(), 'One paired attempt only'
        save(OUT/'paired-progress.json',dict(updated=now(),state='starting',pid=__import__('os').getpid()))
    for path,h in p['files'].items():
        if amended and Path(path)==HERE:assert h==amendment['previous_script_sha256']
        else:assert sha(path)==h,path
    from run_offsetlr_idle123 import gpu_pids
    locks=[open(x,'a+') for x in ['/tmp/shuolee-native122-gpu.lock','/tmp/shuolee-body-export123.lock']]
    for f in locks:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not gpu_pids(),'GPU must be idle'
    import torch
    torch.set_num_threads(4)
    sys.path.insert(0,str(PROJECT/'src'))
    from shuolee_portrait import runtime as rt
    rt.configure(ASSETS)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    model=build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',str(ASSETS/'sam2/sam2.1_hiera_tiny.pt'),device='cuda',apply_postprocessing=False)
    predictor=SAM2ImagePredictor(model,max_hole_area=0,max_sprinkle_area=0)
    obs=dict(np.load(OBS,allow_pickle=False));lookup={f:i for i,f in enumerate(obs['frame_ids'])}
    rows=read(BASE/'data/training-inputs.json')['records'];review={r['frame']:r for r in read(AUDIT/'cpu-result.json')['records']}
    started=time.monotonic();records=[]
    with torch.inference_mode():
        for row in rows:
            frame=row['frame']
            if frame not in p['frames']:continue
            assert sha(row['image'])==row['source_sha256'] and sha(row['mask'])==row['mask_sha256']
            i=lookup[frame];kp=obs['keypoints'][i];scores=obs['scores'][i];box=obs['boxes'][i]
            assert np.allclose(box,row['initial_box_xyxy'],rtol=0,atol=1e-6)
            rgb=np.asarray(Image.open(row['image']).convert('RGB'));old=np.asarray(Image.open(row['mask']).convert('L'))>127
            center=kp[[5,6,11,12]].mean(0)[None]
            predictor.set_image(rgb)
            baseline,bs,_=predictor.predict(box=box,point_coords=center,point_labels=np.ones(1,dtype=np.int32),multimask_output=False)
            discrepancy=int(np.count_nonzero(baseline[0]!=old))
            replay=dict(discrepant_pixels=discrepancy,score=float(bs[0]))
            if not amended:
                save(OUT/(frame+'-baseline-replay.json'),replay)
                assert discrepancy==0,'Preserved baseline mask could not be replayed exactly'
            # The amendment pairs both prompts with one live image embedding.
            # Historical bitwise parity remains unresolved and is not claimed.
            live_original=baseline[0].astype(bool)
            baseline_path=OUT/(frame+'-live-original.png')
            Image.fromarray(live_original.astype('uint8')*255).save(baseline_path)
            pts=[center[0]]
            ids=[j for j in range(5,17) if scores[j]>=p['confidence_min']]
            pts.extend(kp[ids])
            mids=[(a,b) for a,b in p['segments'] if a in ids and b in ids]
            pts.extend((kp[a]+kp[b])*.5 for a,b in mids)
            points=np.asarray(pts,dtype=np.float32)
            masks,sc,_=predictor.predict(box=box,point_coords=points,point_labels=np.ones(len(points),dtype=np.int32),multimask_output=False)
            M=masks[0].astype(bool);assert np.isfinite(sc).all() and sc[0]>=.5
            target=OUT/(frame+'.png');Image.fromarray(M.astype('uint8')*255).save(target)
            b=review[frame]['box'];imgs=[Image.fromarray(rgb).crop(b),Image.fromarray(live_original.astype('uint8')*255).convert('RGB').crop(b),Image.fromarray(M.astype('uint8')*255).convert('RGB').crop(b),Image.fromarray(rgb*live_original[:,:,None]).crop(b),Image.fromarray(rgb*M[:,:,None]).crop(b)]
            w,h=imgs[0].size;sheet=Image.new('RGB',(5*w,h))
            for j,img in enumerate(imgs):sheet.paste(img,(j*w,0))
            dst=Path('/mnt/d/00/记录/demo')/('mask_肢体提示对照_'+frame+'_原图_当前原提示_肢体提示_原人像_新人像.png');sheet.save(dst)
            records.append(dict(frame=frame,baseline_discrepant_pixels=discrepancy,score=float(sc[0]),
                mask=str(target),mask_sha256=sha(target),positive_points=points.tolist(),
                body_joint_ids=ids,midpoint_segments=mids,added_pixels=int((M&~old).sum()),removed_pixels=int((old&~M).sum()),
                live_baseline_mask=str(baseline_path),live_baseline_score=float(bs[0]),
                live_baseline_sha256=sha(baseline_path),paired_added_pixels=int((M&~live_original).sum()),
                paired_removed_pixels=int((live_original&~M).sum()),
                native_comparison=str(dst),native_comparison_sha256=sha(dst)))
            save(OUT/'progress.json',dict(updated=now(),state='running',frames=len(records)))
            assert time.monotonic()-started<120
    for path,h in p['files'].items():
        if amended and Path(path)==HERE:assert h==amendment['previous_script_sha256']
        else:assert sha(path)==h,path
    assert len(records)==8
    save(OUT/'result.json',dict(updated=now(),state='complete_pending_visual_review',records=records,
        seconds=time.monotonic()-started,old_masks_overwritten=False,training_started=False,quality_accepted=False,
        paired_live_control=amended,historical_bitwise_replay_restored=False))
    if amended:save(OUT/'paired-progress.json',dict(updated=now(),state='completed',pid=None,frames=len(records)))
    print('SAM2_LIMB_PROMPT_COMPLETE',len(records),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);a=p.parse_args()
    try:globals()[a.mode]()
    except BaseException as error:
        failure='paired-failure.json' if (OUT/'paired-control-amendment.json').exists() else 'failure.json'
        if OUT.exists():save(OUT/failure,dict(updated=now(),error=repr(error),traceback=traceback.format_exc(),automatic_retry=False))
        raise
