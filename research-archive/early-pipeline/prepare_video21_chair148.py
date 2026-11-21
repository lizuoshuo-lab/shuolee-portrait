"""Prepare source-only chair-occlusion candidates for the existing validity loss."""
import fcntl
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image

from prepare_video21_native148 import DATA, DIGEST, PINS, PROJECT, REPORT
from run_official_gaussians123 import ASSETS, now, read, save, sha


def main():
    plan_path = REPORT.with_name('seated021-fit148')/'plan.json'
    roster = read(plan_path)
    native = read(REPORT/'result.json')
    inputs = DATA/'data/inputs.json'
    assert roster['video_sha256']==native['video_sha256']==DIGEST
    assert sha(inputs)==native['observation_manifest_sha256']
    assert all(sha(path)==digest for path,digest in PINS.items())
    manifest = read(PROJECT/'configs/server-assets.json')['files']
    names = ('detection/yolov8n.pt','sam2/sam2.1_hiera_tiny.pt')
    for name in names:
        assert sha(ASSETS/name)==next(row['sha256'] for row in manifest if row['path']==name)
    rows = {row['frame']:row for row in read(inputs)['records']}
    assert len(roster['source_frames'])==32
    out, review = DATA/'data/chair-occlusion', REPORT/'chair-review'
    assert not out.exists() and not review.exists()
    with open('/tmp/shuolee-native122-gpu.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        out.mkdir()
        save(review/'plan.json',dict(updated=now(),source_fit_plan_sha256=sha(plan_path),frames=roster['source_frames'],
            roster_is_candidate_pool=True,source_manifest_sha256=sha(inputs),
            policy='Same lower55% conservative ROI, 2px erosion and untouched positives as the original external-furniture probe. Only chair detections, no fitted mesh or Gaussian data.',
            zero_person_positive_removed_required=True,quality_accepted=False,training_started=False))
        sys.path.insert(0,str(PROJECT/'src'))
        from shuolee_portrait import runtime as rt
        from shuolee_portrait.observations import load_detector
        rt.configure(ASSETS,DATA)
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        torch.set_num_threads(2)
        assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>=4*2**30
        detector = load_detector()
        assert detector.names[56]=='chair'
        model = build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',str(ASSETS/names[1]),device='cuda',apply_postprocessing=False)
        segmenter = SAM2ImagePredictor(model,max_hole_area=0,max_sprinkle_area=0)
        results=[]
        start=time.monotonic()
        with torch.inference_mode():
            for frame in roster['source_frames']:
                row=rows[frame]
                assert sha(row['image'])==row['source_sha256'] and sha(row['mask'])==row['mask_sha256']
                with Image.open(row['image']) as im:
                    rgb=np.asarray(im.convert('RGB'))
                with Image.open(row['mask']) as im:
                    person=np.asarray(im.convert('L'))>=128
                h,w=person.shape
                assert rgb.shape==(h,w,3) and [w,h]==row['wh']
                box=np.asarray(row['initial_box_xyxy'])
                bw,bh=box[2:]-box[:2]
                roi=np.rint(np.clip([box[0]-.1*bw,box[1]+.45*bh,box[2]+.1*bw,box[3]],[0,0,0,0],[w,h,w,h])).astype(int)
                x0,y0,x1,y1=roi
                detection=detector.predict(rgb[:,:,::-1].copy(),classes=[56],conf=.25,verbose=False,device=0)[0]
                boxes=detection.boxes.xyxy.cpu().numpy()
                overlap=np.maximum(np.minimum(boxes[:,2:],roi[2:])-np.maximum(boxes[:,:2],roi[:2]),0).prod(axis=1)
                ignored=np.zeros_like(person)
                chair=None
                if len(overlap) and overlap.max()>0:
                    chosen=boxes[int(np.argmax(overlap))]
                    segmenter.set_image(rgb)
                    masks,scores,_=segmenter.predict(box=chosen,multimask_output=False)
                    chair=dict(box=chosen.tolist(),sam_score=float(scores[0]))
                    if scores[0]>=.5:
                        area=cv2.erode(masks[0].astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
                        ignored[y0:y1,x0:x1]=area[y0:y1,x0:x1]&~person[y0:y1,x0:x1]
                assert not np.any(ignored&person)
                path=out/(frame+'.png')
                Image.fromarray((~ignored).astype(np.uint8)*255).save(path)
                crop=np.rint(np.clip([box[0]-64,box[1]-64,box[2]+64,box[3]+64],[0,0,0,0],[w,h,w,h])).astype(int)
                a,b,c,d=crop
                original=rgb[b:d,a:c]
                view=original.copy()
                highlighted=ignored[b:d,a:c]
                view[highlighted]=(0.45*view[highlighted]+0.55*np.array([255,145,0])).astype(np.uint8)
                preview=review/(frame+'.jpg')
                Image.fromarray(np.concatenate([original,view],axis=1)).save(preview,quality=92)
                results.append(dict(frame=frame,chair=chair,validity=str(path),validity_sha256=sha(path),
                    ignored_pixels=int(ignored.sum()),person_positive_removed=0,preview=preview.name,preview_sha256=sha(preview)))
                print('VIDEO21_CHAIR',frame,int(ignored.sum()),flush=True)
        save(review/'result.json',dict(updated=now(),plan_sha256=sha(review/'plan.json'),rows=results,
            seconds=time.monotonic()-start,visual_review_pending=True,depth_verified=False,training_admitted=False))


if __name__=='__main__':
    main()
