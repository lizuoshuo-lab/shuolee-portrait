"""Complete masks and flip consistency from hash-bound native forward outputs.

This uses the existing observation algorithm and thresholds without repeating
the already completed native detector and unflipped RTMW forward pass.
"""
import json,time
from pathlib import Path
import numpy as np
from PIL import Image
from shuolee_portrait import runtime as rt
from shuolee_portrait.observations import face_region_box

def observe(root,coverage):
    import torch,onnxruntime as ort
    from rtmlib import RTMPose
    from rtmlib.visualization.skeleton.coco133 import coco133
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    inputs=rt.read(root/'data/inputs.json');rows=inputs['records'];meta=rt.read(coverage/'observations.json')['rows']
    source=np.load(coverage/'observations.npz');assert len(rows)==len(meta)==len(source['keypoints'])
    maskdir=root/'data/masks';maskdir.mkdir()
    net=build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',str(rt.ASSETS/'sam2/sam2.1_hiera_tiny.pt'),device='cuda',apply_postprocessing=False)
    segmenter=SAM2ImagePredictor(net,max_hole_area=0,max_sprinkle_area=0)
    norm=rt.read(rt.ASSETS/'rtmw/normalization.json');model=rt.ASSETS/'rtmw/rtmw-l-384x288.onnx'
    class Pose(RTMPose):
        def __init__(self):
            opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1
            self.session=ort.InferenceSession(str(model),sess_options=opts,providers=['CPUExecutionProvider'])
            self.onnx_model=str(model);self.model_input_size=(288,384);self.mean=np.array(norm['mean']);self.std=np.array(norm['std']);self.backend='onnxruntime';self.device='cpu';self.to_openpose=False
    pose=Pose();names=[coco133['keypoint_info'][i]['name'] for i in range(133)]
    flip=np.array([names.index(coco133['keypoint_info'][i]['swap']) if coco133['keypoint_info'][i]['swap'] else i for i in range(133)])
    assert source['keypoint_names'].tolist()==names
    execution=rt.read(coverage/'execution-plan.json')
    for name,digest in execution['model_sha256'].items():assert rt.sha(rt.ASSETS/name)==digest
    points=[];scores=[];disagreement=[];boxes=[];reports=[];start=time.monotonic()
    with torch.inference_mode():
        for i,(row,m) in enumerate(zip(rows,meta)):
            assert row['frame']==m['frame'] and row['source_frame_index']==m['source_frame_index']
            assert rt.sha(row['image'])==m['source_sha256'] and row['wh']==m['wh']
            assert m['chosen'] is not None and m['selection_issue'] is None
            rgb=np.array(Image.open(row['image']).convert('RGB'));H,W=rgb.shape[:2]
            box=np.array(m['body_box']);a=source['keypoints'][i];sa=source['scores'][i]
            other=[b for j,b in enumerate(m['detections_xyxy']) if j!=m['chosen']]
            fb=np.array([W-1-box[2],box[1],W-1-box[0],box[3]])
            b,sb=pose(rgb[:,::-1].copy(),bboxes=fb[None]);b,sb=b[0][flip],sb[0][flip];b[:,0]=W-1-b[:,0]
            kp=(a+b)/2;err=np.linalg.norm(a-b,axis=-1);assert np.isfinite(kp).all()
            segmenter.set_image(rgb);center=kp[[5,6,11,12]].mean(0)[None]
            masks,sc,_=segmenter.predict(box=box,point_coords=center,point_labels=np.ones(1,dtype=np.int32),multimask_output=False)
            mask=masks[0].astype(bool);fraction=float(mask.mean())
            if not .005<fraction<.95 or sc[0]<.5:raise RuntimeError('Coarse SAM2 mask check failed for '+row['frame'])
            path=maskdir/row['name'];Image.fromarray(mask.astype('uint8')*255).save(path)
            facebox=face_region_box(kp,np.minimum(sa,sb),(W,H))
            row.update(mask=str(path),mask_sha256=rt.sha(path),mask_kind='new_SAM2_observation',initial_box_xyxy=box.tolist(),other_person_boxes_xyxy=other,
                       face_box_xyxy=facebox,face_region_source='RTMW face bounds:ROI only,not material identity')
            points.append(kp);scores.append(np.minimum(sa,sb));disagreement.append(err);boxes.append(box)
            reports.append(dict(frame=row['frame'],mask_fraction=fraction,segmentation_score=float(sc[0]),model_scores_are_not_truth=True,source_forward_reused=True))
            rt.save(root/'progress.json',dict(stage='observations',done=i+1,total=len(rows),updated=rt.now(),seconds=time.monotonic()-start))
            print(f'new masks and flip consistency {i+1}/{len(rows)}',flush=True)
    np.savez_compressed(root/'data/observations.npz',frame_ids=np.array([r['frame'] for r in rows]),keypoints=points,scores=scores,flip_disagreement=disagreement,boxes=boxes,joint_names=names)
    rt.save(root/'data/inputs.json',inputs);rt.save(root/'data/training-inputs.json',inputs)
    rt.save(root/'mask-audit.json',dict(records=reports,new_masks=len(rows),old_masks_loaded=False,visual_quality_accepted=False,
                                     reused_native_forward_sha256=rt.sha(coverage/'observations.npz'),source_sha256=rt.sha(__file__),fresh_flipped_inference=True))
