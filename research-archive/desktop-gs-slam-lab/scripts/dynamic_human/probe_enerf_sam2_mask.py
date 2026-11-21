from pathlib import Path
import os,time,json,hashlib,numpy as np,torch
from PIL import Image,ImageDraw
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
OUT=LAB/'outputs/dynamic-human-enerf-mask-quality-038'
SRC=LAB/'outputs/dynamic-human-4k4d-official-035/native-evaluation'
assert not torch.cuda.is_available(),'This bounded probe must not consume the trainingGPU'
torch.set_num_threads(4);torch.manual_seed(0);start=time.time()
boxes=[dict(label='person_dark_clothing',box=[718,266,990,904]),dict(label='person_light_clothing',box=[919,170,1303,1016]),dict(label='held_toys',box=[914,494,1042,636])]
protocol=dict(status='fixed_before_segmentation',camera='01',frame=750,boxes=boxes,
    prompts='Three boxes manually selected from already inspected training-camera raw photo. No prompt points or human mask painted.',
    method='OfficialSAM2.1HieraTiny CPU float32 image predictor; original1024encoder resolution, native1920x1080mask output; single-mask output, no hole/sprinkle cleanup.',
    scope='One training photograph only; releasedmatte used for comparison, not as modelprompt. Must visuallyreviewallcandidateobjects and background before use.',
    source_rgb_sha256=hashlib.sha256((SRC/'camera01-frame000750-source.png').read_bytes()).hexdigest(),goal_complete=False)
assert not (OUT/'sam2-pilot.json').exists()
(OUT/'sam2-pilot-protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
model=build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',str(LAB/'models/sam2/sam2.1_hiera_tiny.pt'),device='cpu',apply_postprocessing=False)
predictor=SAM2ImagePredictor(model,max_hole_area=0,max_sprinkle_area=0)
rgb=np.array(Image.open(SRC/'camera01-frame000750-source.png').convert('RGB'));input_copy=rgb.copy()
z=np.load(SRC/'camera01-frame000750-raw.npz');released=z['matte'];assert np.array_equal(rgb,z['source'])
masks=[];rows=[]
with torch.inference_mode():
    predictor.set_image(rgb)
    for prompt in boxes:
        predicted,scores,logits=predictor.predict(box=np.array(prompt['box']),multimask_output=False)
        mask=predicted[0].astype(bool);assert mask.shape==(1080,1920)
        masks.append(mask);Image.fromarray((mask*255).astype(np.uint8)).save(OUT/f"camera01-{prompt['label']}-sam2.png")
        np.savez_compressed(OUT/f"camera01-{prompt['label']}-sam2.npz",mask=mask,scores=scores,low_resolution_logits=logits)
        rows.append(dict(**prompt,predicted_iou=float(scores[0]),pixels=int(mask.sum())))
assert np.array_equal(rgb,input_copy)
union=np.any(masks,axis=0);Image.fromarray((union*255).astype(np.uint8)).save(OUT/'camera01-union-sam2.png')
new=(rgb.astype(np.float32)*union[...,None]+127.5*(1-union[...,None])).round().clip(0,255).astype(np.uint8)
old=np.array(Image.open(SRC/'camera01-frame000750-reference.png'))
Image.fromarray(new).save(OUT/'camera01-sam2-composite.png')
for name,box in [('forearm',[736,430,824,610]),('full-person',[710,160,1310,1020])]:
    x0,y0,x1,y1=box;w,h=x1-x0,y1-y0
    canvas=Image.new('RGB',(3*w,h+32),'white');draw=ImageDraw.Draw(canvas)
    for j,(label,arr) in enumerate([('Photo',rgb),('Released matte',old),('SAM2 candidate',new)]):
        canvas.paste(Image.fromarray(arr[y0:y1,x0:x1]),(j*w,32));draw.text((j*w+4,9),label,fill='black')
    canvas.save(OUT/f'camera01-{name}-sam2-comparison.png')
points=[(775,495),(771,515),(767,535),(767,555)]
report=dict(status='one_training_camera_segmentation_complete_visual_review_pending',objects=rows,elapsed_seconds=time.time()-start,
    pixels_unchanged=True,device='cpu',mask_union_pixels=int(union.sum()),
    manual_interior_samples=[dict(x=x,y=y,released_matte=float(released[y,x]),candidate_mask=bool(union[y,x])) for x,y in points],
    quality='PredictedIoU is modelconfidence, not measuredaccuracy. Hard segmentation is not a validated soft alpha matte.',goal_complete=False)
(OUT/'sam2-pilot.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
