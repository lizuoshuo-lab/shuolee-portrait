from pathlib import Path
import json,time,torch,numpy as np,cv2
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');OUT=LAB/'outputs/dynamic-human-enerf-mask-quality-038'
assert not torch.cuda.is_available();torch.set_num_threads(4)
rgb=np.array(Image.open(LAB/'outputs/dynamic-human-4k4d-official-035/native-evaluation/camera01-frame000750-source.png'))
released=np.load(LAB/'outputs/dynamic-human-4k4d-official-035/native-evaluation/camera01-frame000750-raw.npz')['matte']
n,labels,stats,centers=cv2.connectedComponentsWithStats((released>.95).astype(np.uint8))
largest=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]));x,y,w,h,area=stats[largest]
box=[max(0,int(x)-20),max(0,int(y)-20),min(1920,int(x+w)+20),min(1080,int(y+h)+20)]
model=build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',str(LAB/'models/sam2/sam2.1_hiera_tiny.pt'),device='cpu',apply_postprocessing=False)
pred=SAM2ImagePredictor(model,max_hole_area=0,max_sprinkle_area=0)
with torch.inference_mode():
    pred.set_image(rgb);m,score,low=pred.predict(box=np.array(box),multimask_output=False)
mask=m[0].astype(bool);previous=np.array(Image.open(OUT/'camera01-union-sam2.png'))>0
Image.fromarray((mask*255).astype(np.uint8)).save(OUT/'camera01-one-box-sam2.png')
im=np.rint(rgb.astype(np.float32)*mask[...,None]+127.5*(1-mask[...,None])).astype(np.uint8)
Image.fromarray(im).save(OUT/'camera01-one-box-composite.png')
report=dict(status='group_prompt_probe_complete_visual_review_pending',box=box,core_component_area=int(area),pixels=int(mask.sum()),
    overlap_with_three_prompt_candidate=float((mask&previous).sum()/(mask|previous).sum()),predicted_iou=float(score[0]),
    scope='Largest released-matte core component defines only a box, not finalsegmentation. Comparison mask is another SAM2candidate, not groundtruth.',goal_complete=False)
(OUT/'group-prompt-probe.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
