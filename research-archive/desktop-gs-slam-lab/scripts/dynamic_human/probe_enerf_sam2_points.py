from pathlib import Path
import json,time,torch,numpy as np,cv2
from PIL import Image,ImageDraw
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');OUT=LAB/'outputs/dynamic-human-enerf-mask-quality-038'
assert not torch.cuda.is_available();torch.set_num_threads(4);start=time.time()
rgb=np.array(Image.open(LAB/'outputs/dynamic-human-4k4d-official-035/native-evaluation/camera01-frame000750-source.png'))
released=np.load(LAB/'outputs/dynamic-human-4k4d-official-035/native-evaluation/camera01-frame000750-raw.npz')['matte']
core=(released>.95).astype(np.uint8);n,labels,stats,centers=cv2.connectedComponentsWithStats(core)
largest=stats[1:,cv2.CC_STAT_AREA].max();ids=[i for i in range(1,n) if stats[i,cv2.CC_STAT_AREA]>=.05*largest]
core=np.isin(labels,ids).astype(np.uint8);dist=cv2.distanceTransform(core,cv2.DIST_L2,5)
yy,xx=np.where(dist>=6);xy=np.c_[xx,yy];strength=np.minimum(dist[yy,xx],30.)
selected=[];distance=np.full(len(xy),np.inf)
for k in range(12):
    priority=strength if k==0 else np.sqrt(distance)*strength**.25
    index=int(np.argmax(priority));point=xy[index];selected.append(point)
    distance=np.minimum(distance,((xy-point)**2).sum(1));distance[index]=0
ys,xs=np.where(core);box=np.array([max(0,xs.min()-20),max(0,ys.min()-20),min(1920,xs.max()+21),min(1080,ys.max()+21)])
model=build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',str(LAB/'models/sam2/sam2.1_hiera_tiny.pt'),device='cpu',apply_postprocessing=False)
pred=SAM2ImagePredictor(model,max_hole_area=0,max_sprinkle_area=0);rows=[];masks=[]
with torch.inference_mode():
    pred.set_image(rgb)
    for point in selected:
        m,score,low=pred.predict(box=box,point_coords=np.array([point]),point_labels=np.array([1]),multimask_output=False)
        mask=m[0].astype(bool);inside=np.zeros_like(mask);inside[box[1]:box[3],box[0]:box[2]]=True
        overlap=float((mask&core.astype(bool)).sum()/max(1,mask.sum()));outside=float((mask&~inside).sum()/max(1,mask.sum()))
        accepted=bool(score[0]>=.8 and overlap>=.5 and outside<=.03)
        if accepted:masks.append(mask)
        rows.append(dict(point=point.tolist(),predicted_iou=float(score[0]),pixels=int(mask.sum()),released_core_overlap=overlap,outside_prompt_box_fraction=outside,accepted=accepted))
assert masks
union=np.any(masks,axis=0);previous=np.array(Image.open(OUT/'camera01-union-sam2.png'))>0
Image.fromarray((union*255).astype(np.uint8)).save(OUT/'camera01-points-sam2.png')
im=np.rint(rgb.astype(np.float32)*union[...,None]+127.5*(1-union[...,None])).astype(np.uint8)
Image.fromarray(im).save(OUT/'camera01-points-composite.png')
report=dict(status='automatic_multi_prompt_candidate_visual_review_pending',box=box.tolist(),points=rows,
    candidate_union_pixels=int(union.sum()),released_core_coverage=float((union&core.astype(bool)).sum()/core.sum()),
    overlap_with_three_box_candidate=float((union&previous).sum()/(union|previous).sum()),elapsed_seconds=time.time()-start,
    scope='Customautomaticprompts fromreleased-mattecore, noRGBalteration. Candidate confidence/overlap filters do not establishsemantic correctness.',goal_complete=False)
(OUT/'point-prompt-probe.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
