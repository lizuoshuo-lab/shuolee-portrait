"""Bounded, manually prompted SAM2 input-mask candidates; no training/eval RGB changed."""
from pathlib import Path
import json,time,torch,numpy as np,os
from PIL import Image,ImageDraw
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from audit_enerf_input import digest
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
OUT=LAB/'outputs/dynamic-human-enerf-mask-quality-038/train15'
INPUT=OUT
variant=os.environ.get('SAM2_VARIANT','boxes-points')
if variant=='boxes-only':
    OUT=INPUT/'boxes-only';OUT.mkdir(exist_ok=True)
# Boxes/points selected by inspection of training-only photographs at 1/3 preview size.
# y coordinates exclude the 32px sheet header; every coordinate is multiplied by 3.
P={
'02':[[[226,80,342,299],[239,96]],[[222,33,326,342],[281,132]],[[224,153,254,201],[239,174]]],
'03':[[[330,67,424,299],[365,140]],[[405,30,548,331],[491,130]],[[397,151,458,198],[425,173]]],
'04':[[[229,69,325,301],[270,140]],[[313,34,450,325],[399,130]],[[298,152,365,199],[333,176]]],
'05':[[[306,83,424,303],[392,247]],[[293,40,390,344],[338,139]],[[295,158,328,203],[310,181]]],
'07':[[[316,74,405,280],[353,146]],[[252,12,350,321],[298,110]],[[253,132,342,189],[269,158]]],
'08':[[[251,100,338,311],[275,164]],[[278,59,387,352],[338,167]],[[274,175,320,227],[294,203]]],
'09':[[[315,88,410,308],[343,165]],[[375,43,506,336],[452,151]],[[370,160,423,207],[395,185]]],
'10':[[[271,105,366,322],[305,170]],[[328,64,450,354],[395,158]],[[323,178,371,226],[346,203]]],
'11':[[[280,112,381,326],[326,177]],[[332,73,459,360],[409,169]],[[330,184,376,231],[354,207]]],
'13':[[[259,89,372,302],[275,112]],[[266,43,368,344],[328,142]],[[263,160,300,207],[282,182]]],
'14':[[[291,92,383,304],[319,159]],[[328,47,443,341],[388,145]],[[322,163,369,210],[345,183]]],
'15':[[[251,91,366,307],[267,109]],[[252,46,352,345],[302,139]],[[250,162,285,211],[268,183]]],
'16':[[[335,89,422,304],[364,160]],[[363,44,477,342],[426,143]],[[358,161,404,209],[382,183]]],
'17':[[[257,93,366,309],[273,116]],[[273,49,376,349],[327,147]],[[264,163,305,214],[284,189]]]
}
labels=['person_dark_clothing','person_light_clothing','held_objects']
prompts={'01':[dict(label=label,box=box) for label,box in zip(labels,[[718,266,990,904],[919,170,1303,1016],[914,494,1042,636]])]}
prompts.update({cam:[dict(label=label,box=(np.array(bp[0])*3).tolist(),point=(np.array(bp[1])*3).tolist()) for label,bp in zip(labels,ps)] for cam,ps in P.items()})
if variant=='boxes-only':
    for ps in prompts.values():
        for p in ps:p.pop('point',None)
assert not (OUT/'segmentation.json').exists();assert not torch.cuda.is_available()
(OUT/'prompt-protocol.json').write_text(json.dumps(dict(frame=750,prompts=prompts,scope='Manual training-view boxes/points, hard masks only; no heldout photograph used; no model training or RGB modification',goal_complete=False),indent=2)+'\n')
torch.set_num_threads(4);start=time.time()
model=build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',str(LAB/'models/sam2/sam2.1_hiera_tiny.pt'),device='cpu',apply_postprocessing=False)
predictor=SAM2ImagePredictor(model,max_hole_area=0,max_sprinkle_area=0);records=[]
with torch.inference_mode():
    for cam,ps in sorted(prompts.items()):
        rgb=np.array(Image.open(INPUT/f'camera{cam}-source.png'));predictor.set_image(rgb)
        masks=[];rows=[]
        for p in ps:
            kwargs={} if 'point' not in p else dict(point_coords=np.array([p['point']]),point_labels=np.array([1]))
            m,s,l=predictor.predict(box=np.array(p['box']),multimask_output=False,**kwargs)
            mask=m[0].astype(bool);masks.append(mask)
            Image.fromarray((mask*255).astype(np.uint8)).save(OUT/f"camera{cam}-{p['label']}.png")
            rows.append(dict(**p,predicted_iou=float(s[0]),pixels=int(mask.sum())))
        union=np.any(masks,axis=0);Image.fromarray((union*255).astype(np.uint8)).save(OUT/f'camera{cam}-candidate.png')
        candidate=np.rint(rgb.astype(np.float32)*union[...,None]+127.5*(~union[...,None])).astype(np.uint8)
        Image.fromarray(candidate).save(OUT/f'camera{cam}-candidate-composite.png')
        matte=np.load(INPUT/f'camera{cam}-released.npy')
        old=np.rint(rgb.astype(np.float32)*matte[...,None]+127.5*(1-matte[...,None])).astype(np.uint8)
        boxes=np.array([p['box'] for p in ps]);x0,y0=boxes[:,:2].min(0);x1,y1=boxes[:,2:].max(0);w=x1-x0;h=y1-y0
        panel=Image.new('RGB',(3*w,h+32),'white');d=ImageDraw.Draw(panel)
        for j,(label,arr) in enumerate([('Original photo',rgb),('Released matte',old),('SAM2 mask candidate',candidate)]):
            panel.paste(Image.fromarray(arr[y0:y1,x0:x1]),(j*w,32));d.text((j*w+4,10),f'{cam}: {label}',fill='black')
        panel.save(OUT/f'camera{cam}-comparison.png')
        records.append(dict(camera=cam,objects=rows,union_pixels=int(union.sum()),mask_sha256=digest(OUT/f'camera{cam}-candidate.png'),comparison_bbox=[int(v) for v in [x0,y0,x1,y1]]))
        print('SEGMENTED',cam,int(union.sum()),flush=True)
(OUT/'segmentation.json').write_text(json.dumps(dict(status='15_training_candidates_complete_visual_review_pending',records=records,elapsed_seconds=time.time()-start,device='cpu',model_confidence_is_not_accuracy=True,goal_complete=False),indent=2)+'\n')
