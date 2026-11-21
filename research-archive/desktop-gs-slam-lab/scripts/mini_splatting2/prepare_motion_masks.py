"""Manually prompted SAM2 masks for the observed walking background person.

Prompts are based only on training photographs. The foreground subject is not
removed. Masks are candidate supervision, reviewed before any training use.
"""
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image,ImageDraw
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

LAB=Path(__file__).resolve().parents[2]
ROOT=LAB/'outputs/mini-splatting2-mip-depth-001'
OUT=ROOT/'motion-masks-001'
IMAGES=LAB/'outputs/mini-splatting2-computer00-001/hover-sfm-001/dataset/images'
# Normalized full-image boxes and positive garment points, selected from the
# training contact sheet before model prediction. No holdout frame is read.
PROMPTS={
1:([0,.40,.08,.79],[.012,.67]),2:([0,.38,.16,.76],[.04,.62]),
3:([0,.35,.20,.74],[.05,.57]),4:([0,.34,.21,.73],[.07,.55]),
5:([0,.34,.23,.73],[.10,.55]),6:([0,.35,.25,.73],[.12,.55]),
7:([0,.35,.27,.73],[.13,.55]),9:([0,.35,.28,.70],[.13,.53]),
10:([0,.34,.27,.69],[.12,.52]),11:([0,.34,.26,.67],[.12,.50]),
12:([0,.35,.23,.65],[.09,.51]),13:([0,.35,.16,.64],[.04,.50]),
14:([0,.38,.045,.63],[.009,.51])}

def write(p,x):p.write_text(json.dumps(x,indent=2))

def main():
    OUT.mkdir(exist_ok=False);torch.set_num_threads(4);torch.manual_seed(0)
    checkpoint=LAB/'models/sam2/sam2.1_hiera_tiny.pt'
    protocol=dict(status='candidates_pending_visual_review',prompt_source='training-contact.jpg and optical flow epipolar audit',
        prompts=PROMPTS,weights_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        model='SAM2.1 Hiera Tiny',device='CPU',holdout_read=False,mask_use='exclude walking background person from static reconstruction RGB loss only',
        caveat='semantic manual annotation added; does not reconstruct the moving person and may leave unobserved background',
        dilation_native_pixels=4,masks_approved_for_training=False)
    write(OUT/'protocol.json',protocol)
    model=build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',str(checkpoint),device='cpu',apply_postprocessing=False)
    predictor=SAM2ImagePredictor(model,max_hole_area=0,max_sprinkle_area=0);rows=[];thumbs=[];start=time.time()
    with torch.inference_mode():
        for frame in range(59):
            if frame%8==0:continue
            im=np.asarray(Image.open(IMAGES/f'{frame:06d}.jpg').convert('RGB'));h,w=im.shape[:2];raw=np.zeros((h,w),bool);score=None
            if frame in PROMPTS:
                box,pt=PROMPTS[frame];predictor.set_image(im)
                m,s,_=predictor.predict(box=np.array(box)*[w,h,w,h],point_coords=np.array([pt])*[w,h],point_labels=np.array([1]),multimask_output=False)
                raw=m[0].astype(bool);score=float(s[0])
            mask=cv2.dilate(raw.astype('uint8'),np.ones((9,9),dtype='uint8'))>0
            Image.fromarray(mask.astype('uint8')*255).save(OUT/f'{frame:06d}.png')
            row=dict(frame=f'{frame:06d}',pixels=int(mask.sum()),fraction=float(mask.mean()),sam_score=score);rows.append(row)
            if frame in PROMPTS:
                overlay=im.copy();overlay[mask]=(overlay[mask]*.35+np.array([255,0,0])*.65).astype('uint8')
                Image.fromarray(np.concatenate([im,overlay],axis=1)).save(OUT/f'{frame:06d}-review.jpg')
                thumbs.append((frame,Image.fromarray(overlay).resize((180,320))))
            print(row,flush=True)
    contact=Image.new('RGB',(5*180,3*342),'white');d=ImageDraw.Draw(contact)
    for j,(frame,im) in enumerate(thumbs):contact.paste(im,((j%5)*180,(j//5)*342+20));d.text(((j%5)*180+4,(j//5)*342+3),str(frame),fill='black')
    contact.save(OUT/'all-masks-review.jpg')
    write(OUT/'manifest.json',dict(status='candidates_ready',protocol=protocol,records=rows,elapsed_seconds=time.time()-start))

if __name__=='__main__':main()
