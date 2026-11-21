"""Training-only motion evidence from RAFT and calibrated epipolar geometry.

This is an audit, not a trusted segmentation and not a dynamic reconstruction.
No masks enter training until their actual spatial support has been reviewed.
"""
import json
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from torchvision.models.optical_flow import raft_small,Raft_Small_Weights
from local_refinement import ROOT,BASE,cameras,write_json

OUT=ROOT/'motion-audit-001'
W,H=384,672

def skew(t):
    x,y,z=t;return np.array([[0,-z,y],[z,0,-x],[-y,x,0]],dtype='float64')

def intrinsics(c):
    return np.array([[c['fx']*W/c['width'],0,W/2],[0,c['fy']*H/c['height'],H/2],[0,0,1]])

def evidence(c1,c2,flow,back):
    yy,xx=np.mgrid[:H,:W].astype('float32');u=xx+flow[...,0];v=yy+flow[...,1]
    reverse=cv2.remap(back,u,v,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    fb=np.linalg.norm(flow+reverse,axis=-1)
    A=c1['viewmat'][0].cpu().numpy();B=c2['viewmat'][0].cpu().numpy();T=B@np.linalg.inv(A)
    K1,K2=intrinsics(c1),intrinsics(c2);F=np.linalg.inv(K2).T@skew(T[:3,3])@T[:3,:3]@np.linalg.inv(K1)
    p=np.stack([xx,yy,np.ones_like(xx)],-1);q=np.stack([u,v,np.ones_like(xx)],-1)
    line=p@F.T;distance=np.abs((q*line).sum(-1))/np.maximum(np.linalg.norm(line[...,:2],axis=-1),1e-10)
    valid=(u>3)&(u<W-4)&(v>3)&(v<H-4)&(fb<1.5)
    valid[:4]=False;valid[-4:]=False;valid[:,:4]=False;valid[:,-4:]=False
    return distance.astype('float32'),valid

@torch.inference_mode()
def main():
    OUT.mkdir(exist_ok=False);torch.set_num_threads(4)
    protocol=dict(status='running',method='RAFT Small flow + calibrated epipolar residual + forward-backward check',
        weights='torchvision Raft_Small_Weights.DEFAULT',resolution=[W,H],
        forward_backward_max_pixels=1.5,epipolar_min_pixels=3.,minimum_supporting_neighbors=2,
        training_frames_only=True,heldout_images_used=False,masks_used_for_training=False,
        caveat='motion, rolling shutter, correspondence errors and calibration errors may all violate the constraint; visual review required')
    write_json(OUT/'protocol.json',protocol)
    weights=Raft_Small_Weights.DEFAULT
    model=raft_small(weights=weights,progress=False).cuda().eval();transform=weights.transforms()
    cams=[c for c in cameras() if not c['is_test']];images=[]
    for c in cams:
        im=cv2.imread(str(BASE/'hover-sfm-001/dataset/images'/f'{c["img_name"]}.jpg'))
        images.append(cv2.resize(im,(W,H),interpolation=cv2.INTER_AREA))
    votes=[np.zeros((H,W),dtype='uint8') for c in cams];support=[v.copy() for v in votes]
    start=time.time();pairs=[]
    for i in range(len(cams)-1):
        ts=[torch.from_numpy(cv2.cvtColor(images[j],cv2.COLOR_BGR2RGB)).permute(2,0,1)[None].cuda() for j in [i,i+1]]
        a,b=transform(*ts)
        flow=model(a,b,num_flow_updates=12)[-1][0].permute(1,2,0).cpu().numpy()
        back=model(b,a,num_flow_updates=12)[-1][0].permute(1,2,0).cpu().numpy()
        counts=[]
        for j,k,f,r in [(i,i+1,flow,back),(i+1,i,back,flow)]:
            dist,valid=evidence(cams[j],cams[k],f,r);bad=valid&(dist>3.)
            votes[j]+=bad.astype('uint8');support[j]+=valid.astype('uint8')
            counts.append(dict(source=cams[j]['img_name'],target=cams[k]['img_name'],valid_fraction=float(valid.mean()),
                epipolar_median=float(np.median(dist[valid])),epipolar_p95=float(np.quantile(dist[valid],.95)),bad_fraction=float(bad.mean())))
        pairs+=counts
        write_json(OUT/'progress.json',dict(status='running',completed_pairs=i+1,total_pairs=len(cams)-1,elapsed_seconds=time.time()-start))
        print(i+1,counts,flush=True)
    records=[]
    for c,im,v,s in zip(cams,images,votes,support):
        raw=(v>=2)&(s>=2)
        mask=cv2.morphologyEx(raw.astype('uint8'),cv2.MORPH_CLOSE,np.ones((5,5),dtype='uint8'))
        mask=cv2.dilate(mask,np.ones((3,3),dtype='uint8'))>0
        np.savez_compressed(OUT/f'{c["img_name"]}.npz',motion_candidate=mask,votes=v,support=s)
        overlay=im.copy();overlay[mask]=(overlay[mask]*.35+np.array([0,0,255])*.65).astype('uint8')
        cv2.imwrite(str(OUT/f'{c["img_name"]}-review.jpg'),np.concatenate([im,overlay],axis=1))
        records.append(dict(frame=c['img_name'],raw_motion_fraction=float(raw.mean()),expanded_motion_fraction=float(mask.mean()),
                            twice_supported_fraction=float((s>=2).mean())))
    write_json(OUT/'manifest.json',dict(protocol=protocol,status='complete',records=records,pairs=pairs,
        elapsed_seconds=time.time()-start,peak_allocated_gb=torch.cuda.max_memory_allocated()/2**30,
        masks_approved_for_training=False))
    print('COMPLETE',time.time()-start,flush=True)

if __name__=='__main__':main()
