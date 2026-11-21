"""Training-view-only relative depth, aligned to robust SfM inverse z-depth.

Masks combine fit residuals, flip consistency, depth discontinuities and
cross-view consistency. These tests reject some unreliable pixels; they do
not turn monocular predictions or SfM into measured ground truth.
"""
import sys
import json
import time
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares
import torch
from local_refinement import LAB, BASE, ROOT, cameras, write_json


def align(pred,xy,inv,ids):
    values=cv2.remap(pred,xy[:,0].astype('float32').reshape(-1,1),xy[:,1].astype('float32').reshape(-1,1),cv2.INTER_LINEAR).ravel()
    train=(ids%5!=0);val=~train
    if train.sum()<80 or val.sum()<20:raise ValueError('Insufficient robust SfM points for fit validation')
    xm,xs=np.median(values[train]),np.std(values[train])
    ym,ys=np.median(inv[train]),np.std(inv[train])
    x=(values-xm)/max(xs,1e-6);y=(inv-ym)/max(ys,1e-6)
    initial=np.polyfit(x[train],y[train],1)
    fit=least_squares(lambda ab:ab[0]*x[train]+ab[1]-y[train],initial,loss='huber',f_scale=.3)
    a=fit.x[0]*ys/max(xs,1e-6);b=fit.x[1]*ys+ym-a*xm
    aligned=a*pred+b
    relative=np.abs(a*values[val]+b-inv[val])/np.maximum(inv[val],1e-6)
    report=dict(a=float(a),b=float(b),fit_points=int(train.sum()),validation_points=int(val.sum()),
                validation_median_relative_error=float(np.median(relative)),validation_p90_relative_error=float(np.quantile(relative,.9)))
    report['accepted']=a>0 and report['validation_median_relative_error']<.3
    return aligned.astype('float32'),report


def main():
    out=ROOT/'depth-priors';out.mkdir(exist_ok=False)
    sys.path.insert(0,str(LAB/'repos/Depth-Anything-V2'))
    sys.path.insert(0,str(LAB/'repos/hierarchical-3d-gaussians/preprocess'))
    from depth_anything_v2.dpt import DepthAnythingV2
    import read_write_model as rw
    _,images,points=rw.read_model(str(BASE/'hover-sfm-001/dataset/sparse/0'),ext='.bin')
    byname={im.name:im for im in images.values()}
    model=DepthAnythingV2(encoder='vits',features=64,out_channels=[48,96,192,384])
    weights=ROOT/'sources/depth_anything_v2_vits.pth'
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True));model=model.cuda().eval()
    cams=[c for c in cameras() if not c['is_test']]
    priors=[];records=[];start=time.time()
    with torch.no_grad():
        for c in cams:
            name=c['img_name'];im=byname[name+'.jpg']
            rgb=cv2.imread(str(BASE/'hover-sfm-001/dataset/images'/f'{name}.jpg'))
            raw=model.infer_image(rgb,input_size=518)
            flipped=np.fliplr(model.infer_image(np.fliplr(rgb).copy(),input_size=518)).copy()
            ids=np.asarray(im.point3D_ids);valid=np.array([int(p)>=0 and int(p) in points and points[int(p)].error<=1. and len(points[int(p)].image_ids)>=4 for p in ids])
            ids=ids[valid];xy=im.xys[valid]
            xyz=np.array([points[int(p)].xyz for p in ids]);R=rw.qvec2rotmat(im.qvec)
            z=(xyz@R.T+im.tvec)[:,2]
            good=(z>.2)&(xy[:,0]>1)&(xy[:,0]<c['width']-2)&(xy[:,1]>1)&(xy[:,1]<c['height']-2)
            inverse,record=align(raw,xy[good],1/z[good],ids[good])
            other,other_record=align(flipped,xy[good],1/z[good],ids[good])
            consistency=np.abs(inverse-other)/np.maximum((np.abs(inverse)+np.abs(other))/2,1e-5)
            weight=np.clip(1-consistency/.15,0,1)
            gradient=np.maximum(np.abs(cv2.Sobel(inverse,cv2.CV_32F,1,0,ksize=3)),np.abs(cv2.Sobel(inverse,cv2.CV_32F,0,1,ksize=3)))/np.maximum(np.abs(inverse),1e-5)
            edge=cv2.dilate((gradient>.3).astype('uint8'),np.ones((5,5),np.uint8))>0
            weight[(inverse<=0)|edge]=0
            if not record['accepted'] or not other_record['accepted']:weight[:]=0
            priors.append(dict(c=c,inverse=inverse,weight=weight,rgb=rgb.astype('float32')/255))
            record.update(frame=name,flip=other_record,pre_consistency_fraction=float((weight>0).mean()))
            records.append(record)
            np.savez_compressed(out/f'{name}-raw.npz',relative_inverse=raw.astype('float16'))
            print(name,record['validation_median_relative_error'],record['accepted'],flush=True)
    del model;torch.cuda.empty_cache()
    # Cross-view checks use only training cameras and images. Held-out RGB is
    # excluded both as a loss and as a confidence-mask source.
    for idx,item in enumerate(priors):
        c=item['c'];inv=item['inverse'];h,w=inv.shape
        yy,xx=np.mgrid[:h,:w];z=1/np.maximum(inv,1e-5)
        xyz=np.stack([(xx-c['width']/2)/c['fx']*z,(yy-c['height']/2)/c['fy']*z,z],-1).astype('float32').reshape(-1,3)
        V=c['viewmat'][0].cpu().numpy();world=(xyz-V[:3,3])@V[:3,:3]
        support=np.zeros((h,w),dtype=bool)
        neighbors=[]
        for delta in [-3,3]:
            j=idx+delta
            if not 0<=j<len(priors):continue
            source=priors[j];sc=source['c'];S=sc['viewmat'][0].cpu().numpy()
            pts=world@S[:3,:3].T+S[:3,3];zs=pts[:,2].reshape(h,w)
            u=(pts[:,0]/np.maximum(pts[:,2],1e-5)*sc['fx']+sc['width']/2).reshape(h,w).astype('float32')
            v=(pts[:,1]/np.maximum(pts[:,2],1e-5)*sc['fy']+sc['height']/2).reshape(h,w).astype('float32')
            observed=cv2.remap(source['inverse'],u,v,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
            sourceweight=cv2.remap(source['weight'],u,v,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
            color=cv2.remap(source['rgb'],u,v,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
            rel=np.abs(observed*zs-1)
            valid=(zs>.2)&(observed>0)&(sourceweight>.25)&(rel<.12)&(np.mean(np.abs(color-item['rgb']),-1)<.15)
            support|=valid;neighbors.append(sc['img_name'])
        weight=item['weight']*support
        np.savez_compressed(out/f'{c["img_name"]}.npz',inverse_depth=inv.astype('float16'),weight=weight.astype('float16'))
        records[idx].update(cross_view_neighbors=neighbors,accepted_pixel_fraction=float((weight>0).mean()))
        if idx in [0,12,25,38,50]:
            a=np.clip(inv/np.quantile(inv[inv>0],.98)*255,0,255).astype('uint8')
            colored=cv2.applyColorMap(a,cv2.COLORMAP_INFERNO)
            original=(item['rgb']*255).astype('uint8');mask=original.copy();mask[weight==0]=(mask[weight==0]*.2).astype('uint8')
            panel=np.concatenate([original,colored,mask],axis=1)
            cv2.imwrite(str(out/f'{c["img_name"]}-preview.jpg'),panel)
    write_json(out/'manifest.json',dict(status='complete',model='Depth Anything V2 Small',
        checkpoint_sha256='715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378',
        repository_commit='a561b849ebae10a6f5ef49e26c83cbbcd36c71bf',input_size=518,
        frame_count=len(cams),appearance_holdout_excluded=True,elapsed_seconds=time.time()-start,records=records,
        depth_definition='inverse camera z in arbitrary SfM units',fit_validation='fixed point-id modulo-5 split within each training view',
        masks='positive robust scale, sparse fit validation, aligned flip agreement, boundaries, nearest training-view depth and color agreement',
        caveat='prediction and self-consistency, not depth ground truth; moving objects or mutually consistent wrong geometry may survive'))
    print('Complete',len(cams),time.time()-start,flush=True)


if __name__=='__main__':main()
