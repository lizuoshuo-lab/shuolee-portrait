"""Re-triangulate clothing support using only the new full-scene observations."""
import json,sys,time
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares
from PIL import Image,ImageDraw
import shuolee_common as c
from prepare_fullscene_person_075 import R,S,intrinsics


def main():
    out=R/'clothing-support';out.mkdir(exist_ok=True);assert not (out/'surface.npz').exists()
    # read_write_model is vendored beside this module.
    import read_write_model as rw
    _,images,points=rw.read_model(str(S/'refined_sparse'),ext='.bin')
    byname={im.name:im for im in images.values()};rows=c.read(R/'data/training-inputs.json')['records']
    E=np.array([r['w2c'] for r in rows]);K=np.array(rows[0]['K']);P=K@E[:,:3]
    kcv=K.copy();kcv[:2,2]-=.5
    kr,distortion=intrinsics(c.read(S/'cameras.json')['intrinsics'],True)
    centers=-np.einsum('nji,nj->ni',E[:,:3,:3],E[:,:3,3]);tracks={};boxes=[];start=time.monotonic()
    for i,row in enumerate(rows):
        mask=cv2.imread(row['mask'],0)>127;d=cv2.distanceTransform(mask.astype('uint8'),cv2.DIST_L2,5)
        yy,xx=np.where(mask);ymin=int(yy.min());yheight=max(1,int(yy.max())-ymin)
        boxes.append([xx.min(),ymin,xx.max()+1,ymin+yheight+1])
        im=byname[f'raw_{row["source_frame_index"]:06d}.png'];uv=cv2.undistortPoints(im.xys[:,None].astype(float)-.5,kr,distortion,P=kcv)[:,0]
        coords=np.rint(uv).astype(int)
        for pid,p,ip in zip(im.point3D_ids,uv,coords):
            if pid<0:continue
            x,y=ip;mar=float(d[y,x]) if 0<=y<mask.shape[0] and 0<=x<mask.shape[1] else 0.
            fraction=float((p[1]-ymin)/yheight)
            tracks.setdefault(int(pid),[]).append([i,*p,mar,fraction])
        if i%35==0:print(json.dumps(dict(stage='new_sfm_observations',frame=i,seconds=time.monotonic()-start)),flush=True)
    def project(x,ids):
        p=np.einsum('nij,j->ni',E[ids,:3,:3],x)+E[ids,:3,3];q=p@K.T
        return q[:,:2]/q[:,2:]-.5,p[:,2]
    def triangulate(o):
        ids=o[:,0].astype(int);uv=o[:,1:3]+.5
        A=np.stack([uv[:,0,None]*P[ids,2]-P[ids,0],uv[:,1,None]*P[ids,2]-P[ids,1]],1).reshape(-1,4)
        _,_,v=np.linalg.svd(A,full_matrices=False);x=v[-1,:3]/v[-1,3]
        return least_squares(lambda x:(project(x,ids)[0]-o[:,1:3]).ravel(),x,loss='soft_l1',f_scale=2,max_nfev=40).x
    accepted=[];xyz=[];rgb=[];obs=[];candidates=0
    for pid,values in tracks.items():
        o=np.array(values);fraction=float(np.median(o[:,4]))
        if len(o)<8 or o[:,3].min()<=0 or np.median(o[:,3])<8 or not .17<fraction<.88:continue
        ids=o[:,0].astype(int);x=triangulate(o);rays=x-centers[ids];rays/=np.linalg.norm(rays,axis=1,keepdims=True)
        angle=float(np.rad2deg(np.arccos(np.clip((rays@rays.T).min(),-1,1))))
        if angle<8:continue
        candidates+=1;pred,z=project(x,ids);held=[]
        for begin in range(0,175,30):
            test=(ids>=begin)&(ids<begin+30)
            if not test.any() or (~test).sum()<4:continue
            candidate=triangulate(o[~test]);p,zz=project(candidate,ids[test]);held.extend(np.linalg.norm(p-o[test,1:3],axis=1).tolist())
            if (zz<=0).any():held.append(1e6)
        if len(held)<5 or np.median(held)>=2 or np.quantile(held,.9)>=5 or (z<=0).any():continue
        j=len(xyz);xyz.append(x);rgb.append(points[pid].rgb);obs.extend([[j,*a[:3]] for a in o])
        accepted.append(dict(point_id=pid,observations=len(o),ray_angle_deg=angle,held_median=float(np.median(held)),held_p90=float(np.quantile(held,.9))))
    assert 20<len(xyz)<200000,len(xyz)
    np.savez_compressed(out/'surface.npz',xyz=np.asarray(xyz),rgb=np.asarray(rgb),observations=np.asarray(obs),point_ids=[a['point_id'] for a in accepted])
    c.save(out/'audit.json',dict(created=c.stamp(),source='074 fresh full-scene SfM image observations',
        candidates=candidates,accepted_tracks=len(xyz),observations=len(obs),seconds=time.monotonic()-start,
        old_sparse_or_subject_fit_loaded=False,new_triangulation=True,tracks=accepted,
        image_camera_manifest_sha256=c.sha(R/'data/training-inputs.json'),
        source_images_bin_sha256=c.sha(S/'refined_sparse/images.bin'),
        selection='8views,interior8pxmedian,8degrays;blocked30rawframeholdout median<2px,p90<5px',
        limitation='Observed clothing texture supports only; not full clothing surface or independent 3DGT'))
    observed=np.asarray(obs)
    for i in [0,69,135]:
        im=Image.open(rows[i]['image']).convert('RGB');box=boxes[i];im=im.crop(box);scale=800/im.height
        im=im.resize((round(im.width*scale),800));d=ImageDraw.Draw(im)
        for _,_,x,y in observed[observed[:,1].astype(int)==i]:
            x,y=(x-box[0])*scale,(y-box[1])*scale;d.ellipse((x-2,y-2,x+2,y+2),outline='#00eaff',width=2)
        im.save(out/f'observed-{i+1:06d}.jpg',quality=95)
    print(json.dumps(dict(finished=True,candidates=candidates,accepted=len(xyz),seconds=time.monotonic()-start)),flush=True)


if __name__=='__main__':main()
