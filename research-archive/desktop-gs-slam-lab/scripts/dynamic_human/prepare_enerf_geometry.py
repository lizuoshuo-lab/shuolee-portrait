"""Native first-frame train-only geometric audit and initialization observations."""
from pathlib import Path
import argparse,json,itertools,time
import cv2,numpy as np
from PIL import Image,ImageDraw
from enerf_native_loader import EnerfNativeDataset
from audit_enerf_input import digest

def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--out',required=True);p.add_argument('--root');p.add_argument('--frame',type=int,default=0);p.add_argument('--offsets');a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'geometry-audit.json').exists():raise FileExistsError('Completed audit exists')
    ds=EnerfNativeDataset(a.manifest,'train',a.root);obs=[];features=[];sift=cv2.SIFT_create(nfeatures=7000)
    offsets=json.loads(Path(a.offsets).read_text())['offsets'] if a.offsets else {}
    for i,r in enumerate(ds.rows):
        if r.get('common_frame',r['frame'])!=a.frame+offsets.get(r['camera'],0):continue
        o=ds[i];obs.append(o)
        mask=((o['foreground_matte']>.9)&o['valid_pixels']).astype(np.uint8)*255
        kp,de=sift.detectAndCompute(cv2.cvtColor(o['rgb'],cv2.COLOR_RGB2GRAY),mask)
        features.append((np.float64([x.pt for x in kp]),de))
        np.savez_compressed(out/f"train-camera-{o['camera']}.npz",rgb=o['rgb'],matte=o['foreground_matte'],valid=o['valid_pixels'],K=o['K'],w2c=o['w2c'],source_frame=o['frame'],source_sha256=o['source_sha256'])
        print('CAMERA',o['camera'],'features',len(kp),flush=True)
    assert len(obs)==15
    save(out/'source-observations.json',dict(common_frame=a.frame,source_frames={o['camera']:o['frame'] for o in obs},source_rgb_sha256={o['camera']:o['source_sha256'] for o in obs},offset_hypothesis_sha256=digest(a.offsets) if a.offsets else None,split='train',source_manifest_sha256=digest(a.manifest)))
    centers=np.array([-o['w2c'][:3,:3].T@o['w2c'][:3,3] for o in obs])
    # Three nearest cameras per camera, unique unordered pairs.
    pairs=set()
    for i in range(len(obs)):
        for j in np.argsort(np.linalg.norm(centers-centers[i],axis=1))[1:4]:pairs.add(tuple(sorted((i,int(j)))))
    matcher=cv2.BFMatcher();reports=[];points=[];colors=[]
    for i,j in sorted(pairs):
        oi,oj=obs[i],obs[j];pi,di=features[i];pj,dj=features[j]
        matches=[m for m,n in matcher.knnMatch(di,dj,k=2) if m.distance<.7*n.distance]
        x=pi[[m.queryIdx for m in matches]];y=pj[[m.trainIdx for m in matches]]
        Ri,ti=oi['w2c'][:3,:3],oi['w2c'][:3,3];Rj,tj=oj['w2c'][:3,:3],oj['w2c'][:3,3]
        R=Rj@Ri.T;t=tj-R@ti
        tx=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]])
        F=np.linalg.inv(oj['K']).T@tx@R@np.linalg.inv(oi['K'])
        h1=np.c_[x,np.ones(len(x))];h2=np.c_[y,np.ones(len(y))];l2=h1@F.T;l1=h2@F
        distance=np.abs((h2*l2).sum(1))/np.sqrt((l1[:,:2]**2).sum(1)+(l2[:,:2]**2).sum(1))
        good=distance<1.5;xg,yg=x[good],y[good]
        if len(xg)<4:continue
        P1=oi['K']@oi['w2c'][:3];P2=oj['K']@oj['w2c'][:3]
        h=cv2.triangulatePoints(P1,P2,xg.T,yg.T).T;xyz=h[:,:3]/h[:,3:]
        z1=xyz@Ri.T+ti;z2=xyz@Rj.T+tj
        p1=z1@oi['K'].T;p1=p1[:,:2]/p1[:,2:];p2=z2@oj['K'].T;p2=p2[:,:2]/p2[:,2:]
        err=np.maximum(np.linalg.norm(p1-xg,axis=1),np.linalg.norm(p2-yg,axis=1))
        v1=xyz-centers[i];v2=xyz-centers[j];cos=(v1*v2).sum(1)/np.linalg.norm(v1,axis=1)/np.linalg.norm(v2,axis=1)
        valid=(z1[:,2]>.1)&(z2[:,2]>.1)&(err<2)&(cos<np.cos(np.deg2rad(2)))&np.isfinite(xyz).all(1)
        points.append(xyz[valid]);uv=np.rint(xg[valid]).astype(int);colors.append(oi['rgb'][uv[:,1],uv[:,0]]/255.)
        reports.append(dict(cameras=[oi['camera'],oj['camera']],ratio_matches=len(matches),epipolar_inliers=int(good.sum()),epipolar_median_px=float(np.median(distance)),triangulated=int(valid.sum()),retained_reprojection_median_px=float(np.median(err[valid])) if valid.any() else None))
        print('PAIR',reports[-1],flush=True)
    pts=np.concatenate(points);col=np.concatenate(colors);assert len(pts)>500
    np.savez_compressed(out/'train-sift-initialization.npz',points=pts,colors=col)
    canvas=Image.new('RGB',(1920,1080),(240,240,240));draw=ImageDraw.Draw(canvas)
    for k,o in enumerate(obs):
        thumb=Image.fromarray(o['rgb']).resize((384,216));canvas.paste(thumb,(k%5*384,k//5*360))
        m=Image.fromarray((o['foreground_matte']*255).astype('uint8')).convert('RGB').resize((224,126));canvas.paste(m,(k%5*384,k//5*360+224));draw.text((k%5*384+236,k//5*360+244),'camera '+o['camera'],fill='black')
    canvas.save(out/'train-input-contact.png')
    save(out/'geometry-audit.json',dict(status='train_only_native_rectified_observations_and_feature_geometry_checked',train_cameras=[o['camera'] for o in obs],frame=a.frame,resolution=[1920,1080],source_manifest=str(Path(a.manifest).resolve()),calibration_size_sentinel='H/W=-1 means unspecified per official easy_utils.py; K unchanged',undistortion='same K, native size, valid bilinear support, matte remapped identically',pairs=reports,point_count=len(pts),point_quantiles=np.quantile(pts,[.005,.5,.995],axis=0).tolist(),camera_centers=centers.tolist(),geometry='Train camera foreground SIFT correspondences only, no author surface, heldout RGB or heldout matte',not_proven=['metric scale','hardware synchronization timestamps','entire-sequence lack of motion blur'],matte_semantics='Author background subtraction, includes people and held objects; not human-part segmentation'))

if __name__=='__main__':main()
