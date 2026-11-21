"""Recover unused-frame localization with OpenCV matching and fixed-map PnP.

The Windows COLMAP pair matcher crashed after feature extraction. Existing
features and the frozen calibrated map are reused; no original data is changed.
"""
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

import cv2
import numpy as np

LAB=Path(__file__).resolve().parents[2];ROOT=LAB/'outputs/mini-splatting2-mip-depth-001'
BASE=LAB/'outputs/mini-splatting2-computer00-001/hover-sfm-001';OUT=ROOT/'additional-frames-001'

def main():
    cv2.setNumThreads(4);cv2.setRNGSeed(17)
    output=OUT/'localization-pnp';output.mkdir(exist_ok=False);images=output/'images';images.mkdir()
    state=json.loads((OUT/'manifest.json').read_text());state['status']='matching_failed';state['failure']='Windows COLMAP matches_importer exited3221225477 after creating CPU matchers'
    (OUT/'manifest-colmap-match-failed.json').write_text(json.dumps(state,indent=2))
    state.update(status='localizing_fixed_map_pnp',localization='OpenCV SIFT descriptor matching + fixed intrinsic PnP RANSAC; original map and camera parameters are immutable')
    def save(): (OUT/'manifest.json').write_text(json.dumps(state,indent=2))
    save()
    sys.path.insert(0,str(LAB/'repos/hierarchical-3d-gaussians/preprocess'));import read_write_model as rw
    cams,ims,points=rw.read_model(str(BASE/'raw_sparse/0'),ext='.bin')
    undistcams,_,_=rw.read_model(str(BASE/'dataset/sparse/0'),ext='.bin')
    cam=next(iter(cams.values()));u=next(iter(undistcams.values()));assert cam.model=='SIMPLE_RADIAL' and u.model=='PINHOLE'
    f,cx,cy,k=cam.params;K=np.array([[f,0,cx],[0,f,cy],[0,0,1.]])
    distortion=np.array([k,0,0,0,0.]);fx,fy,ux,uy=u.params;newK=np.array([[fx,0,ux],[0,fy,uy],[0,0,1.]])
    mapx,mapy=cv2.initUndistortRectifyMap(K,distortion,None,newK,(u.width,u.height),cv2.CV_32FC1)
    fixed_paths=[BASE/'raw_sparse/0'/x for x in ['cameras.bin','images.bin','points3D.bin']]
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in fixed_paths}
    with sqlite3.connect(f'file:{OUT/"database.db"}?mode=ro',uri=True) as db:
        names=dict(db.execute('SELECT name,image_id FROM images').fetchall())
        def array(table,idx,dtype):
            rows,cols,data=db.execute(f'SELECT rows,cols,data FROM {table} WHERE image_id=?',(idx,)).fetchone()
            return np.frombuffer(data,dtype=dtype).reshape(rows,cols).copy()
        desc={n:array('descriptors',i,np.uint8).astype(np.float32) for n,i in names.items()}
        xy={n:array('keypoints',i,np.float32)[:,:2] for n,i in names.items()}
    image_by_name={im.name:im for im in ims.values()};selection=json.loads((OUT/'selection.json').read_text());orig=selection['original']
    # An index is constructed only for fixed-map features with reliable 3D tracks.
    indices={};point_ids={}
    for name,im in image_by_name.items():
        pid=im.point3D_ids
        valid=np.array([p>=0 and p in points and len(points[p].image_ids)>=4 and points[p].error<=1 for p in pid])
        assert len(pid)==len(desc[name])
        point_ids[name]=pid[valid]
        index=cv2.FlannBasedMatcher(dict(algorithm=1,trees=4),dict(checks=64));index.add([desc[name][valid]]);index.train();indices[name]=index
    rows=[];failures=[];quality=[]
    for source in selection['query']:
        name=f'query_{source:06d}.jpg';nearest=sorted(orig,key=lambda n:abs(orig[n]-source))[:8]
        votes={}
        for ref in nearest:
            for pair in indices[ref].knnMatch(desc[name],k=2):
                if len(pair)<2:continue
                a,b=pair
                if a.distance>=.7*b.distance:continue
                pid=int(point_ids[ref][a.trainIdx]);key=(a.queryIdx,pid)
                if key not in votes:votes[key]=[0,a.distance]
                votes[key][0]+=1;votes[key][1]=min(votes[key][1],a.distance)
        # One-to-one correspondences: require agreement from at least two model views.
        chosen=[];used_q=set();used_p=set()
        for (q,pid),(count,distance) in sorted(votes.items(),key=lambda kv:(-kv[1][0],kv[1][1])):
            if count<2 or q in used_q or pid in used_p:continue
            chosen.append((q,pid));used_q.add(q);used_p.add(pid)
        if len(chosen)<50:failures.append(dict(name=name,reason='too_few_multiview_matches',matches=len(chosen)));continue
        obj=np.asarray([points[p].xyz for _,p in chosen],np.float64)
        pix=np.asarray([xy[name][q] for q,_ in chosen],np.float64)
        ok,rvec,tvec,inlier=cv2.solvePnPRansac(obj,pix,K,distortion,iterationsCount=1000,reprojectionError=3.,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
        if not ok or inlier is None or len(inlier)<50:
            failures.append(dict(name=name,reason='pnp_inliers',matches=len(chosen)));continue
        ids=inlier[:,0];rvec,tvec=cv2.solvePnPRefineLM(obj[ids],pix[ids],K,distortion,rvec,tvec)
        projected,_=cv2.projectPoints(obj[ids],rvec,tvec,K,distortion);error=np.linalg.norm(projected[:,0]-pix[ids],axis=-1)
        R=cv2.Rodrigues(rvec)[0];C=(-R.T@tvec)[:,0]
        if np.median(error)>1.5:failures.append(dict(name=name,reason='reprojection_median',median=float(np.median(error))));continue
        raw=cv2.imread(str(OUT/'raw_images'/name));undist=cv2.remap(raw,mapx,mapy,cv2.INTER_LINEAR)
        target=images/(Path(name).stem+'.png');assert cv2.imwrite(str(target),undist)
        q=dict(name=name,matches=len(chosen),inliers=len(ids),median_reprojection_px=float(np.median(error)),p95_reprojection_px=float(np.percentile(error,95)),
            query_sha256=hashlib.sha256((OUT/'raw_images'/name).read_bytes()).hexdigest())
        quality.append(q)
        rows.append(dict(name=target.name,source_index=source,width=int(u.width),height=int(u.height),fx=float(fx),fy=float(fy),cx=float(ux),cy=float(uy),
            position=C.tolist(),rotation=R.T.tolist(),localization=q))
        print(json.dumps(q),flush=True)
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[p.name] for p in fixed_paths)
    (OUT/'cameras.json').write_text(json.dumps(rows,indent=2));(output/'quality.json').write_text(json.dumps(dict(rows=quality,failures=failures),indent=2))
    state.update(status='localized_not_evaluated',finished=time.time(),registered_queries=len(rows),failed_queries=failures,
        images_path=str(images),frozen_reconstruction_verified=True,source_model_hashes=hashes,
        evaluation_policy='Freeze model hashes before rendering; no training on these images; correlated same-video time interpolation only',
        data_used_for_appearance_training=False,reference_geometry_updated=False)
    save()
    if len(rows)<18:raise RuntimeError('Fewer than18/24 query frames localized; do not silently select only easy cases')

if __name__=='__main__':main()
