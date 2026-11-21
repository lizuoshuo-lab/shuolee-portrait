"""Recover upper-torso motion from native garment texture in a fresh registered video."""

from argparse import ArgumentParser
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import torch

from fit_native_skin_profile import camera, project
from run_video import input_hash, write_json


def shirt_region(mask, joints, box):
    shoulder = joints[[5, 6]].mean(axis=0)
    hips = joints[[11, 12]].mean(axis=0)
    spine = hips - shoulder
    lateral = np.array([spine[1], -spine[0]]) / max(np.linalg.norm(spine), 1.)
    half = max(.42*np.linalg.norm(joints[5]-joints[6]), .22*(box[2]-box[0]))
    top, bottom = shoulder+.08*spine, shoulder+.76*spine
    corners = [top-half*lateral, top+half*lateral, bottom+.87*half*lateral, bottom-.87*half*lateral]
    region = np.zeros_like(mask)
    cv2.fillConvexPoly(region, np.rint(corners).astype(np.int32), 255)
    interior = cv2.erode((mask > 235).astype(np.uint8), np.ones((9,9), np.uint8))
    region *= interior
    for elbow, wrist in ((7,9),(8,10)):
        cv2.line(region, tuple(np.rint(joints[elbow]).astype(int)), tuple(np.rint(joints[wrist]).astype(int)),
                 0, max(5, int(.27*half)))
    return region


def visible(region, xy):
    ij = np.rint(xy).astype(int)
    ok = np.isfinite(xy).all(axis=1) & (ij >= 0).all(axis=1) & (ij < region.shape[::-1]).all(axis=1)
    ok[ok] &= region[ij[ok,1],ij[ok,0]] != 0
    return ok


def track_scene(root, frames, body):
    source = root/"scene"
    history, active = [], []
    xy = np.empty((0,2), np.float32)
    previous = None
    for i, row in enumerate(frames):
        name = row["name"]
        gray = cv2.imread(str(source/"orig_undist/images"/name),cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(source/"mask_undist/images"/name),cv2.IMREAD_GRAYSCALE)
        if gray is None or mask is None or gray.shape != mask.shape:
            raise ValueError(f"Missing source frame or mask for {name}")
        region = shirt_region(mask,body["keypoints"][i],body["boxes"][i])
        if len(xy):
            flow = dict(winSize=(19,19),maxLevel=3,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,24,.01))
            new, s1, _ = cv2.calcOpticalFlowPyrLK(previous,gray,xy,None,**flow)
            back, s2, _ = cv2.calcOpticalFlowPyrLK(gray,previous,new,None,**flow)
            keep = ((s1.ravel()>0) & (s2.ravel()>0) & visible(region,new)
                    & (np.linalg.norm(back-xy,axis=1) <= .8))
            # Check image appearance independently of the flow round trip.
            for j in np.flatnonzero(keep):
                a = cv2.getRectSubPix(previous,(11,11),tuple(xy[j])).astype(float)
                b = cv2.getRectSubPix(gray,(11,11),tuple(new[j])).astype(float)
                a -= a.mean()
                b -= b.mean()
                keep[j] = np.vdot(a,b) >= .77*max(np.linalg.norm(a)*np.linalg.norm(b),1.)
            active = [active[j] for j in np.flatnonzero(keep)]
            xy = new[keep]
            for track,point in zip(active,xy):
                history[track].append((i,*point))
        # Refill spatial tiles so a single logo cannot consume the full budget.
        allow = region.copy()
        for pt in np.rint(xy).astype(int):
            cv2.circle(allow,tuple(pt),7,0,-1)
        xx,yy,ww,hh = cv2.boundingRect(region)
        fresh = []
        for a in range(3):
            for b in range(4):
                x0,x1 = xx+ww*a//3,xx+ww*(a+1)//3
                y0,y1 = yy+hh*b//4,yy+hh*(b+1)//4
                if min(x1-x0,y1-y0)<8:
                    continue
                n = np.count_nonzero((xy[:,0]>=x0)&(xy[:,0]<x1)&(xy[:,1]>=y0)&(xy[:,1]<y1))
                if n>=10:
                    continue
                found = cv2.goodFeaturesToTrack(gray[y0:y1,x0:x1],10-n,.015,8,
                                                mask=allow[y0:y1,x0:x1],blockSize=5)
                if found is not None:
                    fresh.extend(found[:,0]+[x0,y0])
        if fresh:
            fresh = np.asarray(fresh,np.float32)
            for p in fresh:
                active.append(len(history))
                history.append([(i,*p)])
            xy = np.concatenate([xy,fresh])
        previous = gray
    return [np.asarray(rows) for rows in history if len(rows)>=5]


def rotations(v):
    x,y,z = v.unbind(-1)
    zero = torch.zeros_like(x)
    sk = torch.stack([zero,-z,y,z,zero,-x,-y,x,zero],-1).reshape(-1,3,3)
    angle = torch.linalg.vector_norm(v,dim=-1)[:,None,None]
    return (torch.eye(3,device=v.device)[None]+torch.sinc(angle/torch.pi)*sk
            +.5*torch.sinc(angle/(2*torch.pi))**2*(sk@sk))


def triangulate(rays, origins, depth_seed=None):
    normal = np.eye(3)-rays[:,:,None]*rays[:,None,:]
    lhs, rhs = normal.sum(0), (normal@origins[:,:,None]).sum(0)[:,0]
    if depth_seed is not None:
        ridge = .015*len(rays)
        lhs += ridge*np.eye(3)
        rhs += ridge*np.median(depth_seed,axis=0)
    return np.linalg.lstsq(lhs,rhs,rcond=1e-10)[0]


def validate_tracks(tracks, frames, xy, rays, cams, center, scale, motion, shifts):
    results = {}
    for label,rot,tr in (("static",np.tile(np.eye(3),(len(cams),1,1)),np.zeros((len(cams),3))),
                         ("tracked",motion,shifts)):
        errors = []
        for g in np.unique(tracks[tracks%5==0]):
            ii = np.flatnonzero(tracks==g)
            train,test = ii[::2],ii[1::2]
            f = frames[train]
            local_rays = (rays[train,None,:]@rot[f])[:,0]
            origins = np.stack([(cams[j][2]-center)/scale for j in f])
            local_origins = ((origins-tr[f])[:,None,:]@rot[f])[:,0]
            p = triangulate(local_rays,local_origins)
            for j in test:
                i = frames[j]
                world = center+scale*(rot[i]@p+tr[i])
                pred,z = project(world[None],*cams[i])
                errors.append(np.linalg.norm(xy[j]-pred[0]) if z[0]>0 else np.inf)
        e=np.asarray(errors)
        finite=e[np.isfinite(e)]
        results[label]={"samples":len(e),"nonfinite_or_behind":int(len(e)-len(finite)),
                        "over3":int((e>3).sum()),"median":float(np.median(finite)),"p95":float(np.percentile(finite,95))}
    return results


def run(root, out):
    out.mkdir(parents=True,exist_ok=False)
    init_path = root/"head-neck/initialization.json"
    setup=json.loads(init_path.read_bytes())
    rows,scale=setup["frames"],float(setup["scene_scale"])
    names=[r["name"] for r in rows]
    ref=names.index(json.loads((root/"split.json").read_bytes())["reference"])
    cams=[camera(r) for r in rows]
    anatomy=np.load(root/"anatomical-prior/smplx-frames.npz")
    body=np.load(root/"body-observations.npz")
    if not (names==body["names"].tolist()==anatomy["names"].tolist()):
        raise ValueError("Native image, camera and body orders disagree")
    tracked=track_scene(root,rows,body)
    if len(tracked)<12:
        raise RuntimeError(f"Only {len(tracked)} material tracks of at least five frames")
    fxy=np.concatenate(tracked)
    ids=np.repeat(np.arange(len(tracked)),[len(r) for r in tracked])
    fi,xy=fxy[:,0].astype(int),fxy[:,1:]
    torso=anatomy["joints_world"][ref]
    shoulder=torso[[16,17]].mean(0)
    hip=torso[[1,2]].mean(0)
    center=.65*shoulder+.35*hip
    right=torso[17]-torso[16]
    right/=np.linalg.norm(right)
    up=shoulder-hip
    up-=right*np.dot(up,right)
    up/=np.linalg.norm(up)
    axes=np.column_stack([right,up,np.cross(right,up)])
    origins=np.array([(c-center)/scale for _,_,c in cams])
    rays,depth=np.zeros((len(xy),3)),np.zeros(len(xy))
    meshes=anatomy["vertices_world"]
    for i,(k,r,c) in enumerate(cams):
        ii=fi==i
        coords,z=project(meshes[i],k,r,c)
        _,nearest=cKDTree(coords).query(xy[ii],k=16)
        depth[ii]=np.quantile(z[nearest],.2,axis=1)/scale
        direction=np.column_stack([xy[ii],np.ones(ii.sum())])@np.linalg.inv(k).T@r.T
        rays[ii]=direction/np.linalg.norm(direction,axis=1,keepdims=True)
    depth_along_ray=depth/np.einsum('ij,ij->i',rays,np.stack([c[1][:,2] for c in cams])[fi])
    seed=origins[fi]+rays*depth_along_ray[:,None]
    xyz=np.vstack([triangulate(rays[ids==j],origins[fi[ids==j]],seed[ids==j]) for j in range(len(tracked))])
    fit=ids%5!=0
    support=np.bincount(fi[fit],minlength=len(rows))
    if np.count_nonzero(support>=6)<3:
        raise RuntimeError(f"Not enough views with measured garment motion: {support.tolist()}")
    overlap=[len(np.intersect1d(ids[fit & (fi==i)],ids[fit & (fi==i+1)])) for i in range(len(rows)-1)]
    device="cuda" if torch.cuda.is_available() else "cpu"
    def ten(a):
        return torch.as_tensor(a,dtype=torch.float32,device=device)
    f=torch.as_tensor(fi[fit],device=device)
    g=torch.as_tensor(ids[fit],device=device)
    target,seed_depth=ten(xy[fit]),ten(depth[fit])
    ks,cr=ten(np.stack([v[0] for v in cams])),ten(np.stack([v[1] for v in cams]))
    origins_t=ten(origins)
    points=torch.nn.Parameter(ten(xyz))
    rv=torch.nn.Parameter(ten(np.zeros((len(rows),3))))
    offset=torch.nn.Parameter(torch.zeros_like(rv))
    gauge=ten((np.arange(len(rows))!=ref)[:,None])
    params=[points,rv,offset]
    def loss():
        w,t=rv*gauge,offset*gauge
        m=rotations(w)
        pos=(m[f]@points[g,:,None])[:,:,0]+t[f]
        cam=((pos-origins_t[f])[:,None,:]@cr[f])[:,0]
        screen=(ks[f]@cam[:,:,None])[:,:,0]
        err=((screen[:,:2]/screen[:,2:]-target)**2).sum(1)
        data=(4*(torch.sqrt(1+err/4)-1)).mean()
        # Weak metric gauge from fresh anatomy; twisting or translating a monocular rig has scale ambiguity.
        regularity=(.16*((cam[:,2]-seed_depth)/.08).square().mean()
                    +.1*((t/.05).square().mean()+(w/.20).square().mean())
                    +.2*((torch.diff(t,n=2,dim=0)/.006).square().mean()
                          +(torch.diff(w,n=2,dim=0)/.025).square().mean()))
        return data+regularity
    optimizer=torch.optim.Adam(params,lr=.0015)
    for step in range(320):
        if step==220:
            optimizer.param_groups[0]["lr"]=.0005
        optimizer.zero_grad()
        value=loss()
        value.backward()
        optimizer.step()
    optimizer=torch.optim.LBFGS(params,max_iter=60,line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        value=loss()
        value.backward()
        return value
    optimizer.step(closure)
    m=rotations(rv*gauge).detach().cpu().numpy()
    t=(offset*gauge).detach().cpu().numpy()
    if not np.isfinite(m).all() or not np.isfinite(t).all():
        raise ValueError("Nonfinite garment bundle")
    hold=validate_tracks(ids,fi,xy,rays,cams,center,scale,m,t)
    panels=[]
    yaw=np.asarray([row["baseline_yaw_deg"] for row in rows])
    for i in dict.fromkeys([int(np.argmin(np.abs(yaw+75))),ref,int(np.argmin(np.abs(yaw-75)))]):
        raw=cv2.imread(str(root/"scene/orig_undist/images"/names[i]))
        mask=cv2.imread(str(root/"scene/mask_undist/images"/names[i]),cv2.IMREAD_GRAYSCALE)
        region=shirt_region(mask,body["keypoints"][i],body["boxes"][i])
        xx,yy,ww,hh=cv2.boundingRect(region)
        for j in np.flatnonzero(fi==i):
            cv2.circle(raw,tuple(np.rint(xy[j]).astype(int)),3,(0,210,255) if fit[j] else (255,210,0),1,cv2.LINE_AA)
        tile=raw[max(0,yy-28):min(raw.shape[0],yy+hh+28),max(0,xx-35):min(raw.shape[1],xx+ww+35)]
        tile=cv2.copyMakeBorder(tile,26,0,0,0,cv2.BORDER_CONSTANT)
        cv2.putText(tile,f"{names[i]}  {support[i]} fit",(4,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
        panels.append(tile)
    high=max(len(im) for im in panels)
    preview=np.hstack([cv2.copyMakeBorder(im,0,high-len(im),0,0,cv2.BORDER_CONSTANT) for im in panels])
    if not cv2.imwrite(str(out/"garment-tracks.png"),preview):
        raise IOError("Could not save garment track evidence")
    np.savez_compressed(out/"poses.npz",names=names,world_from_torso=m@axes,
                        origin_world=center+scale*t,frame_index=fi,track_id=ids,xy=xy,
                        reference=names[ref],initialization_sha256=input_hash(init_path))
    report={"tracks":len(tracked),"observations":len(xy),"fit_tracks":len(set(ids[fit])),
            "per_frame_fit_counts":support.tolist(),"reference":names[ref],
            "adjacent_shared_track_counts":overlap,
            "weak_frames":[n for n,c in zip(names,support) if c<6],
            "weak_transitions":[[names[i],names[i+1]] for i,c in enumerate(overlap) if c<3],
            "texture_validation":hold,"median_track_length":float(np.median([len(v) for v in tracked])),
            "motion_from_reference_max_deg":float(np.rad2deg(Rotation.from_matrix(m).magnitude()).max()),
            "origin_motion_from_reference_max_m":float(np.linalg.norm(t,axis=1).max()),
            "longest_track":max(map(len,tracked)),"anatomy_role":"ROI, depth-scale prior and reference axes only",
            "held_policy":"Every fifth track excluded from motion fitting; alternating observations triangulate and test these tracks.",
            "loss":float(loss().detach()),"initialization_sha256":input_hash(init_path)}
    write_json(out/"tracks.json",report)
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    cv2.setNumThreads(1)
    args=parser.parse_args()
    run(args.run,args.output)
