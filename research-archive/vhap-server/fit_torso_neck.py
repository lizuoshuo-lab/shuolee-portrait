"""Fit one FLAME neck-base attachment in the measured upper-torso frame."""

from argparse import ArgumentParser
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import torch
import tyro
from vhap.config.base import BaseTrackingConfig
from vhap.model.tracker import GlobalTracker

from fit_guided_vhap import camera_gauge
from fit_native_skin_profile import camera, mesh_intersections, project
from fit_vhap_neck_sequence import contour_edges
from prepare_video_avatar import setup_vhap_cameras
from run_video import input_hash, write_json


class VideoPixels:
    """Map registered pixel centers back to the decoded source video, including the calibrated lens."""
    def __init__(self,root,views,names):
        sys.path.insert(0,str(Path.home()/"gaussian-splatting/scene"))
        from colmap_loader import read_extrinsics_binary, read_intrinsics_binary
        scene=root/"scene"
        source=json.loads((scene/"source-frames.json").read_bytes())
        self.index={row["name"]:row["source_index"] for row in source["frames"]}
        self.video=cv2.VideoCapture(source["video"])
        self.video.set(cv2.CAP_PROP_ORIENTATION_AUTO,1)
        ok,frame=self.video.read()
        if not ok:
            raise IOError("Cannot decode original source pixels")
        self.size=np.array(frame.shape[1::-1])
        und=scene/"orig_undist/sparse"
        rect={im.name:im for im in read_extrinsics_binary(und/"images.bin").values()}
        models=sorted((scene/"colmap_bg/sparse").glob("*/images.bin"))
        for path in models:
            raw={im.name:im for im in read_extrinsics_binary(path).values()}
            if set(raw)==set(names) and all(np.allclose(raw[n].tvec,rect[n].tvec,atol=1e-7) for n in names):
                break
        else:
            raise ValueError("No matching pre-undistortion camera model")
        cams=read_intrinsics_binary(path.parent/"cameras.bin")
        cam=cams[raw[names[0]].camera_id]
        if cam.model!="OPENCV" or len(cams)!=1:
            raise ValueError("Expected the pipeline's OPENCV source calibration")
        fx,fy,cx,cy,*d=cam.params
        self.k=np.array([[fx,0,cx-.5],[0,fy,cy-.5],[0,0,1.]])
        self.dist=np.asarray(d)
        self.scale=self.size/[cam.width,cam.height]
        self.rect_k=views[0][0]
        self.roundtrip_px=0.
        # Independent correspondences between the distorted and rectified COLMAP exports.
        for name in names:
            ri,ui=raw[name],rect[name]
            # A 3D track can have more than one keypoint in one image. The undistorter
            # preserves point2D indices; verify the order before pairing observations.
            if not np.array_equal(ri.point3D_ids,ui.point3D_ids):
                raise ValueError(f"COLMAP point2D indices differ in {name}")
            measured=ri.point3D_ids>=0
            if not measured.any():
                raise ValueError(f"No registered source observations in {name}")
            u,v=ui.xys[measured],ri.xys[measured]
            e=np.linalg.norm(self.map(u-.5)-(v*self.scale-.5),axis=1)
            self.roundtrip_px=max(self.roundtrip_px,float(e.max()))
        if self.roundtrip_px>.05:
            raise ValueError(f"Original video mapping disagrees with native COLMAP points by {self.roundtrip_px:.3f} px")

    def map(self,xy):
        xy=np.asarray(xy,dtype=float).reshape(-1,2)
        if not len(xy):
            return xy
        rays=np.column_stack([xy,np.ones(len(xy))])@np.linalg.inv(self.rect_k).T
        uv=cv2.projectPoints(rays,np.zeros(3),np.zeros(3),self.k,self.dist)[0][:,0]
        return (uv+.5)*self.scale-.5

    def frame(self,name):
        index=self.index[name]
        self.video.set(cv2.CAP_PROP_POS_FRAMES,index)
        ok,image=self.video.read()
        if not ok or int(self.video.get(cv2.CAP_PROP_POS_FRAMES))!=index+1:
            raise IOError(name)
        return image


def contour_x(mesh, view, obs, edges, eligible):
    xy,z=project(mesh,*view)
    out=np.full(len(obs),np.nan)
    for label,side in (("left",-1),("right",1)):
        ids=[j for j,q in enumerate(obs) if q["side"]==label]
        if ids:
            ys=np.asarray([obs[j]["native_xy"][1] for j in ids])
            out[ids]=mesh_intersections(xy,z,mesh,edges,eligible,ys,side)[0]
    return out


def summarize(values):
    arr=np.asarray(values,dtype=float)
    finite=arr[np.isfinite(arr)]
    return {"observations":len(arr),"missing":int(len(arr)-len(finite)),"over3":int((finite>3).sum()),
            "median_px":float(np.median(finite)) if len(finite) else None,
            "max_px":float(np.max(finite)) if len(finite) else None}


def score(mesh,ref,views,names,records,obs,held,edges,eligible,video):
    by_frame, pooled, measured = {},{},{}
    for i,name in enumerate(names):
        if name not in held:
            continue
        yaw=float(records[i]["baseline_yaw_deg"])
        sector=("negative" if yaw<0 else "positive") if abs(yaw)>=55 else "front_or_oblique"
        row={"yaw_deg":yaw,"sector":sector}
        for mode,j in (("fixed",ref),("own",i)):
            predicted=contour_x(mesh[j],views[i],obs[i],edges,eligible)
            targets=np.array([q["native_xy"] for q in obs[i]]).reshape(-1,2)
            targets_raw=video.map(targets)
            pred_raw=video.map(np.column_stack([predicted,targets[:,1]]))
            err=np.linalg.norm(pred_raw-targets_raw,axis=1)
            row[mode]=summarize(err)
            row[mode+"_registered"]=summarize(np.abs(predicted-targets[:,0]))
            pooled.setdefault((sector,mode),[]).extend(err.tolist())
            if mode=="fixed":
                measured[name]=[{"xy":xy.tolist(),"region":q["region"],
                                 "pred_xy":p.tolist() if np.isfinite(p).all() else None}
                                for q,xy,p in zip(obs[i],targets_raw,pred_raw)]
        by_frame[name]=row
    sector_scores={sector:{mode:summarize(pooled.get((sector,mode),[])) for mode in ("fixed","own")}
                   for sector in ("negative","positive","front_or_oblique")}
    return {"by_view":by_frame,"by_sector":sector_scores,"fixed_witnesses":measured}


def review_grid(out,records,views,names,reference,baseline,model,faces,scores,base_scores,video):
    folder=out/"grid"
    folder.mkdir()
    saved={}
    for i,(name,row) in enumerate(zip(names,records)):
        if name not in scores["by_view"] and i!=reference:
            continue
        raw=video.frame(name)
        clouds=[video.map(project(surface,*views[i])[0]) for surface in (baseline[reference],model[reference],model[i])]
        allxy=np.vstack(clouds)
        lo=np.maximum(np.floor(allxy.min(0)).astype(int)-18,0)
        hi=np.minimum(np.ceil(allxy.max(0)).astype(int)+19,raw.shape[1::-1])
        image=raw[lo[1]:hi[1],lo[0]:hi[0]]
        tiles=[]
        for col,title in enumerate(("Source "+Path(name).stem,"VHAP "+Path(names[reference]).stem,
                                    "Torso "+Path(names[reference]).stem,"Torso own pose")):
            tile=image.copy()
            if col:
                alpha=np.zeros(tile.shape[:2],np.uint8)
                local=np.rint(clouds[col-1]-lo).astype(np.int32)
                for tri in local[faces]:
                    cv2.fillConvexPoly(alpha,tri,255)
                lines=cv2.findContours(alpha,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
                cv2.drawContours(tile,lines,-1,(0,220,255),1,cv2.LINE_AA)
            if col in (1,2):
                table=base_scores if col==1 else scores
                for q in table["fixed_witnesses"].get(name,[]):
                    tx,ty=np.rint(np.asarray(q["xy"])-lo).astype(int)
                    cv2.circle(tile,(tx,ty),2,(0,255,0),-1)
                    if q["pred_xy"] is not None:
                        pp=tuple(np.rint(np.asarray(q["pred_xy"])-lo).astype(int))
                        cv2.line(tile,(tx,ty),pp,(220,85,240),1)
                result=table["by_view"].get(name,{}).get("fixed",{})
                value=result.get("max_px")
                subtitle=("max %.2f px"%value if value is not None else "no witness")+(" *" if result.get("missing") else "")
            else:
                subtitle=("yaw %+.1f deg"%row["baseline_yaw_deg"]) if col==0 else "own camera / pose"
            tile=cv2.copyMakeBorder(tile,46,0,0,0,cv2.BORDER_CONSTANT,value=(32,32,32))
            cv2.putText(tile,title,(3,17),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(tile,subtitle,(3,36),cv2.FONT_HERSHEY_SIMPLEX,.37,(220,220,220),1,cv2.LINE_AA)
            tiles.append(tile)
        combined=np.hstack(tiles)
        if not cv2.imwrite(str(folder/(Path(name).stem+".png")),combined):
            raise IOError(name)
        saved[name]=combined
    # Show all measured side cameras, with the same pose. Keep the named witness-poor cameras visible.
    selected=[n for n in names if n in scores["by_view"] and scores["by_view"][n]["sector"]!="front_or_oblique"]
    if not selected:
        raise ValueError("No registered side views in the frozen evaluation split")
    width=max(saved[n].shape[1] for n in selected)
    grid=np.vstack([cv2.copyMakeBorder(saved[n],0,0,0,width-saved[n].shape[1],cv2.BORDER_CONSTANT) for n in selected])
    if not cv2.imwrite(str(folder/"both-sides.png"),grid):
        raise IOError("Both-side grid write failed")
    pair=[]
    for side in ("negative","positive"):
        group=[n for n in selected if scores["by_view"][n]["sector"]==side]
        if group:
            pair.append(max(group,key=lambda n:(scores["by_view"][n]["fixed"]["observations"],
                            -abs(abs(scores["by_view"][n]["yaw_deg"])-90))))
    if len(pair)==2:
        detail=np.vstack([cv2.copyMakeBorder(saved[n],0,0,0,width-saved[n].shape[1],cv2.BORDER_CONSTANT) for n in pair])
        if not cv2.imwrite(str(folder/"two-side-detail.png"),detail):
            raise IOError("Two-camera grid write failed")
    return selected


@torch.no_grad()
def run(root,out,mask_path):
    out.mkdir(parents=True,exist_ok=False)
    init=root/"head-neck/initialization.json"
    settings=json.loads(init.read_bytes())
    records=settings["frames"]
    names=[row["name"] for row in records]
    ref=names.index(json.loads((root/"split.json").read_bytes())["reference"])
    fit=json.loads((root/"neck-pitch/neck-joint-fit.json").read_bytes())
    held=set(fit["held_frames"])
    train=np.array([n not in held for n in names])
    tape=np.load(root/"registered-head/registered-head.npz")
    torso=np.load(root/"torso-texture/poses.npz")
    src=Path(json.loads((root/"rear-rgb/result.json").read_bytes())["checkpoint"]).resolve(strict=True)
    src.relative_to(root.resolve())
    if (torso["names"].tolist()!=names or tape["filenames"].tolist()!=names
            or torso["reference"].item()!=names[ref] or tape["initialization_sha256"].item()!=input_hash(init)
            or torso["initialization_sha256"].item()!=input_hash(init) or tape["checkpoint_sha256"].item()!=input_hash(src)):
        raise ValueError("Fresh source, camera or frame order mismatch")
    source_fit=[[q for q in fit["frame_results"][n]["contours"] if q["split"]=="fit" and q["after_x"] is not None]
                if train[i] else [] for i,n in enumerate(names)]
    check=[[q for q in fit["frame_results"][n]["contours"] if q["split"]=="check"] for n in names]
    active=[i for i,f in enumerate(source_fit) if f]
    if not active:
        raise ValueError("No fit contours on the nonheld cameras")
    write_json(out/"plan.json",{"reference":names[ref],"held":sorted(held),
               "fit_rows":sum(map(len,source_fit)),"fit_frames":[names[i] for i in active],
               "position_orientation":"one metric position and SO3 orientation in the tracked torso, identical in all frames",
               "fit":"native same-time nonheld skin contour residuals + weak face projection preservation",
               "acceptance":"fixed reference geometry unchanged across held cameras, visible native row errors <=3 px on both signed sides; missing rows and cameras reported"})
    scale=float(tape["scene_scale"])
    origin=tape["scene_origin"]
    world_rot=tape["tracker_to_world_rotation"]
    world_axes=torso["world_from_torso"]
    body_axes=world_rot.T@world_axes
    body_origin=((torso["origin_world"]-origin)/scale)@world_rot
    views=[camera(row) for row in records]
    setup_vhap_cameras(camera_gauge(records,scale)[2])
    seq=root/"head-neck"
    cfg=tyro.cli(BaseTrackingConfig,args=["--data.root-folder",str(seq.parent),"--data.sequence",seq.name,
                 "--data.calibrated","--data.use-alpha-map","--exp.output-folder",str(out/"native"),"--no-async-func"])
    tracker=GlobalTracker(cfg)
    tracker.load_from_tracked_flame_params(src)
    ids=torch.arange(len(names),device=tracker.device)
    def forward():
        batches=[tracker.forward_flame(b)[:3] for b in ids.split(cfg.batch_size)]
        return [torch.cat(v).cpu().numpy() for v in zip(*batches)]
    base,canonical,lm=forward()
    np.testing.assert_allclose(base,tape["vertices_tracker"],atol=1e-6,rtol=0)
    def to_world(values):
        return values@(world_rot.T*scale)+origin
    faces=tape["faces"]
    edges,eligible,_,_=contour_edges(faces,mask_path)
    fitting_edges=edges[eligible]
    fitting_eligible=np.ones(len(fitting_edges),dtype=bool)
    j0=np.einsum("v,nvc->nc",tracker.flame.J_regressor[0].cpu().numpy(),canonical)
    old_rot=Rotation.from_rotvec(tracker.rotation.cpu().numpy())
    old_neck=Rotation.from_rotvec(tracker.neck_pose.cpu().numpy())
    composite=old_rot*old_neck
    body=Rotation.from_matrix(body_axes)
    old_pivot=j0+tracker.translation.cpu().numpy()
    local_pos=body.inv().apply(old_pivot-body_origin)
    pos0=np.median(local_pos[train],axis=0)
    rot0=(body.inv()*old_rot)[train].mean()
    def set_pose(x):
        local=rot0*Rotation.from_rotvec(np.deg2rad(x[:3]))
        top=body*local
        pos=pos0+.001*x[3:]
        pivot=body_origin+body.apply(np.tile(pos,(len(names),1)))
        for name,value in (("rotation",top.as_rotvec()),("translation",pivot-j0),("neck_pose",(top.inv()*composite).as_rotvec())):
            dest=getattr(tracker,name)
            dest.copy_(torch.as_tensor(value).to(dest))
        return local,pos
    protect=[27,30,36,39,42,45,48,54]
    face_base=to_world(lm[:,protect])
    target=[np.array([p["native_xy"][0] for p in rows]) for rows in source_fit]
    def residual(x):
        set_pose(x)
        mesh,_,landmarks=forward()
        world,landmarks=to_world(mesh),to_world(landmarks[:,protect])
        errors=[]
        for i in active:
            # A measured skin boundary supervises the neck. The check uses the complete
            # mesh and marks a contour unavailable when hair, jaw or the open rim hides it.
            pred=contour_x(world[i],views[i],source_fit[i],fitting_edges,fitting_eligible)
            d=pred-target[i]
            d[~np.isfinite(d)]=24.
            face_delta=project(landmarks[i],*views[i])[0]-project(face_base[i],*views[i])[0]
            errors.extend(d/np.sqrt(len(d)))
            errors.extend(.55*face_delta.ravel()/np.sqrt(len(protect)))
        return np.r_[errors,x[:3]/12.,x[3:]/40.]
    def jacobian(x):
        steps=np.array([.08,.08,.08,.25,.25,.25])
        return np.column_stack([(residual(x+e)-residual(x-e))/(2*h) for e,h in zip(np.diag(steps),steps)])
    solution=least_squares(residual,np.zeros(6),jac=jacobian,loss="huber",f_scale=1.5,
                           x_scale=[5,5,5,15,15,15],max_nfev=25,ftol=1e-5,xtol=1e-4)
    local,pos=set_pose(solution.x)
    best,_,new_lm=forward()
    # Verify the attachment on actual FLAME parameters, not merely its requested target.
    read_rot=Rotation.from_rotvec(tracker.rotation.cpu().numpy())
    actual=(body.inv()*read_rot)
    np.testing.assert_allclose((local.inv()*actual).magnitude(),0,atol=2e-6)
    actual_pos=body.inv().apply(j0+tracker.translation.cpu().numpy()-body_origin)
    np.testing.assert_allclose(actual_pos,np.broadcast_to(pos,actual_pos.shape),atol=2e-6)
    face_shift=np.concatenate([np.linalg.norm(project(a,*v)[0]-project(b,*v)[0],axis=1)
                               for a,b,v in zip(to_world(lm[:,17:68]),to_world(new_lm[:,17:68]),views)])
    baseline=to_world(base)
    candidate=to_world(best)
    video=VideoPixels(root,views,names)
    before=score(baseline,ref,views,names,records,check,held,edges,eligible,video)
    after=score(candidate,ref,views,names,records,check,held,edges,eligible,video)
    review=review_grid(out,records,views,names,ref,baseline,candidate,faces,after,before,video)
    video.video.release()
    tracker.timestep=int(np.load(src)["n_processed_frames"])
    tracker.evaluate(make_visualization=True,epoch=0)
    tracker.tb_writer.close()
    saved=out/"tracked_flame_params.npz"
    shutil.copyfile(tracker.out_dir/"tracked_flame_params_0.npz",saved)
    grid=tracker.out_dir/"eval_0/image_grid"
    if {p.name for p in grid.glob("*.jpg")}!={f"frame_{i:05d}.jpg" for i in range(len(names))}:
        raise RuntimeError("Incomplete native VHAP grid")
    with np.load(src) as old,np.load(saved) as new:
        allowed={"rotation","translation","neck_pose"}
        if set(new.files)!=set(old.files) or any(not np.array_equal(old[k],new[k]) for k in old.files if k not in allowed):
            raise ValueError("Unexpected parameter changed during torso fitting")
    tracker.load_from_tracked_flame_params(saved)
    np.testing.assert_allclose(forward()[0],best,rtol=0,atol=1e-6)
    side_scores=[after["by_sector"][k]["fixed"] for k in ("negative","positive")]
    observed_pass=all(v["observations"] and v["missing"]==0 and v["over3"]==0 for v in side_scores)
    empty_views=[n for n in review if after["by_view"][n]["fixed"]["observations"]==0]
    write_json(out/"result.json",{"reference":names[ref],"native_grid":str(grid),"grid_views":review,
        "baseline":before,"candidate":after,"complete":True,"all_side_witnesses_within_3px":bool(observed_pass),
        "side_views_with_no_witness":empty_views,"all_side_cameras_verified":bool(observed_pass and not empty_views),
        "source_video_size":video.size.tolist(),"source_mapping_check_max_px":video.roundtrip_px,
        "units":"Neck max_px uses original oriented video pixels; per-camera *_registered is the undistorted raster.",
        "frames":len(names),"face_projection_shift_registered_px":summarize(face_shift),"optimizer_status":int(solution.status),
        "evaluations":int(solution.nfev),"loss":float(solution.cost),"shared_root_position_m":pos.tolist(),
        "torso_from_neck_root_rotation":local.as_matrix().tolist(),"checkpoint_sha256":input_hash(saved),
        "baseline_sha256":input_hash(src),"fresh_run":str(root),"body_track_sha256":input_hash(root/"torso-texture/poses.npz")})
    print("Fixed pose held sides",side_scores,"unmeasured side cameras",empty_views,flush=True)


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    for arg in ("run","output","flame-masks"):
        parser.add_argument("--"+arg,type=Path,required=True)
    cv2.setNumThreads(1)
    a=parser.parse_args()
    run(a.run,a.output,a.flame_masks)
