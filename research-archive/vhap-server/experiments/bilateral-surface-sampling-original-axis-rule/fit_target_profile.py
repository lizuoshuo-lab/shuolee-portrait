"""Refine one frontal FLAME pose from real profile views after the monocular neck fit."""

from argparse import ArgumentParser, Namespace
from pathlib import Path
from shutil import copyfile

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import torch
import tyro
from vhap.config.base import BaseTrackingConfig
from vhap.model.tracker import GlobalTracker

from fit_guided_vhap import camera_gauge
from fit_native_skin_profile import camera, project
from fit_torso_neck import VideoPixels, contour_x
from fit_vhap_neck_sequence import contour_edges, load_observations
from neutral_neck_seed import inclination, torso_axes
from prepare_video_avatar import setup_vhap_cameras
from review_native_own_views import read, stats, title, witnesses
from run_video import input_hash, write_json
from track_garment_frame import run as track_torso


def select_profiles(yaw, rows):
    profiles = np.flatnonzero((abs(yaw) >= 80) & (abs(yaw) <= 100)).tolist()
    chosen = []
    for sign in (1, -1):
        usable = [i for i in profiles if np.sign(yaw[i]) == sign and
                  len({q["native_xy"][1] for q in rows[i] if q["split"] == "fit"}) >= 3]
        if usable:
            chosen.append(min(usable, key=lambda i: (abs(abs(yaw[i])-90), i)))
    return chosen, profiles


def draw_row(video, name, label, view, meshes, faces, rows):
    raw = video.frame(name)
    clouds = [video.map(project(mesh, *view)[0]) for mesh in meshes]
    pts = np.vstack(clouds + [np.asarray([q["source"] for q in group]).reshape(-1,2) for group in rows])
    lo = np.maximum(np.floor(pts.min(0)).astype(int)-17, 0)
    hi = np.minimum(np.ceil(pts.max(0)).astype(int)+18, video.size)
    crop = raw[lo[1]:hi[1],lo[0]:hi[0]]
    tiles = [title(crop, [name, label])]
    for xy, group, desc in zip(clouds, rows, ("Own 12-deg baseline", "Fixed target before", "Fixed target after"), strict=True):
        tile = crop.copy()
        mask = np.zeros(crop.shape[:2], np.uint8)
        tris = np.rint((xy-lo)*16).astype(np.int32)[faces]
        for tri in tris:
            cv2.fillConvexPoly(mask, tri, 255, shift=4)
        loops = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cv2.drawContours(tile, loops, -1, (210,230,105), 1, cv2.LINE_AA)
        for q in group:
            a = tuple(np.rint(np.asarray(q["source"])-lo).astype(int))
            if q["mesh"] is None:
                cv2.circle(tile, a, 3, (0,175,255), 1)
                continue
            b = tuple(np.rint(np.asarray(q["mesh"])-lo).astype(int))
            c = (25,45,250) if q["error_px"] > 3 else (60,220,70)
            cv2.line(tile, a, b, c, 1, cv2.LINE_AA)
            cv2.circle(tile, a, 2, c, -1)
            cv2.drawMarker(tile, b, c, cv2.MARKER_CROSS, 5, 1)
        m = stats(group)
        line = (f"{m['over3']}/{m['measured']} >3, max {m['max_px']:.1f}px" if m["measured"] else "No measured pairs")
        if m["missing"]:
            line += f", {m['missing']} missing"
        tiles.append(title(tile,[desc,line]))
    row=np.hstack(tiles)
    legend=np.full((28,row.shape[1],3),(30,29,27),np.uint8)
    cv2.putText(legend,"Dot: observed   +: mesh    Red > 3 px    Green <= 3 px    Amber: unmatched",
                (5,19),0,.43,(235,235,235),1,cv2.LINE_AA)
    return np.vstack((row,legend))


@torch.no_grad()
def run(root, profile_space, both_sides=False):
    if both_sides and profile_space != "fixed":
        raise ValueError("The bilateral target uses one world-space pose in every camera")
    out = root/("target-profile-bilateral" if both_sides else "target-profile-fixed" if profile_space=="fixed" else "target-profile")
    out.mkdir(exist_ok=False)
    run_state = read(root/"run.json")
    cfg_run = run_state["configuration"]
    if (run_state["stages"].get("neck-pitch",{}).get("status") != "complete" or not cfg_run["monocular_neck_only"]
        or cfg_run["initial_neck_posture"] != "template-neutral" or cfg_run["previous_run_inputs"]):
        raise ValueError("Expected fresh monocular 12-degree fit with a neutral neck input")
    neck = read(root/"neck-pitch/neck-joint-fit.json")
    seed = read(root/"vhap-rigid/input-neck.json")
    seq = root/"head-neck"
    init = seq/"initialization.json"
    rec = read(init)["frames"]
    names = [q["name"] for q in rec]
    ref = names.index(read(root/"split.json")["reference"])
    yaw = np.array([q["baseline_yaw_deg"] for q in rec])
    if (abs(yaw[ref]) > 30 or neck["pitch_limit_deg"] != 12 or neck["frontal_pitch_limit_deg"] != 3
        or seed["near_target_frames"] != len(names)):
        raise ValueError("Frontal target, neutral input or 12-degree baseline mismatch")
    src = Path(neck["checkpoint"]).resolve(strict=True)
    src.relative_to(root)
    tape = np.load(root/"geometry-binding/registered-head.npz")
    if (names != tape["filenames"].tolist() or str(tape["initialization_sha256"]) != input_hash(init)
        or neck["source"]["initialization_sha256"] != input_hash(init) or list(neck["frame_results"]) != names
        or str(tape["checkpoint_sha256"]) != input_hash(Path(neck["source"]["checkpoint"]))):
        raise ValueError("The 12-degree receipt, source geometry and camera sequence disagree")
    with np.load(src) as candidate, np.load(neck["previous_checkpoint"]) as prev:
        for key in set(prev.files)-{"rotation","translation","neck_pose"}:
            np.testing.assert_array_equal(candidate[key],prev[key])
        np.testing.assert_allclose(np.rad2deg(candidate["neck_pose"][:,0]),
            [neck["frame_results"][n]["flame_neck_x_deg"][1] for n in names],atol=1e-5,rtol=0)
    scale = float(tape["scene_scale"])
    rot = tape["tracker_to_world_rotation"]
    origin = tape["scene_origin"]
    def world(x):
        return x @ rot.T * scale + origin
    setup_vhap_cameras(camera_gauge(rec,scale)[2])
    cfg=tyro.cli(BaseTrackingConfig,args=["--data.root-folder",str(seq.parent),"--data.sequence",seq.name,
        "--data.calibrated","--data.use-alpha-map","--exp.output-folder",str(out/"native"),"--no-async-func"])
    tracker = GlobalTracker(cfg)
    tracker.load_from_tracked_flame_params(src)
    ids=torch.arange(len(names),device=tracker.device)
    own=world(np.concatenate([tracker.forward_flame(group)[0].cpu().numpy() for group in ids.split(cfg.batch_size)]))
    faces = tape["faces"]
    edges, eligible, _, regions = contour_edges(faces, Path("asset/flame/FLAME_masks.pkl"))
    views = [camera(r) for r in rec]
    video = VideoPixels(root, views, names)
    rows = [[] for n in names]
    for i, sign, x, y, split, region, provider in load_observations(Namespace(scene=root/"scene",sequence=seq,
            targets=root/"neck-expanded.json",native_fallback=root/"neck-narrow.json",supplement=root/"front-contours.json"),rec):
        rows[i].append(dict(side="left" if sign < 0 else "right", native_xy=(x,y), split=split, region=region, provider=provider))
    def measure(mesh,i,qs):
        xy = np.asarray([q["native_xy"] for q in qs]).reshape(-1,2)
        pred = contour_x(mesh,views[i],qs,edges,eligible)
        return witnesses(video,xy,np.column_stack((pred,xy[:,1])),[q["region"] for q in qs])
    def check(i):
        return [q for q in rows[i] if q["split"]=="check"]
    sides = {}
    if both_sides:
        from bilateral_profile_views import select_both
        train, profiles, sides = select_both(root,rec,rows)
    else:
        train, profiles = select_profiles(yaw, rows)
    test = [i for i in profiles if i not in train]
    closest = {min(group,key=lambda i:abs(abs(yaw[i])-90)) for sign in (-1,1)
               if (group:=[i for i in profiles if np.sign(yaw[i])==sign])}
    # Previously examined obliques remain diagnostic only; never use their contours in the solve.
    oblique = [min(range(len(names)), key=lambda i:abs(yaw[i]-a)) for a in (55,-67)]
    if both_sides:
        oblique = list(dict.fromkeys(oblique + [names.index(v["farthest"]) for v in sides.values() if "farthest" in v]))
    display = list(dict.fromkeys([ref, *train, *sorted(test,key=lambda i:(not bool(rows[i]), i)), *oblique]))
    front_checks = stats(measure(own[ref],ref,check(ref)))
    receipt = dict(reference=names[ref], native_steps=read(root/"vhap-rigid/result.json")["trained_steps"],
                   requested_both_sides=both_sides, side_selection=sides,
                   supervision_space=profile_space,
                   baseline_pose_status=neck["status"],
                   neutral_deg=seed["neutral_template_deg"], neutral_frames=seed["near_target_frames"],
                   pitch_limit=neck["pitch_limit_deg"], frontal_pitch_limit=neck["frontal_pitch_limit_deg"],
                   source_checkpoint=str(src), source_checkpoint_sha256=input_hash(src), script_sha256=input_hash(Path(__file__)),
                   yaw_range=[float(yaw.min()),float(yaw.max())], train=[names[i] for i in train],
                   profile_inventory={names[i]:{"yaw":float(yaw[i]),"fit_rows":len(rows[i])-len(check(i)),
                       "check_rows":len(check(i))} for i in profiles},
                   held_profiles=[names[i] for i in test], diagnostic_obliques=[names[i] for i in oblique],
                   front_baseline=front_checks, source_mapping_max_px=video.roundtrip_px)
    write_json(out/"selection.json",receipt)
    if (not train or (both_sides and len(train)!=2) or not front_checks["measured"]
            or front_checks["missing"] or front_checks["over3"]):
        write_json(out/"result.json",dict(receipt,status="blocked_at_input", reason="No qualified profile or frontal check failed"))
        video.video.release()
        tracker.tb_writer.close()
        return
    torso_dir=root/"target-profile/torso"
    if not (torso_dir/"tracks.json").is_file():
        torso_dir=out/"torso"
        with torch.enable_grad():
            track_torso(root,torso_dir)
    tracks = read(torso_dir/"tracks.json")
    garment = np.load(torso_dir/"poses.npz")
    if (garment["names"].tolist()!=names or garment["reference"].item()!=names[ref]
        or garment["initialization_sha256"].item()!=input_hash(init)):
        raise ValueError("Garment track pose mapping does not match this input")
    axes = garment["world_from_torso"]
    centers = garment["origin_world"]
    def transport(mesh,i):
        return (mesh-centers[ref])@axes[ref]@axes[i].T+centers[i]
    def supervised(mesh,i):
        return mesh if profile_space=="fixed" else transport(mesh,i)
    anatomy = np.load(root/"anatomical-prior/smplx-frames.npz")
    world_left, world_up, world_ahead = torso_axes(anatomy["joints_world"][ref])
    left = world_left @ rot
    ahead = world_ahead @ rot
    idx = torch.tensor([ref],device=tracker.device)
    verts,canonical,lm,_ = tracker.forward_flame(idx)
    np.testing.assert_allclose(world(verts[0].cpu().numpy()),own[ref],atol=2e-6,rtol=0)
    joint = (tracker.flame.J_regressor[:2]@canonical[0]).cpu().numpy()
    arm = joint[1]-joint[0]
    oldrot = Rotation.from_rotvec(tracker.rotation[ref].cpu().numpy())
    combined = oldrot*Rotation.from_rotvec(tracker.neck_pose[ref].cpu().numpy())
    t0 = tracker.translation[ref].cpu().numpy().copy()
    ref_lm = video.map(project(world(lm[0,17:68].cpu().numpy()),*views[ref])[0])
    def apply(x):
        newrot = Rotation.from_rotvec(np.deg2rad(x[0])*left)*oldrot
        tr = t0 + oldrot.apply(arm)-newrot.apply(arm)+(.001*x[1])*ahead
        for k,v in (("rotation",newrot.as_rotvec()),("translation",tr),("neck_pose",(newrot.inv()*combined).as_rotvec())):
            row=getattr(tracker,k)[ref]
            row.copy_(torch.as_tensor(v).to(row))
        vv,_,lm,_=tracker.forward_flame(idx)
        return world(vv[0].cpu().numpy()),video.map(project(world(lm[0,17:68].cpu().numpy()),*views[ref])[0])
    sample=tracker.dataset.getitem_by_timestep(ref)
    tracker.fill_cam_params_into_sample(sample)
    def photo():
        tracker.clear_cache()
        return float(tracker.compute_energy(sample)[1]["photo"])
    fitrows = {i:[q for q in rows[i] if q["split"]=="fit"] for i in [ref,*train]}
    def pixel_res(mesh,i):
        ms=measure(mesh,i,fitrows[i])
        # Use one signed distance per row. Unmatched samples incur a fixed cost and are audited separately.
        return np.asarray([24. if q["mesh"] is None else np.sign(q["mesh"][0]-q["source"][0])*q["error_px"] for q in ms])
    def residual(x,with_photo):
        mesh,lm=apply(x)
        parts=[]
        for i in train:
            d=pixel_res(supervised(mesh,i),i)
            parts.append(d/np.sqrt(len(d)*len(train)))
        f=pixel_res(mesh,ref)
        if len(f):
            parts.append(f/np.sqrt(len(f)))
        parts.extend(((lm-ref_lm).ravel()/np.sqrt(len(lm)), x/np.array([12.,25.])))
        if with_photo:
            parts.append([np.sqrt(2*photo())])
        return np.concatenate(parts)
    def jac(x,with_photo):
        steps=np.diag([.08,.25])
        return np.column_stack([(residual(x+h,with_photo)-residual(x-h,with_photo))/(2*h.max()) for h in steps])
    x=np.zeros(2)
    phases=[]
    native_photo=photo()
    def profile_data(x):
        mesh,_=apply(x)
        return np.concatenate([pixel_res(supervised(mesh,i),i) for i in train])
    eps=np.diag([.08,.25])
    j=np.column_stack([(profile_data(x+h)-profile_data(x-h))/(2*h.max()) for h in eps])
    singular=np.linalg.svd(j,compute_uv=False)
    sensitivity=dict(columns=["px per degree","px per mm"], rows=len(j), singular_values=singular.tolist(),
                     rank=int(np.linalg.matrix_rank(j)))
    if both_sides:
        offset=0
        per_view={}
        for i in train:
            n=len(fitrows[i])
            block=j[offset:offset+n]
            offset+=n
            per_view[names[i]]=dict(rows=n,baseline=stats(measure(own[ref],i,fitrows[i])),
                                    sensitivity=np.linalg.norm(block,axis=0).tolist())
        sensitivity["per_view"]=per_view
        if any(not m["baseline"]["measured"] or max(m["sensitivity"]) < 1e-5 for m in per_view.values()):
            write_json(out/"result.json",dict(receipt,status="blocked_at_input", reason="Side has no geometric response",
                                              profile_data_jacobian=sensitivity))
            video.video.release()
            tracker.tb_writer.close()
            return
    for label,rgb_on in (("profile_with_front_protection",False),("front_VHAP_RGB_with_profile",True)):
        start=x.copy()
        solution=least_squares(residual,start,jac=jac,args=(rgb_on,),bounds=([-12,-25],[12,25]),
                    x_scale=[5.,10.],loss="soft_l1",f_scale=2.,max_nfev=32)
        x=solution.x
        apply(x)
        phases.append(dict(name=label,start=start.tolist(),pitch_deg=float(x[0]),forward_mm=float(x[1]),
                    evaluations=int(solution.nfev),cost=float(solution.cost),front_photo=photo(),status=int(solution.status)))
        print(phases[-1],flush=True)
    final,lm=apply(x)
    saved=out/"target-flame.npz"
    fields=("rotation","translation","neck_pose")
    with np.load(src) as old:
        state={k:old[k] for k in old.files}
        for k in fields:
            state[k]=state[k].copy()
            state[k][ref]=getattr(tracker,k)[ref].cpu().numpy()
        np.savez_compressed(saved,**state)
    with np.load(src) as old,np.load(saved) as new:
        for k in old.files:
            if k in fields:
                np.testing.assert_array_equal(np.delete(old[k],ref,axis=0),np.delete(new[k],ref,axis=0))
            else:
                np.testing.assert_array_equal(old[k],new[k])
    tracker.load_from_tracked_flame_params(saved)
    restored=world(tracker.forward_flame(idx)[0][0].cpu().numpy())
    np.testing.assert_allclose(restored,final,rtol=0,atol=1e-6)
    folder=out/"grid"
    folder.mkdir()
    # Stock VHAP nine-panel for the final target itself; the cross-camera comparisons below use one fixed mesh.
    tracker.clear_cache()
    _,_,v,face_t,landmarks,albedo,outputs=tracker.compute_energy(sample)
    tracker.log_media(v,face_t,landmarks,albedo,outputs,sample,ref,session="eval",epoch=0)
    copyfile(tracker.out_dir/"eval_0/image_grid"/f"frame_{ref:05d}.jpg",folder/"target-after-native.jpg")
    copyfile(Path(neck["image_grid"])/f"frame_{ref:05d}.jpg",folder/"target-before-native.jpg")
    tracker.tb_writer.close()
    results={}
    tiles=[]
    for i in display:
        label="target" if i==ref else "profile fit" if i in train else "profile held" if i in profiles else "oblique check"
        # Even on a training camera, score the separately tagged rows for that camera.
        qs=check(i) if i in [ref,*train] else rows[i]
        surfaces=[own[i],own[ref],final,transport(own[ref],i),transport(final,i)]
        audits=[measure(m,i,qs) for m in surfaces]
        row=draw_row(video,names[i],f"{label}  {yaw[i]:+.1f} deg",views[i],surfaces[:3],faces,audits[:3])
        if not cv2.imwrite(str(folder/(Path(names[i]).stem+".png")),row):
            raise IOError(names[i])
        if i in [ref,*train,*oblique] or i in closest or (i in test and rows[i]):
            tiles.append(row)
        results[names[i]]=dict(label=label,yaw=float(yaw[i]),regions=[q["region"] for q in qs],
              metrics={key:stats(v) for key,v in zip(("own","fixed_before","fixed_after","torso_before","torso_after"), audits)},
              witnesses=dict(before=audits[1],after=audits[2]))
    width=max(t.shape[1] for t in tiles)
    padded=[cv2.copyMakeBorder(t,0,0,0,width-t.shape[1],cv2.BORDER_CONSTANT) for t in tiles]
    grid=np.vstack(padded)
    if not cv2.imwrite(str(folder/"grid.png"),grid):
        raise IOError("grid.png")
    if both_sides and not cv2.imwrite(str(folder/"front-and-both-sides.jpg"),
                                      np.vstack(padded[:1+len(train)]),[cv2.IMWRITE_JPEG_QUALITY,95]):
        raise IOError("front-and-both-sides.jpg")
    video.video.release()
    evals=[results[names[i]]["metrics"]["fixed_after"] for i in test]
    signed={int(np.sign(yaw[i])) for i in test if results[names[i]]["metrics"]["fixed_after"]["measured"]>0}
    passed=(signed=={-1,1} and all(m["measured"] and not m["over3"] and not m["missing"] for m in evals)
            and not results[names[ref]]["metrics"]["fixed_after"]["over3"]
            and not results[names[ref]]["metrics"]["fixed_after"]["missing"])
    if both_sides:
        for i in train:
            m=results[names[i]]["metrics"]["fixed_after"]
            passed=bool(passed and m["measured"] and not m["over3"] and not m["missing"])
    rim=np.intersect1d(regions["neck"],regions["boundary"])
    ears=np.r_[regions["left_ear"],regions["right_ear"]]
    neck_deg=[float(inclination(m,rim,ears,world_up,world_ahead)) for m in (own[ref],final)]
    write_json(out/"result.json",dict(receipt,status="completed",accepted=passed,phases=phases,
        anatomical_inclination_before_after_deg=neck_deg,profile_data_jacobian=sensitivity,
        shared_neck_angle=False,updated_targets=[names[ref]],checkpoint=str(saved),checkpoint_sha256=input_hash(saved),
        front_photo_before=native_photo,front_landmark_shift_max_px=float(np.linalg.norm(lm-ref_lm,axis=1).max()),
        verified_unchanged_frames=len(names)-1,torso_texture_validation=tracks["texture_validation"],
        torso_weak_frames=tracks["weak_frames"],scores=results))


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    p.add_argument("--profile-space",choices=("fixed","torso"),default="fixed")
    p.add_argument("--both-sides",action="store_true",help="Use the farthest observed oblique when a true profile is unavailable")
    cv2.setNumThreads(1)
    args=p.parse_args()
    run(args.run.resolve(),args.profile_space,args.both_sides)
