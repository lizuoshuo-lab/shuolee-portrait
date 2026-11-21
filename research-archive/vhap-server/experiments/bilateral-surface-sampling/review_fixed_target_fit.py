"""Score one immutable target mesh from every registered video camera."""

from argparse import ArgumentParser, Namespace
from pathlib import Path

import cv2
import numpy as np

from fit_native_skin_profile import camera, project
from fit_torso_neck import VideoPixels, contour_x
from fit_vhap_neck_sequence import contour_edges, load_observations
from bilateral_profile_views import select_both
from review_vhap_mesh_offset import exported_pose_vertices
from fit_vhap_ear_offset import EAR_IDS
from probe_vhap_profile_translation import visible_points
from review_native_own_views import read, upper_outline, witnesses
from run_video import input_hash, write_json


def summary(rows):
    values=[p["error_px"] for p in rows if p["error_px"] is not None]
    return dict(n=len(values),missing=len(rows)-len(values),over3=sum(x>3 for x in values),
                median=float(np.median(values)) if values else None,
                p90=float(np.quantile(values,.9)) if values else None,max=max(values,default=None))


def run(root, flame):
    state=read(root/"run.json")
    source=Path(state["source_run"])
    seq=source/"head-neck"
    records=read(seq/"initialization.json")["frames"]
    receipt=read(source/"target-profile-bilateral/result.json")
    names=[q["name"] for q in records]
    ref=names.index(state["reference"])
    with np.load(root/"registered-head/registered-head.npz") as tape:
        if (tape["filenames"].tolist()!=names or str(tape["checkpoint_sha256"])!=state["target_sha256"]
                or str(tape["initialization_sha256"])!=input_hash(seq/"initialization.json")):
            raise ValueError("Expected the exported target mesh in its registered gauge")
        mesh=tape["vertices_tracker"][ref] @ tape["tracker_to_world_rotation"].T * float(tape["scene_scale"]) + tape["scene_origin"]
        faces=tape["faces"]
    mesh=exported_pose_vertices(root,names[ref],mesh,faces)
    edges,eligible,_,_=contour_edges(faces,flame/"FLAME_masks.pkl")
    rows=[[] for _ in names]
    for i,sign,x,y,split,region,provider in load_observations(Namespace(scene=source/"scene",sequence=seq,
            targets=source/"neck-expanded.json",native_fallback=source/"neck-narrow.json",
            supplement=source/"front-contours.json"),records):
        rows[i].append(dict(native_xy=(x,y),side="left" if sign<0 else "right",split=split,region=region,provider=provider))
    profiles,_,_=select_both(source,records,rows)
    if [names[i] for i in profiles]!=receipt["train"]:
        raise ValueError("Different side-camera observations")
    embed=np.load(flame/"landmark_embedding_with_eyes.npy",allow_pickle=True,encoding="latin1").item()
    ids=np.arange(17,68)
    triangles=faces[embed["full_lmk_faces_idx"][0,ids]]
    sites=np.einsum("nv,nvc->nc",embed["full_lmk_bary_coords"][0,ids],mesh[triangles])
    joint_names=[f"face-{k}" for k in ids]+["left_ear","right_ear"]
    sites=np.vstack((sites,mesh[list(EAR_IDS)]))
    groups=["brows" if j<27 else "nose" if j<36 else "eyes" if j<48 else "mouth" for j in ids]+["ear_proxy"]*2
    views=[camera(r) for r in records]
    video=VideoPixels(source,views,names)
    results={}
    all_groups={}
    with np.load(source/"body-observations.npz") as rtmw:
        if rtmw["names"].tolist()!=names:
            raise ValueError("Landmark observation order differs from the registered cameras")
        ix=[rtmw["joint_names"].tolist().index(n) for n in joint_names]
        scores,flip,points=rtmw["scores"],rtmw["flip_disagreement"],rtmw["keypoints"]
    for i,(name,view) in enumerate(zip(names,views,strict=True)):
        k,r,c=view
        visible=visible_points((mesh-c)@r,(sites-c)@r,faces,k)
        valid=visible & (scores[i,ix]>0) & (flip[i,ix]<=3.)
        uv=project(sites,*view)[0]
        lm=witnesses(video,points[i,ix][valid],uv[valid],np.array(groups)[valid].tolist())
        outline,_,_=upper_outline(source/"scene",name,mesh,faces,view,video)
        measured={g:[q for q in lm if q["region"]==g] for g in dict.fromkeys(groups)}
        measured["inner_face"]=[q for q in lm if q["region"]!="ear_proxy"]
        measured["upper_outline"]=outline
        if name in receipt["scores"]:
            obs=[q for q in rows[i] if q["split"]=="check"] if i in [ref,*profiles] else rows[i]
            xy=np.array([q["native_xy"] for q in obs]).reshape(-1,2)
            predicted=contour_x(mesh,view,obs,edges,eligible)
            measured["neck"]=witnesses(video,xy,np.column_stack((predicted,xy[:,1])),[q["region"] for q in obs])
            expected=receipt["scores"][name]["witnesses"]["after"]
            np.testing.assert_allclose([q["source"] for q in measured["neck"]],[q["source"] for q in expected],atol=1e-5)
        for group,observations in measured.items():
            all_groups.setdefault(group,[]).extend(observations)
        results[name]={g:summary(rows) for g,rows in measured.items()}
    video.video.release()
    write_json(root/"fixed-target-fit.json",dict(reference=names[ref],views=results,
        pose_sha256=state["target_sha256"],observation="registered upper-person matte and repeatable RTMW face/ear proxies; original decoded pixel units",
        eligible_face="visible on target mesh, detector score > 0, mirrored detector disagreement <= 3 registered pixels",
        overall_pooled={g:summary(rows) for g,rows in all_groups.items()}))


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("gaussian_run",type=Path)
    p.add_argument("flame_assets",type=Path)
    cv2.setNumThreads(1)
    args=p.parse_args()
    run(args.gaussian_run.resolve(),args.flame_assets.resolve())
