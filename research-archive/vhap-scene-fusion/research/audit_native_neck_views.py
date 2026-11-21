"""Compare fresh native VHAP and the later neck solve in each frame's own camera."""

from argparse import ArgumentParser
import json
from pathlib import Path

import cv2
import numpy as np

from fit_native_skin_profile import camera, project
from fit_torso_neck import VideoPixels, contour_x, summarize
from fit_vhap_neck_sequence import contour_edges


def run(root,out,masks):
    out.mkdir()
    stages={}
    run_state=json.loads((root/"run.json").read_bytes())
    for label,sub,step in (("native","geometry-binding","native_vhap"),("neck","registered-head","neck-pitch")):
        tape=np.load(root/sub/"registered-head.npz")
        stages[label]=tape["vertices_tracker"]@tape["tracker_to_world_rotation"].T*float(tape["scene_scale"])+tape["scene_origin"]
    names=tape["filenames"].tolist()
    faces=tape["faces"]
    edges,eligible,_,_=contour_edges(faces,masks)
    frames=json.loads((root/"head-neck/initialization.json").read_bytes())["frames"]
    fit=json.loads((root/"neck-pitch/neck-joint-fit.json").read_bytes())
    views=[camera(f) for f in frames]
    video=VideoPixels(root,views,names)
    all_err={k:[] for k in stages}
    held_err={k:[] for k in stages}
    per_frame={}
    selected={"00360.jpg","00560.jpg","00770.jpg","00850.jpg","00860.jpg"}
    main_panels=[]
    for i,name in enumerate(names):
        q=[p for p in fit["frame_results"][name]["contours"] if p["split"]=="check"]
        target=np.array([p["native_xy"] for p in q]).reshape(-1,2)
        target_px=video.map(target)
        record={}
        draw={}
        for label,meshes in stages.items():
            x=contour_x(meshes[i],views[i],q,edges,eligible)
            pixels=video.map(np.column_stack([x,target[:,1]]))
            err=np.linalg.norm(pixels-target_px,axis=1)
            all_err[label].extend(err)
            if name in fit["held_frames"]:
                held_err[label].extend(err)
            record[label]=summarize(err)
            draw[label]=pixels
        per_frame[name]=record
        if name not in selected:
            continue
        coords={label:video.map(project(meshes[i],*views[i])[0]) for label,meshes in stages.items()}
        joined=np.vstack(list(coords.values()))
        lo=np.maximum(np.floor(joined.min(0)).astype(int)-16,0)
        hi=np.minimum(np.ceil(joined.max(0)).astype(int)+18,video.size)
        crop=video.frame(name)[lo[1]:hi[1],lo[0]:hi[0]]
        panels=[]
        for label,title in ((None,"Source "+name[:5]),("native","17:17 own"),("neck","After neck fit / own")):
            tile=crop.copy()
            subtitle="same camera and timestep"
            if label:
                xy=np.rint(coords[label]-lo).astype(np.int32)
                alpha=np.zeros(tile.shape[:2],np.uint8)
                for tri in xy[faces]:
                    cv2.fillConvexPoly(alpha,tri,255)
                cv2.drawContours(tile,cv2.findContours(alpha,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],
                                 -1,(0,220,255),1,cv2.LINE_AA)
                for a,b in zip(target_px,draw[label]):
                    ga=tuple(np.rint(a-lo).astype(int))
                    cv2.circle(tile,ga,2,(0,255,0),-1)
                    if np.isfinite(b).all():
                        cv2.line(tile,ga,tuple(np.rint(b-lo).astype(int)),(255,70,220),1)
                row=record[label]
                subtitle="no check point" if not len(q) else f"{len(q)} pts, {row['missing']} missing"
                if row["max_px"] is not None: subtitle+=f", max {row['max_px']:.1f}"
            tile=cv2.copyMakeBorder(tile,43,0,0,0,cv2.BORDER_CONSTANT)
            cv2.putText(tile,title,(3,17),cv2.FONT_HERSHEY_SIMPLEX,.44,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(tile,subtitle,(3,34),cv2.FONT_HERSHEY_SIMPLEX,.34,(230,230,230),1,cv2.LINE_AA)
            panels.append(tile)
        row=np.hstack(panels)
        if not cv2.imwrite(str(out/(name[:5]+".png")),row):
            raise IOError(name)
        if name in ("00360.jpg","00850.jpg"):
            main_panels.append(row)
    video.video.release()
    w=max(v.shape[1] for v in main_panels)
    grid=np.vstack([cv2.copyMakeBorder(v,0,0,0,w-v.shape[1],cv2.BORDER_CONSTANT) for v in main_panels])
    if not cv2.imwrite(str(out/"own-views.png"),grid):
        raise IOError("Own-view comparison")
    report={"frames":len(names),"units":"decoded original video pixels",
            "scoring":"All registered check rows, own mesh in its own camera, includes missing intersections.",
            "all":{k:summarize(x) for k,x in all_err.items()},
            "held":{k:summarize(x) for k,x in held_err.items()},
            "by_frame":per_frame,"native_command":run_state["stages"]["native_vhap"]["command"],
            "missing_views":[n for n,r in per_frame.items() if r["native"]["observations"]==0]}
    (out/"result.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report["all"]),json.dumps(report["held"]))


if __name__=="__main__":
    cli=ArgumentParser(description=__doc__)
    for k in ("run","output","flame-masks"):
        cli.add_argument("--"+k,type=Path,required=True)
    a=cli.parse_args()
    run(a.run,a.output,a.flame_masks)
