"""Read back initial FLAME seeds on two registered original video frames."""

from argparse import ArgumentParser
import json
from pathlib import Path
import pickle

import cv2
import numpy as np
import torch
from vhap.model.flame import FlameHead

from fit_native_skin_profile import camera, project
from fit_guided_vhap import camera_gauge
from fit_torso_neck import VideoPixels


@torch.no_grad()
def main(run, seed, output):
    init = json.loads((run/"head-neck/initialization.json").read_bytes())
    rows = init["frames"]
    names = [r["name"] for r in rows]
    ref = names.index(json.loads((run/"split.json").read_bytes())["reference"])
    cameras = [camera(r) for r in rows]
    video = VideoPixels(run,cameras,names)
    data = json.loads((seed/"input-neck.json").read_bytes())
    if data["frames"] != names:
        raise ValueError("Unexpected input frame order")
    base = np.load(json.loads((run/"hair/result.json").read_bytes())["checkpoint"])
    new = np.load(seed/"input-seeds.npz")
    if new["names"].tolist()!=names:
        raise ValueError("GPU seed readback does not match the camera order")
    mesh = FlameHead(300,100,add_teeth=True)
    tris = mesh.faces.numpy()
    with Path("asset/flame/FLAME_masks.pkl").open("rb") as f:
        regions = pickle.load(f,encoding="latin1")
    rim = np.intersect1d(regions["neck"],regions["boundary"])
    ears = np.r_[regions["left_ear"],regions["right_ear"]]
    scale = init["scene_scale"]
    origin,gauge = camera_gauge(rows,scale)[3:5]
    clouds = []
    for state, angle in [(base,data["old_deg"][ref]),(new,data["new_deg"][ref])]:
        def value(k):
            return torch.from_numpy(state[k][ref:ref+1]).float()
        neck = value("neck_pose") if state is new else torch.zeros(1,3)
        v = mesh(torch.zeros(1,300),torch.zeros(1,100),value("rotation"),neck,torch.zeros(1,3),
                 torch.zeros(1,6),value("translation"))[0][0].numpy()@gauge*scale+origin
        clouds.append((np.vstack([v,v[rim].mean(0),v[ears].mean(0)]),angle))
    frames=[]
    for name in ("00770.jpg","00360.jpg"):
        c = cameras[names.index(name)]
        raw = video.frame(name)
        pix = [video.map(project(w,*c)[0]) for w,_ in clouds]
        lo = np.maximum(np.min(np.vstack(pix),axis=0).astype(int)-20,0)
        hi = np.minimum(np.max(np.vstack(pix),axis=0).astype(int)+21,raw.shape[1::-1])
        crop = raw[lo[1]:hi[1],lo[0]:hi[0]]
        panels = []
        for i,label in enumerate((name,"Old init %.2f deg"%clouds[0][1],"New init %.2f deg"%clouds[1][1])):
            tile = crop.copy()
            if i:
                xy = np.rint(pix[i-1]-lo).astype(np.int32)
                alpha = np.zeros(tile.shape[:2],np.uint8)
                for tri in xy[tris]:
                    cv2.fillConvexPoly(alpha,tri,255)
                lines = cv2.findContours(alpha,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
                cv2.drawContours(tile,lines,-1,(0,215,255),1,cv2.LINE_AA)
                cv2.line(tile,tuple(xy[-2]),tuple(xy[-1]),(255,220,0),2,cv2.LINE_AA)
            tile = cv2.copyMakeBorder(tile,27,0,0,0,cv2.BORDER_CONSTANT)
            cv2.putText(tile,label,(4,18),cv2.FONT_HERSHEY_SIMPLEX,.44,(255,255,255),1,cv2.LINE_AA)
            panels.append(tile)
        frames.append(np.hstack(panels))
    width = max(f.shape[1] for f in frames)
    grid = np.vstack([cv2.copyMakeBorder(f,0,0,0,width-f.shape[1],cv2.BORDER_CONSTANT) for f in frames])
    if not cv2.imwrite(str(output),grid):
        raise IOError(output)
    video.video.release()


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    for k in ("run","seed","output"):
        parser.add_argument("--"+k,type=Path,required=True)
    a=parser.parse_args()
    main(a.run,a.seed,a.output)
