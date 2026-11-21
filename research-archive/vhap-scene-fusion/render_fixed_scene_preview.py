"""Compare the exported fixed-pose Gaussian PLY to the registered source images."""

from argparse import ArgumentParser
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
import torch

from fuse_scene import project, read_gaussians, read_scene_cameras, render, save_jpg, to_bgr
from msv2_scene_engine import make_rasterizer
from review_vhap_mesh_offset import exported_pose_vertices


def caption(canvas, text, x, width, baseline):
    size=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,.5,1)[0][0]
    scale=.5*min(1.,(width-12)/max(size,1))
    cv2.putText(canvas,text,(x+5,baseline),cv2.FONT_HERSHEY_SIMPLEX,scale,(248,248,248),1,cv2.LINE_AA)


def panel(left, right, label, reference):
    pair=np.hstack((left,right))
    pair=cv2.copyMakeBorder(pair,28,0,0,0,cv2.BORDER_CONSTANT)
    caption(pair,label,0,left.shape[1],19)
    caption(pair,f"Gaussian | pose {reference}",left.shape[1],right.shape[1],19)
    return pair


def mesh_overlay(source, xy, camera_vertices, faces):
    triangles=np.rint(xy[faces]*16).astype(np.int32)
    xyz=camera_vertices[faces]
    normals=np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0])
    normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-10)
    light=.32+.68*np.abs(normals@np.array([-.35,-.25,-.90]))
    paint=source.copy()
    for i in np.argsort(xyz[:,:,2].mean(1))[::-1]:
        cv2.fillConvexPoly(paint,triangles[i],(light[i]*np.array([215,214,106])).astype(int).tolist(),shift=4)
    return cv2.addWeighted(source,.42,paint,.58,0)


@torch.no_grad()
def run(root, output=None):
    state=json.loads((root/"run.json").read_bytes())
    source=Path(state["source_run"])
    model=root/"joint"
    result=json.loads((source/"target-profile-bilateral/result.json").read_bytes())
    if state["status"]!="complete" or result["checkpoint_sha256"]!=state["target_sha256"]:
        raise ValueError("Expected a completed fit with the requested bilateral target")
    scene=source/"scene"
    k,size,cameras=read_scene_cameras(scene,model)
    camera_file=json.loads((model/"cameras.json").read_bytes())
    if any(not np.allclose([c["cx"],c["cy"]],[k[0,2]+.5,k[1,2]+.5],rtol=0,atol=1e-7) for c in camera_file):
        raise ValueError("Exported principal point differs from the source")
    points=read_gaussians(model/"point_cloud.ply")
    ref=state["reference"]
    with np.load(root/"registered-head/registered-head.npz") as z:
        names=z["filenames"].tolist()
        vertices=z["vertices_tracker"][names.index(ref)] @ z["tracker_to_world_rotation"].T * float(z["scene_scale"]) + z["scene_origin"]
        faces=z["faces"]
    vertices=exported_pose_vertices(root, ref, vertices, faces)
    main=[ref,*result["train"]]
    held=state["rgb_holdout"]
    output=output or root/"fixed-preview"
    output.mkdir(exist_ok=False)
    rows={}
    triples={}
    photo={}
    view_size=(round(720*size[0]/size[1]/2)*2,720)
    encoder=["ffmpeg","-nostdin","-v","error","-f","rawvideo","-pix_fmt","bgr24","-s",
             f"{view_size[0]*2}x{view_size[1]+28}","-r","3","-i","-","-c:v","libx264","-pix_fmt","yuv420p",
             "-movflags","+faststart",str(output/f"fixed-{Path(ref).stem}.mp4")]
    with subprocess.Popen(encoder,stdin=subprocess.PIPE) as writer:
        for name in names:
            stem=Path(name).stem
            R,C=cameras[stem]
            original=cv2.imread(str(scene/"orig_undist/images"/name))
            if original is None or original.shape[1::-1]!=size:
                raise ValueError(f"Source camera or native raster missing for {name}")
            image=to_bgr(render(make_rasterizer(k,size,R,C),points))
            tag="RGB holdout" if name in held else "RGB train"
            video_pair=panel(cv2.resize(original,view_size,interpolation=cv2.INTER_AREA),
                             cv2.resize(image,view_size,interpolation=cv2.INTER_AREA),f"Source {stem}  {tag}",Path(ref).stem)
            writer.stdin.write(video_pair.tobytes())
            camera_vertices=(vertices-C)@R
            xy=project(camera_vertices,k)
            lo=np.maximum(0,np.floor(xy.min(0)).astype(int)-[32,20])
            hi=np.minimum(size,np.ceil(xy.max(0)).astype(int)+[32,52])
            crop=np.s_[lo[1]:hi[1],lo[0]:hi[0]]
            delta=image[crop].astype(np.float32)-original[crop]
            photo[name]=dict(mae_255=float(np.abs(delta).mean()),
                             psnr_db=float(10*np.log10(255**2/max(1e-10,float(np.mean(delta**2))))),
                             crop_xyxy=[*lo.tolist(),*hi.tolist()])
            if name in main or name in held or name in result["held_profiles"]:
                pair=panel(original[crop],image[crop],f"Source {stem}  {tag}",Path(ref).stem)
                save_jpg(output/f"{stem}.jpg",pair)
                rows[name]=pair
            if name in main:
                overlay=mesh_overlay(original[crop],xy-lo,camera_vertices,faces)
                line=np.hstack((original[crop],overlay,image[crop]))
                line=cv2.copyMakeBorder(line,34,0,0,0,cv2.BORDER_CONSTANT)
                w=overlay.shape[1]
                for j,label in enumerate((f"Source | camera {stem}",f"Mesh | pose {Path(ref).stem}",f"Gaussian | pose {Path(ref).stem}")):
                    caption(line,label,j*w,w,23)
                triples[name]=line
        writer.stdin.close()
        if writer.wait()!=0:
            raise RuntimeError("Could not encode fixed-pose preview")
    for filename, selected in [("front-sides.jpg",main),("rgb-holdout.jpg",held)]:
        width=max(rows[n].shape[1] for n in selected)
        save_jpg(output/filename,np.vstack([cv2.copyMakeBorder(rows[n],0,0,0,width-rows[n].shape[1],cv2.BORDER_CONSTANT)
                                            for n in selected]))
    width=max(t.shape[1] for t in triples.values())
    save_jpg(output/"source-mesh-gaussian.jpg",np.vstack([cv2.copyMakeBorder(triples[n],0,0,0,width-triples[n].shape[1],cv2.BORDER_CONSTANT)
                                                          for n in main]))
    (output/"rgb-fit.json").write_text(json.dumps(dict(reference=ref,metric="fixed-pose head and neck rectangular crop, registered source pixels",views=photo),indent=2))


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("gaussian_run",type=Path)
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    run(args.gaussian_run.resolve(),args.output)
