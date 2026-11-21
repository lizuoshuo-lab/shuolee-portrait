"""Measure fixed-pose exports against identical full-resolution source pixels."""

from argparse import ArgumentParser
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from fuse_scene import read_gaussians, read_scene_cameras, render
from msv2_scene_engine import make_rasterizer
from render_fixed_scene_preview import caption


def read(path):
    return json.loads(path.read_bytes())


def errors(target, rendered, mask):
    def highpass(image):
        return image - cv2.GaussianBlur(image, (7, 7), 1.0)
    diff = rendered[mask] - target[mask]
    a, b = highpass(target)[mask], highpass(rendered)[mask]
    return dict(rgb_mae=float(np.abs(diff).mean()), highpass_mae=float(np.abs(a-b).mean()),
                source_detail=float(np.abs(a).mean()), rendered_detail=float(np.abs(b).mean()))


@torch.no_grad()
def run(roots, output):
    splits = [read(root/"run.json") for root in roots]
    anchor = splits[0]
    for s in splits:
        if s["status"] != "complete" or any(s[k]!=anchor[k] for k in ("source_run","reference","rgb_holdout","target_sha256")):
            raise ValueError("Model source, export pose or RGB split differs")
    source = Path(anchor["source_run"])
    scene = source/"scene"
    K,size,cameras = read_scene_cameras(scene,roots[0]/"joint")
    models = [read_gaussians(root/"joint/point_cloud.ply") for root in roots]
    main = [anchor["reference"],*read(source/"target-profile-bilateral/result.json")["train"]]
    held = anchor["rgb_holdout"]
    labels = ["Frozen mesh", "Learned mesh", "4 sites / triangle", "4 sites + fitted splats"][:len(models)]
    if len(models) not in (3,4):
        raise ValueError("Expected frozen, learned, four-site and optionally fitted splats, in order")
    per = {}
    panels = {}
    final_panels = {}
    output.mkdir(parents=True,exist_ok=False)
    for name in dict.fromkeys(main+held):
        reference = cv2.imread(str(scene/"orig_undist/images"/name))
        coarse = cv2.imread(str(scene/"headmask_undist/images"/Path(name).with_suffix(".png")),0)
        person = cv2.imread(str(scene/"mask_undist/images"/name),0)
        if reference.shape[1::-1]!=size or coarse.shape!=person.shape or coarse.shape!=reference.shape[:2]:
            raise ValueError(name)
        mask = cv2.erode(((coarse>128)&(person>250)).astype(np.uint8),np.ones((7,7),np.uint8))
        x,y,w,h = cv2.boundingRect(mask)
        x0,y0=max(0,x-16),max(0,y-16)
        crop = np.s_[y0:y+h+16,x0:x+w+16]
        support = mask[crop].astype(bool)
        if support.sum()<1000:
            raise ValueError("Insufficient native head interior for "+name)
        original = reference[crop].astype(np.float32)
        tiles = [original.astype(np.uint8)]
        scores = {}
        camera = make_rasterizer(K,size,*cameras[Path(name).stem])
        for label,model in zip(labels,models):
            rgb=render(camera,model).clamp(0,1).permute(1,2,0).cpu().numpy()
            bgr=np.ascontiguousarray(rgb[crop][...,::-1]*255)
            scores[label]=errors(original,bgr,support)
            tiles.append(np.rint(bgr).astype(np.uint8))
        line=cv2.copyMakeBorder(np.hstack(tiles),31,0,0,0,cv2.BORDER_CONSTANT)
        width=original.shape[1]
        for i,text in enumerate([f"Source {Path(name).stem}",*labels]):
            caption(line,text,i*width,width,21)
        panels[name]=line
        final_panels[name]=np.hstack((line[:,:width],line[:,-width:]))
        per[name]=dict(support_pixels=int(support.sum()),crop_origin=[x0,y0],models=scores)
    for filename,names in (("00560-front-sides.png",main),("four-rgb-holdouts.png",held)):
        width=max(panels[n].shape[1] for n in names)
        grid=np.vstack([cv2.copyMakeBorder(panels[n],0,0,0,width-panels[n].shape[1],cv2.BORDER_CONSTANT) for n in names])
        if not cv2.imwrite(str(output/filename),grid):
            raise IOError(filename)
    width=max(final_panels[n].shape[1] for n in main)
    if not cv2.imwrite(str(output/"original-final-three-cameras.png"),np.vstack(
            [cv2.copyMakeBorder(final_panels[n],0,0,0,width-final_panels[n].shape[1],cv2.BORDER_CONSTANT) for n in main])):
        raise IOError("Cannot write final fixed-pose comparison")
    avg={label:{key:float(np.mean([per[n]["models"][label][key] for n in held]))
                for key in per[held[0]]["models"][label]} for label in labels}
    own={}
    for root,label in zip(roots,labels):
        metrics=read(root/"joint/metrics.json")["frames"]
        own[label]={key:float(np.mean([metrics[n]["trained"][key] for n in held if metrics[n]["trained"][key] is not None]))
                    for key in ("head_rgb_mae_255","body_mae_255","room_mae_255")}
    (output/"fixed-detail.json").write_text(json.dumps(dict(roots=list(map(str,roots)),pose=anchor["reference"],
        rgb_holdout=held,registered_image_size=size,holdout_frame_mean=avg,own_pose_holdout_frame_mean=own,per_view=per,
        method="Exported fixed pose, identical source masks eroded by 3 px, native 0-255 RGB; signed highpass I-Gaussian(sigma=1)"),indent=2))


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("roots",type=Path,nargs="+")
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    run(args.roots,args.output)
