"""Filter only the head suffix of a joint export and verify it with the MSv2 CUDA rasterizer."""
from argparse import ArgumentParser
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from plyfile import PlyData, PlyElement
import torch
from diff_gaussian_rasterization_ms import GaussianRasterizationSettings, GaussianRasterizer


def gpu(array):
    return torch.as_tensor(np.ascontiguousarray(array), device="cuda", dtype=torch.float32)


def tensors(v):
    def cols(keys):
        return gpu(np.column_stack([v[k] for k in keys]))
    def numbered(prefix):
        fields=sorted((x for x in v.dtype.names if x.startswith(prefix)), key=lambda x:int(x[len(prefix):]))
        return cols(fields)
    xyz=cols("xyz")
    return dict(means3D=xyz,means2D=torch.zeros_like(xyz),dc=numbered("f_dc_")[:,None].contiguous(),
                shs=numbered("f_rest_").reshape(len(v),3,15).transpose(1,2).contiguous(),
                opacities=cols(["opacity"]).sigmoid(),scales=numbered("scale_").exp(),
                rotations=torch.nn.functional.normalize(numbered("rot_")),
                culling=torch.zeros(len(v),device="cuda",dtype=torch.bool))


def raster(c):
    w,h=c["width"],c["height"]
    fx,fy=c["fx"],c["fy"]
    r,pos=np.array(c["rotation"]),np.array(c["position"])
    view=np.eye(4)
    view[:3,:3],view[:3,3]=r.T,-r.T@pos
    near,far=.01,500.
    proj=np.array([[2*fx/w,0,2*c["cx"]/w-1,0],[0,2*fy/h,2*c["cy"]/h-1,0],
                   [0,0,far/(far-near),-far*near/(far-near)],[0,0,1,0]])
    v=gpu(view.T)
    return GaussianRasterizer(GaussianRasterizationSettings(h,w,w/(2*fx),h/(2*fy),gpu([0,0,0]),1.,
                                   v,v@gpu(proj.T),3,gpu(pos),False,False))


@torch.no_grad()
def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--min-alpha",type=float,required=True)
    args=p.parse_args()
    if not 0<args.min_alpha<1:
        p.error("--min-alpha must be between zero and one")
    stats=json.loads((args.run/"metrics.json").read_bytes())
    with ZipFile(args.run/"point-cloud-and-cameras.zip") as src:
        cams=src.read("cameras.json")
        cloud=PlyData.read(BytesIO(src.read("point_cloud.ply")))
    v=cloud["vertex"].data
    n,head=stats["scene_and_head_point_counts"]
    assert n+head==len(v) and n>0 and head>0 and len(cloud.elements)==1
    keep=np.r_[np.ones(n,dtype=bool),v["opacity"][n:]>=np.log(args.min_alpha/(1-args.min_alpha))]
    packed=BytesIO()
    PlyData([PlyElement.describe(v[keep],"vertex")],text=False,byte_order=cloud.byte_order).write(packed)
    partial=args.output.with_suffix(".partial")
    existing=args.output.exists()
    if not existing:
        with ZipFile(partial,"x",compression=ZIP_DEFLATED,compresslevel=1) as dst:
            dst.writestr("point_cloud.ply",packed.getvalue())
            dst.writestr("cameras.json",cams)
    with ZipFile(args.output if existing else partial) as check:
        assert check.testzip() is None and check.read("cameras.json")==cams
        compact=PlyData.read(BytesIO(check.read("point_cloud.ply")))["vertex"].data
        assert compact.tobytes()==v[keep].tobytes()
    original,small=tensors(v),tensors(compact)
    frames={}
    for c in json.loads(cams):
        r=raster(c)
        ref=r(**original)[0].clamp(0,1)
        test=r(**small)[0].clamp(0,1)
        peak=(ref-test).abs().amax(0)*255
        # Use every native pixel, including hair outlines and the surrounding scene.
        frames[c["img_name"]]=dict(max_delta_255=peak.max().item(),pixels_over_1=int((peak>1).sum()),
                                   pixels_over_3=int((peak>3).sum()))
    if len(frames)!=len(json.loads(cams)) or set(frames)!={Path(k).stem for k in stats["frames"]}:
        raise ValueError("Camera names differ from the recorded training run")
    report=dict(threshold=args.min_alpha,scene=n,original_head=head,kept_head=int(keep.sum()-n),
                unchanged_camera_sha256=sha256(cams).hexdigest(),frames=frames,
                worst_max_255=max(f["max_delta_255"] for f in frames.values()),
                total_pixels_over_1=sum(f["pixels_over_1"] for f in frames.values()),
                total_pixels_over_3=sum(f["pixels_over_3"] for f in frames.values()))
    report["passed_export_gate"]=report["worst_max_255"]<=1.
    args.output.with_suffix(".json").write_text(json.dumps(report,indent=2))
    if not report["passed_export_gate"]:
        if not existing:
            partial.unlink()
        raise ValueError("The compact model exceeds 1/255 native RGB difference; keep the full export")
    if not existing:
        partial.rename(args.output)


if __name__=="__main__":
    main()
