"""Compare Gaussian centers and axes inside calibrated, manually delineated head silhouettes."""
import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image
from numba import njit
from plyfile import PlyData
from scipy.spatial import cKDTree
from scipy.special import expit


def read_sog(path):
    with ZipFile(path) as archive:
        meta = json.loads(archive.read("meta.json"))
        assert meta["version"] == 2
        count = meta["count"]

        def texture(field, n=0):
            with Image.open(io.BytesIO(archive.read(meta[field]["files"][n]))) as im:
                return np.asarray(im.convert("RGBA")).reshape(-1, 4)[:count].copy()

        means = meta["means"]
        levels = texture("means")[:, :3].astype(np.uint16) | (texture("means", 1)[:, :3].astype(np.uint16) << 8)
        encoded = np.asarray(means["mins"]) + (np.asarray(means["maxs"]) - means["mins"]) * (levels / 65535)
        xyz = np.sign(encoded) * np.expm1(np.abs(encoded))
        axes = np.exp(np.take(meta["scales"]["codebook"], texture("scales")[:, :3]))
        sh = texture("sh0")
        dc = np.take(meta["sh0"]["codebook"], sh[:, :3])
        packed = texture("quats")
        tag = packed[:, 3].astype(int) - 252
        assert np.all((tag >= 0) & (tag < 4))
        three = (packed[:, :3] / 255 * 2 - 1) / np.sqrt(2)
        q = np.empty((count, 4))
        for k in range(4):
            rows = tag == k
            q[np.ix_(rows, np.arange(4) != k)] = three[rows]
            q[rows, k] = np.sqrt(np.maximum(0, 1 - np.square(three[rows]).sum(1)))
        return xyz, axes, sh[:, 3] / 255, dc, q


def read_ply(path, skip):
    with ZipFile(path) as archive:
        v = PlyData.read(io.BytesIO(archive.read("point_cloud.ply")))["vertex"].data[skip:]
    def stack(keys):
        return np.column_stack([v[k] for k in keys])
    return (stack("xyz"), np.exp(stack([f"scale_{i}" for i in range(3)])), expit(v["opacity"]),
            stack([f"f_dc_{i}" for i in range(3)]), stack([f"rot_{i}" for i in range(4)]))


def camera(c):
    h, w = 640, round(640 * c["width"] / c["height"])
    sx, sy = w / c["width"], h / c["height"]
    fx, fy = c["fx"] * sx, c["fy"] * sy
    cx = c.get("cx", c["width"] / 2) * sx - .5
    cy = c.get("cy", c["height"] / 2) * sy - .5
    return w, h, fx, fy, cx, cy, np.array(c["rotation"]), np.array(c["position"])


def project(xyz, c):
    w, h, fx, fy, cx, cy, r, pos = camera(c)
    p = (xyz - pos) @ r
    uv = p[:, :2] / np.maximum(p[:, 2:], 1e-12) * [fx, fy] + [cx, cy]
    return uv, p[:, 2]


def render(data, c, viewport=None):
    # DC-only CPU preview at the same pixel centers used to trace each silhouette.
    from scipy.spatial.transform import Rotation
    xyz, axes, alpha, dc, quat = data
    w,h,fx,fy,cx,cy,r,pos = camera(c)
    if viewport is not None:
        x,y,w,h = viewport
        fx,fy = c["fx"],c["fy"]
        cx,cy = c.get("cx",c["width"]/2)-.5-x,c.get("cy",c["height"]/2)-.5-y
    pc = (xyz-pos) @ r
    valid = (pc[:,2] > .05) & (alpha > 1/255)
    z = pc[valid,2]
    uv = pc[valid,:2] / z[:,None] * [fx,fy] + [cx,cy]
    ids = np.flatnonzero(valid)
    # Permit large offscreen footprints; annotations are well inside each image.
    sel = np.ones(len(uv),bool) if viewport is not None else (np.abs(uv[:,0]-cx)<2*w) & (np.abs(uv[:,1]-cy)<2*h)
    ids, uv, z = ids[sel],uv[sel],z[sel]
    a = Rotation.from_quat(quat[ids][:,[1,2,3,0]]).as_matrix() * axes[ids,None,:]
    a = r.T @ a
    dx = (fx*a[:,0] - (uv[:,0:1]-cx)*a[:,2])/z[:,None]
    dy = (fy*a[:,1] - (uv[:,1:2]-cy)*a[:,2])/z[:,None]
    xx = np.square(dx).sum(1)+.3
    yy = np.square(dy).sum(1)+.3
    xy = (dx*dy).sum(1)
    rgb = np.clip(.5+.2820947918*dc[ids],0,1)
    return composite(w,h,uv,xx,xy,yy,alpha[ids],rgb,np.argsort(z))


@njit(cache=True)
def composite(w,h,uv,xx,xy,yy,alpha,rgb,order):
    image = np.zeros((h,w,3),np.float32)
    trans = np.ones((h,w),np.float32)
    for i in order:
        u,v=uv[i]
        rx,ry=3*np.sqrt(xx[i]),3*np.sqrt(yy[i])
        det=xx[i]*yy[i]-xy[i]**2
        for y in range(max(0,int(np.ceil(v-ry))),min(h,int(v+ry)+1)):
            for x in range(max(0,int(np.ceil(u-rx))),min(w,int(u+rx)+1)):
                if trans[y,x]<.001:
                    continue
                dx,dy=x-u,y-v
                d=(dx*dx*yy[i]+dy*dy*xx[i]-2*dx*dy*xy[i])/det
                if d>9:
                    continue
                a=min(.99,alpha[i]*np.exp(-.5*d))
                if a<1/255:
                    continue
                image[y,x]+=trans[y,x]*a*rgb[i]
                trans[y,x]*=1-a
    return image, trans


def ablate(data, cams, views, polygons, out):
    from PIL import ImageDraw
    thresholds = (.01,.03,.1)
    reports, rows = {}, []
    for view in views:
        c = cams[view]
        w,h,*_ = camera(c)
        poly = (np.asarray(polygons[str(view)])+.5)*[c["width"]/w,c["height"]/h]-.5
        lo = np.maximum(0,np.floor(poly.min(0)-8).astype(int))
        hi = np.minimum([c["width"],c["height"]],np.ceil(poly.max(0)+9).astype(int))
        viewport = (*lo,*(hi-lo))
        mask = np.zeros(tuple((hi-lo)[::-1]),np.uint8)
        cv2.fillPoly(mask,[np.rint(poly-lo).astype(np.int32)],1)
        mask = cv2.dilate(mask,np.ones((7,7),np.uint8)).astype(bool)
        rgb, trans = render(data,c,viewport)
        def tile(arr):
            return Image.fromarray(np.uint8(255*np.clip(arr,0,1)))
        tiles = [tile(rgb)]
        metrics = {}
        for t in thresholds:
            keep = data[2]>=t
            alt_rgb, alt_trans = render(tuple(a[keep] for a in data),c,viewport)
            black = alt_rgb-rgb
            white = black + (alt_trans-trans)[:,:,None]
            peak = np.maximum(np.abs(black).max(2),np.abs(white).max(2))*255
            delta = peak[mask]
            metrics[str(t)] = dict(retained_layer=int(keep.sum()), removed_layer=int((~keep).sum()),
                          p99_max_channel_delta_255=float(np.quantile(delta,.99)),
                          max_channel_delta_255=float(delta.max()),fraction_over_3=float((delta>3).mean()),
                          mean_abs_black_255=float(np.abs(black[mask]).mean()*255),
                          max_alpha_delta=float(np.abs(alt_trans-trans)[mask].max()))
            tiles.append(tile(alt_rgb))
            if t==thresholds[-1]:
                tiles.append(Image.fromarray(cv2.applyColorMap(np.uint8(np.clip(peak/12,0,1)*255),cv2.COLORMAP_MAGMA)[:,:,::-1]))
        reports[c["img_name"]] = metrics
        row=Image.new("RGB",(5*220,250))
        draw=ImageDraw.Draw(row)
        for i,(im,label) in enumerate(zip(tiles,["All","alpha >= .01","alpha >= .03","alpha >= .10","delta .10 (0-12/255)"])):
            im.thumbnail((216,215))
            row.paste(im,(i*220+(220-im.width)//2,23))
            draw.text((i*220+4,5),label,fill="white")
        draw.text((4,232),"camera "+c["img_name"],fill="white")
        rows.append(row)
    plate=Image.new("RGB",(rows[0].width,sum(r.height for r in rows)))
    for i,row in enumerate(rows):
        plate.paste(row,(0,i*row.height))
    plate.save(out/"opacity-ablation.png")
    return reports


def quantiles(x):
    return dict(zip(("p10","p50","p90","p95"), np.quantile(x, [.1,.5,.9,.95]).tolist()))


def all_camera_ablation(data,cams):
    # A static export viewed by all recorded cameras, independent of per-frame pose fitting.
    thresholds=(.005,.01)
    variants={t:tuple(a[data[2]>=t] for a in data) for t in thresholds}
    checks={}
    for c in cams:
        uv,z=project(data[0],c)
        if (z<=.01).any():
            raise ValueError("Cannot form the head crop across the near plane")
        w,h,*_=camera(c)
        pixels=(uv+.5)*[c["width"]/w,c["height"]/h]-.5
        lo=np.maximum(0,np.floor(pixels.min(0)-24).astype(int))
        hi=np.minimum([c["width"],c["height"]],np.ceil(pixels.max(0)+25).astype(int))
        viewport=(*lo,*(hi-lo))
        rgb,trans=render(data,c,viewport)
        inside=(1-trans)>.02
        if not inside.any():
            raise ValueError("Empty head view")
        checks[c["img_name"]]={}
        for t,v in variants.items():
            alt,alt_trans=render(v,c,viewport)
            black=alt-rgb
            white=black+(alt_trans-trans)[:,:,None]
            peak=np.maximum(np.abs(black).max(2),np.abs(white).max(2))[inside]*255
            checks[c["img_name"]][str(t)]=dict(p99=float(np.quantile(peak,.99)),max=float(peak.max()),
                                       fraction_over_3=float((peak>3).mean()),retained=len(v[0]))
    return checks


def measure(data, cams, views, polygons, out, name):
    xyz, scales, alpha, _, _ = data
    counts = {}
    margins = (-.04, 0., .04)
    keep = {m: np.ones(len(xyz), bool) for m in margins}
    front = None
    overlays = []
    for view in views:
        points = np.array(polygons[str(view)], np.int32)
        c = cams[view]
        w,h,*_ = camera(c)
        mask = np.zeros((h,w),np.uint8)
        cv2.fillPoly(mask,[points],1)
        signed = cv2.distanceTransform(mask, cv2.DIST_L2, 5) - cv2.distanceTransform(1-mask, cv2.DIST_L2, 5)
        uv,z = project(xyz,c)
        valid = (z > .01) & (uv[:,0]>=0) & (uv[:,0]<w-.5) & (uv[:,1]>=0) & (uv[:,1]<h-.5)
        idx = np.rint(np.clip(uv,[0,0],[w-1,h-1])).astype(int)
        dist = signed[idx[:,1],idx[:,0]]
        height = np.ptp(points[:,1])
        for m in margins:
            keep[m] &= valid & (dist >= -m*height)
        if front is None:
            front = c, uv, z, points
        im = np.array(Image.open(out/f"{name}-{view}.png"))
        cv2.polylines(im,[points],True,(80,255,150),1,cv2.LINE_AA)
        lo = np.maximum(points.min(0)-10,0)
        hi = np.minimum(points.max(0)+11,[w,h])
        crop = im[lo[1]:hi[1],lo[0]:hi[0]]
        overlays.append(Image.fromarray(crop).resize((round(200*crop.shape[1]/crop.shape[0]),200)))
    sheet = Image.new("RGB",(sum(i.width for i in overlays),200))
    x=0
    for im in overlays:
        sheet.paste(im,(x,0))
        x+=im.width
    sheet.save(out/f"{name}-contours.png")
    for m in margins:
        counts[str(m)] = [int(np.sum(keep[m] & (alpha >= t))) for t in (0,.1,.5)]
    c, uv, z, points = front
    active = keep[0] & (alpha>=.1)
    height = np.ptp(points[:,1]) * np.median(z[active]) / camera(c)[3]
    coord = xyz[active]
    nn = cKDTree(coord).query(coord,k=2)[0][:,1]
    axes = np.sort(scales[active],axis=1)
    frac = (uv[active,1]-points[:,1].min()) / np.ptp(points[:,1])
    band_counts = np.histogram(np.clip(frac,0,1),[0,.25,.5,.75,1])[0]
    # Same head-relative bin width and color scale across the two subjects.
    head_px=np.ptp(points[:,1])
    origin=[np.mean([points[:,0].max(),points[:,0].min()]),points[:,1].min()]
    xy=(uv-np.asarray(origin))/head_px
    grids=[np.histogram2d(xy[sel,1],xy[sel,0],bins=(np.arange(33)/32,np.arange(-16,17)/32))[0]
           for sel in (keep[0],active)]
    np.savez_compressed(out/f"{name}-front-bins.npz",total=grids[0],active=grids[1])
    valid_nn = nn>0
    return dict(layer_count=len(xyz), counts_margins_alpha0_01_05=counts,active_count=len(coord),
                source_front_head_height_px=float(np.ptp(points[:,1])*c["height"]/640),
                head_height_units=float(height), median_alpha=float(np.median(alpha[active])),
                nearest_fraction_height=quantiles(nn/height), max_sigma_fraction_height=quantiles(axes[:,2]/height),
                middle_sigma_fraction_height=quantiles(axes[:,1]/height), axis_ratio=quantiles(axes[:,2]/axes[:,0]),
                max_sigma_over_nearest=quantiles(axes[valid_nn,2]/nn[valid_nn]), duplicate_nn_fraction=float((~valid_nn).mean()),
                low_alpha_counts={str(t):int(np.sum(keep[0] & (alpha<t))) for t in (.001,.01,.03,.1)},
                front_height_quarters_top_to_chin=band_counts.tolist())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest",type=Path)
    p.add_argument("--all-cameras",action="store_true",help="Also measure export-only opacity cuts on each fixed-pose camera.")
    args=p.parse_args()
    config=json.loads(args.manifest.read_bytes())
    out=args.manifest.parent
    results={}
    for name,s in config.items():
        model=Path(s["model"])
        if model.suffix==".sog":
            data=read_sog(model)
            cams=json.loads(Path(s["cameras"]).read_bytes())
        else:
            data=read_ply(model,s["scene_prefix"])
            with ZipFile(model) as archive:
                cams=json.loads(archive.read("cameras.json"))
        assert all(np.isfinite(t).all() for t in data) and (data[1]>0).all()
        for view in s["views"]:
            rgb,_=render(data,cams[view])
            Image.fromarray(np.uint8(rgb.clip(0,1)*255)).save(out/f"{name}-{view}.png")
        if "polygons" in s:
            result=measure(data,cams,s["views"],s["polygons"],out,name)
            result["sha256"]=hashlib.sha256(model.read_bytes()).hexdigest()
            results[name]=result
            if s.get("opacity_ablation"):
                result["native_opacity_ablation"]=ablate(data,cams,s["views"],s["polygons"],out)
                if args.all_cameras:
                    result["fixed_pose_camera_sweep"]=all_camera_ablation(data,cams)
    if len(results)==len(config):
        (out/"distribution.json").write_text(json.dumps(results,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
