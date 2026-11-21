"""Rasterize fixed diagnostic anatomy proxies for the registered R21 cameras."""

import argparse
from hashlib import sha256
import io
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parent))
import review_round21_camera_grid as camera_grid
from shuolee_portrait.face_supplement import _front_depth


read,pin,sha,require,descriptor = camera_grid.read,camera_grid.pin,camera_grid.sha,camera_grid.require,camera_grid.descriptor
MANIFEST_SHA = {
    '016':'b3e7f1d775d9be5a493b199b0e3b8b6997fcdb4a8af3545b5c6f5748e4ca9f0d',
    '006':'c860d552162f96bb5cc335d489e744975aef1bb616881be459147fd860876505',
}
INNER_FACE = tuple(range(67,118))
REGIONS = dict(right_eye=tuple(range(86,92)),left_eye=tuple(range(92,98)),nose=tuple(range(77,86)),mouth=tuple(range(98,118)))


def camera_manifest(subject):
    path = camera_grid.DELIVERY/'input-audit'/f'head-support-views-{subject}.json'
    return pin(path,MANIFEST_SHA[subject])


def raster_support(uv, depth, faces, box):
    """A mesh triangle union on the exact native pixel grid, within one crop."""
    x0,y0,x1,y1 = box
    triangles = uv[faces]
    low,high = triangles.min(axis=1),triangles.max(axis=1)
    usable = (depth[faces]>.20).all(axis=1) & (high>=[x0,y0]).all(axis=1) & (low<[x1,y1]).all(axis=1)
    mask = np.zeros((y1-y0,x1-x0),np.uint8)
    for triangle in np.rint((triangles[usable]-[x0,y0])*16).astype(np.int32):
        cv2.fillConvexPoly(mask,triangle,1,lineType=cv2.LINE_8,shift=4)
    return mask


def hull_mask(points,box):
    x0,y0,x1,y1 = box
    result = np.zeros((y1-y0,x1-x0),np.uint8)
    if len(points)>=3:
        hull = cv2.convexHull(np.rint((points-[x0,y0])*16).astype(np.int32))
        if len(hull)>=3:
            cv2.fillConvexPoly(result,hull,1,lineType=cv2.LINE_8,shift=4)
    return result


def sha_array(value):
    return sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def reference_geometry(manifest):
    proof = manifest['proof']
    paths = {key:pin(proof[key]['file'],proof[key]['sha256']) for key in ('reference_fitted','reference_mesh','smplx_asset')}
    with np.load(paths['reference_fitted'],allow_pickle=False) as fit, \
         np.load(paths['reference_mesh'],allow_pickle=False) as mesh, \
         np.load(paths['smplx_asset'],allow_pickle=False) as asset:
        slot = fit['frame_ids'].tolist().index(manifest['reference_frame'])
        faces = fit['faces'].copy()
        require(np.array_equal(faces,mesh['faces']) and np.array_equal(faces,asset['f'])
                and mesh['reference_frame_id'].item()==manifest['reference_frame'],'The proxy topology or reference frame changed.')
        E = fit['w2c'][slot]
        landmarks = (fit['joints_camera'][slot].astype(np.float64)-E[:3,3])@E[:3,:3]
        weights = asset['weights'][:,manifest['head_proxy_lbs_joint_ids']].sum(axis=1)
        head = (weights[faces]>=manifest['head_proxy_lbs_threshold']).all(axis=1)
        clothed = mesh['vertices_world'].copy()
        bare = fit['vertices_world'][slot].copy()
        require(np.max(np.abs(bare[faces[head]]-clothed[faces[head]]))<1e-6,'Anatomical head proxy contains fitted clothing offsets.')
    return clothed,faces,head,landmarks


def build(manifest):
    vertices,faces,head,landmarks = reference_geometry(manifest)
    result,arrays = {},{}
    for key,view in manifest['views'].items():
        cam = view['world_camera']
        K,E = (np.asarray(cam[name],dtype=np.float64) for name in ('K','w2c'))
        u,z = camera_grid.project_world(vertices,K,E)
        uv,depth = camera_grid.project_world(landmarks,K,E)
        box = view['head_crop_xyxy']
        mask = raster_support(u,z,faces[head],box)
        body = raster_support(u,z,faces[~head],box)
        lo,hi = np.asarray(view['fixed_head_proxy_bounds_xyxy']).reshape(2,2)
        scale = int(max(2,round((hi-lo).max()*.025)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*scale+1,2*scale+1))
        band_radius = min(max(5,scale*4),max(8,(min(mask.shape)-1)//6))
        dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*band_radius+1,2*band_radius+1))
        inner = cv2.erode(mask,kernel)>0
        exterior = (cv2.dilate(mask,dilate)>0)&(mask==0)&(body==0)
        cam_xyz = vertices@E[:3,:3].T+E[:3,3]
        support = _front_depth(uv[list(INNER_FACE)],cam_xyz,faces[head],np.asarray(cam['K_integer_centers']))
        visible = [slot for slot,d in zip(INNER_FACE,support,strict=True)
                   if np.isfinite(d) and -.012<=depth[slot]-d<=.012]
        main_face = (hull_mask(uv[visible],box)>0)&inner
        diagnostic = dict(head_silhouette=mask>0,head_inner=inner,head_exterior_ring=exterior,visible_face_proxy=main_face)
        for region,indices in REGIONS.items():
            observed = [slot for slot in indices if slot in visible]
            diagnostic[region] = (hull_mask(uv[observed],box)>0)&inner
        for name,mask_value in diagnostic.items():
            arrays[key+'__'+name] = mask_value.astype(np.uint8)
        require(inner.any() and exterior.any(),'The fixed head proxy has no useful diagnostic samples.')
        result[key] = dict(head_crop_xyxy=box,inner_erosion_px=scale,external_band_radius_px=band_radius,
            visible_face_slots=visible,shape=list(mask.shape),
            masks={name:dict(pixels=int(value.sum()),sha256=sha_array(arrays[key+'__'+name])) for name,value in diagnostic.items()},
            exterior_excludes_projected_nonhead_mesh=True,real_rgb_ground_truth_available=False)
    return result,arrays


def save(subject):
    manifest_file = camera_manifest(subject)
    manifest = read(manifest_file)
    result,arrays = build(manifest)
    output = manifest_file.with_name(f'head-support-proxy-{subject}.npz')
    stream = io.BytesIO()
    np.savez_compressed(stream,**arrays)
    data = stream.getvalue()
    if output.exists():
        require(output.read_bytes()==data,'Preserve the registered fixed anatomy proxy arrays.')
    else:
        with output.open('xb') as target:
            target.write(data)
    proof = dict(schema='shuolee-head-support-anatomy-proxy/1',subject=subject,
                 camera_manifest=descriptor(manifest_file),mask_archive=descriptor(output),planner=descriptor(Path(__file__)),
                 views=result,semantic_scope='Bare fitted head and per-view visible facial landmark hull. Hair and glasses extend beyond this geometry. No synthetic view has a real target alpha mask.')
    path = output.with_suffix('.json')
    encoded = (json.dumps(proof,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
    if path.exists():
        require(path.read_bytes()==encoded,'The registered fixed anatomy proxy proof differs.')
    else:
        with path.open('xb') as handle:
            handle.write(encoded)
    sample = [item['masks']['visible_face_proxy']['pixels'] for item in result.values()]
    print(json.dumps(dict(subject=subject,receipt=descriptor(path),views=len(result),visible_face_sample_range=[min(sample),max(sample)],
                          empty_visible_face_views=sum(n==0 for n in sample))),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject',choices=('016','006'))
    args = parser.parse_args()
    save(args.subject)


if __name__=='__main__':
    main()
