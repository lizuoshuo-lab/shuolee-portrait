"""Render and screen every 021 fit before selecting Gaussian training views."""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from fit_video21_native148 import CONTACTS, NATIVE, OUT
from review_full_sequence123 import bound_file, joined, mesh_renderer
from review_leafwoman_comparison144 import side_comparison
import screen_official_body_contract124 as body_screen
from select_quality_frames124 import POLICY
from update_leafwoman129_state import atomic, read, sha


def region(observed, projected, ids):
    error = np.linalg.norm(projected[ids]-observed[ids,:2],axis=1)
    w = observed[ids,2]**2
    valid = w>0
    if not valid.any():
        return None, None
    return float(np.sqrt(np.average(error**2,weights=w))),float(error[valid].max())


def knee_flexion(j):
    a,b = j[[9,12]]-j[[10,13]],j[[11,14]]-j[[10,13]]
    cosine = (a*b).sum(axis=1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1))
    return (180-np.degrees(np.arccos(np.clip(cosine,-1,1)))).tolist()


def review(anchors):
    cv2.setNumThreads(1)
    plan = read(OUT/'plan.json')
    base = OUT/'shape-anchors' if anchors else OUT
    result_path = base/'result.json'
    result = read(result_path)
    if anchors:
        assert result['plan_sha256']==sha(base/'plan.json')
    else:
        assert result['plan_sha256']==sha(OUT/'plan.json')
        assert result['fitting_complete'] or result.get('full_fits_recorded') is True
    expected = plan['shape_anchors' if anchors else 'source_frames']
    assert [r['frame'] for r in result['frames']]==expected and not set(expected)&set(plan['retained_holdouts'])
    contact = {row['frame']:row for row in read(CONTACTS)['records']}
    crops = {row['frame']:row for row in read(NATIVE/'observation-review/inputs-review.json')['rows']}
    draw_mesh,renderer_source = mesh_renderer()
    jweights = np.ones(118)
    jweights[list(body_screen.IGNORED)] = 0
    out = base/'review'
    out.mkdir()
    records,pages,pose_vectors = [],[],[]
    for i,row in enumerate(result['frames']):
        name = row['frame']
        assert sha(row['output'])==row['sha256']
        with np.load(row['output'],allow_pickle=False) as a:
            fit = dict(a)
        assert np.array_equal(fit['K_integer_centers'],plan['K_integer_centers'])
        source = contact[name]
        path = bound_file(source['paths']['body'])
        box = np.asarray(crops[name]['crop_xyxy'])
        obs, uv, joints = fit['observed_native_keypoints'][0],fit['joints_projected_native'][0],fit['joints_camera'][0]
        sharp = None
        with Image.open(path) as im:
            raw = im.convert('RGB').crop(tuple(box))
            if (obs[67:118,2] > 0).any():
                x0,y0,x1,y1 = map(int,source['face_box_xyxy'])
                face = np.asarray(im.convert('L').crop((x0,y0,x1,y1)))
                assert face.size>0
                sharp = float(cv2.Laplacian(face,cv2.CV_64F).var())
        overlay = draw_mesh(raw,box,fit)
        native = out/(name+'-native.jpg')
        joined([raw,overlay]).save(native,quality=94)
        sides = side_comparison(draw_mesh,[fit['vertices_camera']],fit['faces'])
        side = out/(name+'-side.jpg')
        sides.save(side,quality=93)
        height = source['initial_box_xyxy'][3]-source['initial_box_xyxy'][1]
        support = body_screen.measure(obs,uv,height,jweights)
        body_rms,_ = region(obs,uv,np.asarray(support['supported_body_indices']))
        face_rms,face_max = region(obs,uv,np.arange(67,118))
        feet_rms,feet_max = region(obs,uv,np.arange(19,25))
        rotation = Rotation.from_rotvec(fit['global_orient'][0]).as_matrix()
        ray = rotation.T@(-joints[8])
        yaw = float(np.degrees(np.arctan2(ray[0],ray[2])))
        flags = ['feet_reprojection_over_existing3percent_height'] if feet_max is not None and feet_max/height > POLICY['feet_max_over_height'] else []
        min_depth = float(fit['vertices_camera'][..., 2].min())
        if not np.isfinite(min_depth) or min_depth <= .05:
            flags.append('invalid_camera_mesh_depth')
        record = dict(frame=name,time_seconds=source['time_seconds'],source_fit_sha256=sha(result_path),
            yaw_proxy_deg=yaw,face_sharpness=sharp,body_confidence=float(obs[support['supported_body_indices'],2].mean()),
            body_weighted_rms_px=body_rms,face_weighted_rms_px=face_rms,face_max_px=face_max,
            feet_weighted_rms_px=feet_rms,feet_max_px=feet_max,knee_flexion_deg_right_left=knee_flexion(joints),
            min_camera_mesh_depth=min_depth,screen_flags=flags,native_image=dict(name=native.name,sha256=sha(native)),side_image=dict(name=side.name,sha256=sha(side)))
        records.append(body_screen.apply(record,support))
        pose_vectors.append(fit['body_pose_axis_angle'].reshape(21,3))
        if i % 4 == 0:
            sheet = Image.new('RGB',(1120,1320),'#171b20')
        image = joined([raw,overlay]); image.thumbnail((535,300),Image.Resampling.LANCZOS)
        sides.thumbnail((565,300),Image.Resampling.LANCZOS)
        y = (i%4)*330
        ImageDraw.Draw(sheet).text((8,y+5),name,fill='white')
        sheet.paste(image,((540-image.width)//2,y+24))
        sheet.paste(sides,(550+(565-sides.width)//2,y+24))
        if i % 4 == 3 or i+1==len(expected):
            page = out/f'overview-{i//4+1:02d}.jpg'
            sheet.save(page,quality=91)
            pages.append(dict(name=page.name,sha256=sha(page),frames=expected[i-i%4:i+1]))
        print('VIDEO21_REVIEW',name,round(body_rms,2),None if feet_rms is None else round(feet_rms,2),flush=True)
    yaw = np.rad2deg(np.unwrap(np.deg2rad([r['yaw_proxy_deg'] for r in records])))
    for r,y in zip(records,yaw,strict=True):
        r['yaw_proxy_deg'] = float(y)
    temporal = []
    for i,(before,after) in enumerate(zip(records[:-1],records[1:],strict=True)):
        diff = Rotation.from_rotvec(pose_vectors[i]).inv()*Rotation.from_rotvec(pose_vectors[i+1])
        speed = np.degrees(diff.magnitude())/(after['time_seconds']-before['time_seconds'])
        j = int(np.argmax(speed))
        temporal.append(dict(before=before['frame'],after=after['frame'],peak_deg_per_sec=float(speed[j]),joint_id=j+1,
                             review_required=bool(speed[j]>POLICY['joint_speed_deg_per_second'])))
    atomic(out/'result.json',dict(source_fit_sha256=sha(result_path),renderer_sha256=sha(renderer_source),
        frames=expected,records=records,temporal=temporal,pages=pages,only_detector_agreement=True,
        pose_depth_requires_visual_review=True,view_angle_is_a_proxy=True,actual_visual_review=False,gaussian_admitted=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--anchors',action='store_true')
    review(parser.parse_args().anchors)
