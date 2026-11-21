"""Freeze capture coverage and the original appearance holdout before ablations."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from PIL import Image, ImageDraw

LAB = Path(__file__).resolve().parents[2]
BASE = LAB / 'outputs/mini-splatting2-computer00-001'
OUT = LAB / 'outputs/mini-splatting2-mip-depth-001'


def main():
    OUT.mkdir(exist_ok=True)
    cams = sorted(json.loads((BASE/'viewer/cameras.json').read_text()), key=lambda c:c['img_name'])
    split = json.loads((BASE/'hover-sfm-001/state.json').read_text())['appearance_split']
    centers = np.array([c['position'] for c in cams])
    rotations = np.array([c['rotation'] for c in cams])
    forward = rotations[:,:,2]
    # Best intersection of optical rays. Arbitrary SfM units, not metric depth.
    projectors = np.eye(3)[None] - forward[:,:,None]*forward[:,None,:]
    focus = np.linalg.solve(projectors.sum(0), np.einsum('nij,nj->i',projectors,centers))
    up = -rotations[:,:,1].mean(0)
    up /= np.linalg.norm(up)
    rays = centers-focus
    radii = np.linalg.norm(rays,axis=1)
    elevation = np.rad2deg(np.arcsin(np.clip(rays@up/radii,-1,1)))
    optical_pitch = np.rad2deg(np.arcsin(np.clip(forward@up,-1,1)))
    nearest_rotation=[]
    train_ids=[i for i,c in enumerate(cams) if c['img_name']+'.jpg' in split['train']]
    rows=[]
    for i,c in enumerate(cams):
        relative = np.einsum('ij,njk->nik', rotations[i].T, rotations[train_ids])
        angles = np.rad2deg(np.arccos(np.clip((np.trace(relative,axis1=1,axis2=2)-1)/2,-1,1)))
        rows.append(dict(name=c['img_name'],split='train' if i in train_ids else 'test',
                         radius_sfm=float(radii[i]),radius_relative_median=float(radii[i]/np.median(radii)),
                         elevation_deg=float(elevation[i]),optical_pitch_deg=float(optical_pitch[i]),
                         nearest_train_rotation_deg=float(angles.min())))
    qd=np.quantile(radii,[1/3,2/3]);qp=np.quantile(elevation,[1/3,2/3])
    for r in rows:
        r['distance_group']=['near','middle','far'][int(np.searchsorted(qd,r['radius_sfm']))]
        r['elevation_group']=['low','middle','high'][int(np.searchsorted(qp,r['elevation_deg']))]
    sys.path.insert(0,str(LAB/'repos/hierarchical-3d-gaussians/preprocess'))
    import read_write_model as rw
    intrinsics,images,points=rw.read_model(str(BASE/'hover-sfm-001/dataset/sparse/0'),ext='.bin')
    image_rows=[]
    for im in sorted(images.values(),key=lambda x:x.name):
        good=[int(pid) for pid in im.point3D_ids if pid>=0 and pid in points]
        image_rows.append(dict(name=im.name,points=len(good),
                               median_track_length=float(np.median([len(points[p].image_ids) for p in good])),
                               median_reprojection_error_px=float(np.median([points[p].error for p in good]))))
    video=LAB/'inputs/computer00-local-pipeline/HOVER_VERSA_0102.mp4'
    result=dict(video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
                video_bytes=video.stat().st_size,train_views=len(split['train']),test_views=len(split['test']),
                input_video_seconds=7.366083,registered_views=len(cams),sparse_points=len(points),
                coordinate_units='arbitrary COLMAP similarity scale; not metres',
                orientation_reference='mean camera up; not measured gravity',
                focus_method='least-squares intersection of all optical rays; approximate subject centre',
                radius_min_max=[float(radii.min()),float(radii.max())],radius_ratio=float(radii.max()/radii.min()),
                elevation_min_max_deg=[float(elevation.min()),float(elevation.max())],
                optical_pitch_min_max_deg=[float(optical_pitch.min()),float(optical_pitch.max())],
                focus=focus.tolist(),approximate_up=up.tolist(),cameras=rows,sfm_observations=image_rows,
                intrinsics=[dict(model=c.model,width=c.width,height=c.height,params=c.params.tolist()) for c in intrinsics.values()],
                protocol={'appearance_train':split['train'],'appearance_test':split['test'],
                          'calibration':'all 59 views participated in SfM; not unseen calibration',
                          'selection':'fixed evaluation frames; repeated use is development validation, not final independent test',
                          'multi_scale':[1,2,4,8],
                          'multi_scale_gt':'area downsample the real held-out frame and rescale intrinsics; sampling-scale test, not physical camera-distance ground truth',
                          'novel_distance_pitch':'qualitative only outside observed trajectory; no ground-truth photo or depth',
                          'human':'red-jacket subject approximately static; moving background people violate static-scene assumptions',
                          'depth_gt':'none; sparse SfM and monocular priors are not dense measured ground truth'})
    (OUT/'capture-audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    width,height=252,448
    sheet=Image.new('RGB',(width*4,(height+28)*2),'white')
    draw=ImageDraw.Draw(sheet)
    for j,name in enumerate(split['test']):
        im=Image.open(BASE/'hover-sfm-001/dataset/images'/name).convert('RGB').resize((width,height))
        x,y=(j%4)*width,(j//4)*(height+28)
        sheet.paste(im,(x,y+28));draw.text((x+6,y+8),f'test {j:02d} | source {name}',fill='black')
    sheet.save(OUT/'heldout-source-contact.jpg',quality=95)
    print(json.dumps({k:v for k,v in result.items() if k not in ['cameras','sfm_observations','protocol']},indent=2))


if __name__=='__main__':
    main()
