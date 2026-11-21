"""Compare source-visible hand-skin witness rays against archived SMPL-X surface ownership."""
import ast
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from prepare_video21_shoe_binding163 import visible_hit
from review_full_sequence123 import relocate
from run_official_gaussians123 import ASSETS, now
from update_leafwoman129_state import ROOT, atomic, read, sha

OUT = ROOT / 'grass-native-hand-rays264'
FIT = ROOT / 'grass-original-source-guard241'
NAMES = ROOT.parents[1] / 'novel-view-face/head-structure/private-runtime/python-deps/smplx-official/smplx/joint_names.py'
COLORS = {'right_hand': (234,127,36), 'left_hand': (247,212,67), 'torso': (156,113,203),
          'right_arm': (159,71,43), 'left_arm': (116,133,42), 'other': (111,123,136), 'miss': (13,15,18)}


def check_raycast():
    """Two same-image triangles at two depths, plus a ray outside both."""
    faces = np.array([[0,1,2], [3,4,5]])
    uv = np.tile([[0.,0.],[4.,0.],[0.,4.]], (2,1))
    for order in ([2.,3.],[3.,2.]):
        z = np.repeat(order,3)
        face, weights, depth = visible_hit(uv,faces,z,[1.,1.])
        assert face == int(np.argmin(order)) and np.allclose(weights,[.5,.25,.25])
        assert np.isclose(depth,2.) and visible_hit(uv,faces,z,[8.,8.]) is None
    # Perspective-correct interpolation must differ from linear image-space depth interpolation.
    z = np.array([2.,4.,4.])
    face, weights, depth = visible_hit(uv[:3],faces[:1],z,[1.,1.])
    assert face == 0 and np.isclose(depth,8./3) and np.allclose(weights,[2./3,1./6,1./6])


def region_names(faces, weights):
    tree = ast.parse(NAMES.read_text(encoding='utf-8'))
    assignment = next(t for t in tree.body if isinstance(t,ast.Assign)
                      and any(isinstance(a,ast.Name) and a.id == 'JOINT_NAMES' for a in t.targets))
    joints = ast.literal_eval(assignment.value)[:weights.shape[1]]
    assert joints[20:22] == ['left_wrist','right_wrist'] and len(joints) == 55
    torso = {'pelvis','spine1','spine2','spine3','neck','left_collar','right_collar'}
    hand_parts = ('wrist','index','middle','pinky','ring','thumb')
    groups = {side+'_hand':[i for i,n in enumerate(joints)
                            if n.startswith(side+'_') and n[len(side)+1:].startswith(hand_parts)]
              for side in ('right','left')}
    assert all(len(ids) == 16 for ids in groups.values())
    for side in ('right','left'):
        groups[side+'_arm'] = [joints.index(side+'_'+name) for name in ('shoulder','elbow')]
    groups['torso'] = [i for i,n in enumerate(joints) if n in torso]
    tags = np.full(len(faces),'other',dtype='<U10')
    for group, ids in groups.items():
        chosen = weights[:,ids].sum(1)[faces].mean(1) > .5
        assert not (chosen & (tags != 'other')).any()
        tags[chosen] = group
    assert all(np.any(tags == tag) for tag in groups)
    return tags


def inspect_witness(xy, expected, uv, faces, z, tags, image):
    x,y = xy
    hits = []
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            p = [x+dx,y+dy]
            first = visible_hit(uv,faces,z,p)
            region = 'miss' if first is None else str(tags[first[0]])
            layers = {}
            for group in (expected,'torso'):
                subset = np.flatnonzero(tags == group)
                hit = visible_hit(uv,faces[subset],z,p)
                layers[group] = None if hit is None else dict(triangle=int(subset[hit[0]]),z_model_m=hit[2])
            torso_before = (layers['torso'] is not None and layers[expected] is not None
                            and layers['torso']['z_model_m'] + 1e-6 < layers[expected]['z_model_m'])
            hits.append(dict(xy=p,first_region=region,first_triangle=None if first is None else first[0],
                             first_z_model_m=None if first is None else first[2],layer_intersections=layers,
                             torso_in_front_of_hand_on_same_ray=torso_before))
    crop = image.crop((x-20,y-20,x+20,y+20)).resize((160,160),Image.Resampling.NEAREST)
    ImageDraw.Draw(crop).rectangle((76,76,87,87),outline=(0,255,255),width=2)
    return dict(expected=expected,center_xy=xy,owner_counts=dict(Counter(r['first_region'] for r in hits)),
                hand_first_rays=sum(r['first_region']==expected for r in hits),
                torso_in_front_rays=sum(r['torso_in_front_of_hand_on_same_ray'] for r in hits),
                ray_records=hits), crop


def main():
    check_raycast()
    result_path = OUT / 'result.json'
    assert not result_path.exists()
    plan_path = OUT / 'plan.json'
    plan = read(plan_path)
    fit_path = FIT / 'result.json'
    assert sha(fit_path) == plan['fit_result_sha256']
    refs = {r['frame']:r for r in read(fit_path)['frames']}
    native_path = ROOT / 'grass-native3fps177/result.json'
    assert sha(native_path) == plan['native_receipt_sha256']
    native = read(native_path)
    manifest = relocate(native['data_root']) / 'data/inputs.json'
    assert sha(manifest) == plan['manifest_sha256']
    rows = {r['frame']:r for r in read(manifest)['records']}
    review_path = FIT / 'review/result.json'
    review = read(review_path)
    assert review['source_fit_sha256'] == plan['fit_result_sha256']
    prior_review = {r['frame']:r for r in review['records']}
    asset = relocate(ASSETS / 'smplx/SMPLX_NEUTRAL.npz')
    with np.load(asset,allow_pickle=False) as archive:
        weights = archive['weights'].astype(np.float64)
        topo = archive['f']
    records = []
    for item in plan['cohort']:
        frame = item['frame']
        assert frame not in plan['protected_training_holdouts']
        row, ref = rows[frame], refs[frame]
        raw = relocate(row['image'])
        pose = relocate(ref['output'])
        assert sha(raw)==row['source_sha256']==item['source_rgb_sha256'] and sha(pose)==ref['sha256']
        assert row['source_frame_index']==ref['source_index']==item['source_index']
        with np.load(pose,allow_pickle=False) as data:
            vertices = data['vertices_camera'][0].astype(np.float64)
            faces = data['faces']
            K = data['K_integer_centers']
            assert np.array_equal(K,row['K_integer_centers']) and np.array_equal(faces,topo)
        assert np.isfinite(vertices).all() and (vertices[:,2]>.05).all()
        q = vertices @ K.T
        uv = q[:,:2]/q[:,2,None]
        tags = region_names(faces,weights)
        with Image.open(raw) as im:
            assert im.size == tuple(row['wh'])
            source = im.convert('RGB')
        patch_views, measurements = [], []
        for hand in ('right_hand','left_hand'):
            xy = item[hand+'_skin_xy']
            if xy is None:
                continue
            hit, crop = inspect_witness(xy,hand,uv,faces,vertices[:,2],tags,source)
            measurements.append(hit)
            patch_views.append((hit,crop))

        sub = prior_review[frame]
        paths = [FIT/'review'/sub[k]['name'] for k in ('native_image','side_image')]
        for p,k in zip(paths,('native_image','side_image'),strict=True):
            assert sha(p)==sub[k]['sha256']
        with Image.open(paths[0]) as im:
            native_overlay = im.convert('RGB')
        with Image.open(paths[1]) as im:
            side = im.convert('RGB')
        scale = min(1., 1400/(native_overlay.width+side.width))
        if scale < 1:
            native_overlay = native_overlay.resize(tuple(round(n*scale) for n in native_overlay.size),Image.Resampling.LANCZOS)
            side = side.resize(tuple(round(n*scale) for n in side.size),Image.Resampling.LANCZOS)
        canvas = Image.new('RGB',(max(1020,native_overlay.width+side.width),
                           262+max(native_overlay.height,side.height)),(18,21,24))
        pen = ImageDraw.Draw(canvas)
        legend = ['Right hand','Left hand','Torso','Right arm','Left arm','Other','No hit']
        for i,(group,name) in enumerate(zip(COLORS,legend,strict=True)):
            x=10+i*135
            pen.rectangle((x,226,x+14,240),fill=COLORS[group])
            pen.text((x+20,228),name,fill='white')
        for i,(hit,patch) in enumerate(patch_views):
            x=12+i*476
            pen.text((x,12),f'{frame[-3:]} | source {hit["expected"].replace("_"," ")} skin',fill='white')
            canvas.paste(patch,(x,36))
            pen.text((x+172,12),'First mesh region (3x3 native pixels)',fill='white')
            for j,r in enumerate(hit['ray_records']):
                dx,dy=j%3,j//3
                x1,y1=x+172+dx*34,36+dy*34
                pen.rectangle((x1,y1,x1+31,y1+31),fill=COLORS[r['first_region']],outline=(104,108,112))
            pen.text((x+172,149),f'Expected region first: {hit["hand_first_rays"]}/9',fill='white')
            pen.text((x+172,170),f'Torso before expected hand: {hit["torso_in_front_rays"]}/9',fill='white')
        if len(patch_views)==1:
            pen.text((488,75),'Far left hand hidden in this source frame. No target.',fill='white')
        pen.text((10,247),'Archived native overlay and synthetic side view (same fit, no update)',fill='white')
        canvas.paste(native_overlay,(0,262))
        canvas.paste(side,(native_overlay.width,262))
        pic = OUT / f'{frame}-native-hand-rays.jpg'
        canvas.save(pic,quality=95,subsampling=0)
        records.append(dict(frame=frame,role=item['role'],source_sha256=row['source_sha256'],fit_sha256=ref['sha256'],
                            mesh_visual_hashes=[sha(p) for p in paths],witnesses=measurements,image=pic.name,
                            image_sha256=sha(pic)))
    atomic(result_path,dict(updated=now(),plan_sha256=sha(plan_path),source_fit_sha256=sha(fit_path),
           native_manifest_sha256=sha(manifest),region_names_sha256=sha(NAMES),topology_sha256=sha(asset),
           historical_review_sha256=sha(review_path),worker_sha256=sha(Path(__file__)),
           ray_method_module_sha256=sha(Path(__file__).with_name('prepare_video21_shoe_binding163.py')),
           synthetic_controls_passed=True,records=records,gaussian_optimizer_updates=0,fit_changed=False,
           true_depth_recovered=False,pose_admitted=False))
    for row in records:
        print(row['frame'], [(w['expected'],w['owner_counts'],w['torso_in_front_rays']) for w in row['witnesses']])


if __name__ == '__main__':
    main()
