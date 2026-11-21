"""Freeze fifty physical static-PLY review cameras before the R21 candidate exists."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from plyfile import PlyData

sys.path.insert(0,str(Path(__file__).resolve().parent))
import review_smpl_delivery_round20 as previous


pin, read, sha, require, descriptor = previous.pin,previous.read,previous.sha,previous.require,previous.descriptor
review, round1 = previous.review,previous.round1
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-head-support-20261111')
DELIVERY = previous.DELIVERY.with_name('smpl-delivery-round21')
POLICY = Path('/mnt/c/Users/Administrator/Documents/New project/shuolee-coordination/point-cloud-upper-lower-view-policy-20261111.json')
POLICY_SHA = '9a5896fde8f9cec7a2158b94cce622740828c26722d94ea6215fba42667c9e4a'
DIRECTOR = previous.DIRECTOR.with_name('round20_director_review_20261111.json')
DIRECTOR_SHA = '0222df3ae07a44c10507db9695dd285cce52bbe6ad545a1a9452c14adff2e835'
R20_RECEIPT_SHA = 'be40feef7d37dc34de765746ac8f44ab4a1a02e83fd553a3cd228525def52b9d'
R20_COMMIT = 'ad52fbee2d78527e022ef145b20a497c4c4e2fa1'
RASTER_NEAR_M = .20
PROJECTION_NEAR_M = .01
HEAD_JOINTS = (15,22,23,24)

sys.path.insert(0,str(PROJECT/'src'))
from shuolee_portrait import head_support_views as camera_math

camera_center,orbit_basis,orbit_camera,project_world,to_portable = (
    camera_math.camera_center,camera_math.orbit_basis,camera_math.orbit_camera,camera_math.project_world,camera_math.to_portable)


def read_baselines(subject):
    old,locked,_,proof = previous.history(subject)
    source = pin(previous.DELIVERY/'receipt.json',R20_RECEIPT_SHA)
    director = pin(DIRECTOR,DIRECTOR_SHA)
    require(read(director)['receipt_sha256']==R20_RECEIPT_SHA and read(director)['commit']==R20_COMMIT,
            'The R20 director receipt points to a different baseline.')
    ledger = read(source)
    subject_receipt = pin(previous.DELIVERY/subject/'receipt.json',ledger['subjects'][subject]['delivery']['sha256'])
    arm = read(subject_receipt)
    run = Path(arm['result']).parent
    new = review.planned_model(run,previous.PROJECT,arm['result_sha256'])
    require(new.provenance['reference_frame_id']==old['old'].provenance['reference_frame_id']
            and list(new.rows)==locked['matched_frames'],'The fitted R20 reference frame or native roster differs.')
    proof.update(round20_receipt=descriptor(source),round20_director=descriptor(director),round20_archive=descriptor(subject_receipt),
                 user_view_priority=descriptor(pin(POLICY,POLICY_SHA)))
    arms = {}
    for name,model in (('R15',old['old']),('R20',new)):
        info = model.provenance
        run = Path(info['run'])
        terminal = read(pin(run/'result.json',info['terminal_sha256']))
        parts = round1.delivery_paths(run,terminal)
        native = model.rows[info['reference_frame_id']]
        static = previous.r13.portable(info,native,parts)
        ply = pin(static['ply'],static['ply_sha256'])
        cam = pin(static['camera_path'],static['camera_sha256'])
        if name=='R15':
            require(proof['delivered_r15_ply']['sha256']==sha(ply) and proof['delivered_r15_camera']['sha256']==sha(cam),
                    'Static R15 files differ from the delivered R15 baseline.')
        else:
            published = Path('/mnt/d/00/点云/局部改善_20261111_R20')/subject
            copies = list(published.glob('*.ply'))
            require(len(copies)==1,'Expected one existing released R20 PLY for this person.')
            proof['delivered_r20_ply'] = descriptor(pin(copies[0],sha(ply)))
            proof['delivered_r20_camera'] = descriptor(pin(published/'cameras.json',sha(cam)))
        data = PlyData.read(ply)['vertex'].data
        positions = np.column_stack([data[axis] for axis in ('x','y','z')])
        T = static['world_to_viewer']
        Q,O = np.asarray(T['rotation']),np.asarray(T['origin_world'])
        np.testing.assert_allclose(Q.T@Q,np.eye(3),rtol=0,atol=1e-12)
        require(np.linalg.det(Q)>.999999 and T['scale']==1.,'A static export changed the fitted metric coordinate scale.')
        inverse = positions @ Q + O
        world = pin(parts[0]/'world.ply',parts[2]['inputs_sha256']['world_ply'])
        world_data = PlyData.read(world)['vertex'].data
        xyz = np.column_stack([world_data[axis] for axis in ('x','y','z')])
        np.testing.assert_allclose(inverse,xyz,rtol=0,atol=2e-6)
        arms[name] = dict(static=static,centers_world=inverse,term=descriptor(run/'result.json'),
                          world_ply=descriptor(world),bound=parts[2],snapshot=parts[0],model=model)
    return arms,proof


def signed(value):
    return f'{"m" if value<0 else "p"}{abs(value):02d}'


def fixed_reference(arms):
    current = arms['R20']
    model = current['model']
    frame = model.provenance['reference_frame_id']
    row = model.rows[frame]
    slot = model.slots[frame]
    path = pin(Path(model.provenance['run'])/'fit/fitted.npz',model.provenance['fitted_sha256'])
    asset = review.ASSET
    with np.load(path,allow_pickle=False) as fit,np.load(asset,allow_pickle=False) as smpl:
        E,K = (np.asarray(row[key],dtype=np.float64) for key in ('w2c','K'))
        points = fit['joints_camera'][slot,67:118]
        source = fit['observed_native_keypoints'][slot,67:118]
        uv = points@K.T
        use = (source[:,2]>=.5)&(points[:,2]>.2)&(np.linalg.norm(uv[:,:2]/uv[:,2:]-.5-source[:,:2],axis=1)<=6.)
        require(int(use.sum())>=8 and np.array_equal(fit['faces'],smpl['f']),
                'The fixed source face or SMPL-X triangle topology is incomplete.')
        anchor = (np.median(points[use],axis=0).astype(np.float64)-E[:3,3])@E[:3,:3]
        head = np.asarray(smpl['weights'][:,HEAD_JOINTS].sum(axis=1)>=.5)
        bare = fit['vertices_world'][slot].copy()
        faces = fit['faces'].copy()
    snapshot_mesh = pin(current['snapshot']/'reference-mesh.npz',current['bound']['inputs_sha256']['reference_mesh'])
    with np.load(snapshot_mesh,allow_pickle=False) as mesh:
        require(mesh['reference_frame_id'].item()==frame and np.array_equal(mesh['faces'],faces),
                'The common static head geometry changed its reference or topology.')
        clothed = mesh['vertices_world'].copy()
    Q = np.asarray(current['static']['world_to_viewer']['rotation'])
    front,up = orbit_basis(E,anchor,Q.T@[0.,1.,0.])
    return dict(frame=frame,row=row,anchor=anchor,front=front,up=up,head=head,bare=bare,clothed=clothed,
                faces=faces,native_anchor_points=int(use.sum()),fit_file=path,mesh_file=snapshot_mesh,asset=asset)


def crop(points, wh, margin):
    W,H = wh
    low = np.maximum(np.floor(points.min(axis=0)-margin).astype(int),[0,0])
    high = np.minimum(np.ceil(points.max(axis=0)+margin+1).astype(int),[W,H])
    require(np.all(high>low),'A frozen projection crop is empty.')
    return np.r_[low,high].tolist()


def center_receipt(centers,K,E,wh):
    p,z = project_world(centers,K,E)
    W,H = wh
    image = (p[:,0]>=0)&(p[:,0]<W)&(p[:,1]>=0)&(p[:,1]<H)
    return dict(points=len(z),min_center_depth_m=float(z.min()),
        clipped_centers_total=int((z<=RASTER_NEAR_M).sum()),
        positive_clipped_centers_projected_in_sensor=int((image&(z>0)&(z<=RASTER_NEAR_M)).sum()),
        visible_centers=int((image&(z>RASTER_NEAR_M)).sum()),
        positive_visible_centers_outside_sensor=int((~image&(z>RASTER_NEAR_M)).sum()))


def grid(arms,reference):
    row = reference['row']
    W,H = wh = row['wh']
    K = np.asarray(row['K'])
    require(row['K_integer_centers']==(K-np.array([[0.,0.,.5],[0.,0.,.5],[0.,0.,0.]])).tolist(),
            'The raw camera has a different pixel-center convention.')
    target,front,up = (reference[key] for key in ('anchor','front','up'))
    head = reference['bare'][reference['head']]
    basis = arms['R20']['static']['world_to_viewer']
    views = {}
    for label,distance in (('normal',camera_math.NORMAL_DISTANCE_M),('close',camera_math.CLOSE_DISTANCE_M)):
        for az in camera_math.AZIMUTHS_DEG:
            for el in camera_math.PITCHES_DEG:
                E = orbit_camera(target,front,up,az,el,distance)
                C = camera_center(E)
                R = E[:3,:3]
                focal,depth = project_world(target,K,E)
                np.testing.assert_allclose(focal,K[:2,2]-.5,rtol=0,atol=1e-10)
                np.testing.assert_allclose(R@R.T,np.eye(3),rtol=0,atol=1e-12)
                require(np.linalg.det(R)>.999999 and abs(float(depth)-distance)<1e-12,
                        'The fixed view is not a metric rigid camera.')
                p,z = project_world(head,K,E)
                require((z>RASTER_NEAR_M).all() and (p>=0).all() and (p<[W,H]).all(),
                        'The common anatomical head crosses the camera bounds.')
                pixels = crop(p,wh,48 if label=='close' else 28)
                body,body_z = project_world(reference['clothed'],K,E)
                if label=='normal':
                    require((body_z>RASTER_NEAR_M).all() and (body>=0).all() and (body<[W,H]).all(),
                            'Normal distance cuts the frozen full body.')
                index = f'{label}_yaw_{signed(az)}_pitch_{signed(el)}'
                native = dict(frame=row['frame'],wh=wh,K=row['K'],K_integer_centers=row['K_integer_centers'],w2c=E.tolist())
                portable = dict(native,w2c=to_portable(E,basis).tolist())
                for arm in arms.values():
                    other = to_portable(E,arm['static']['world_to_viewer'])
                    back = previous.r10.baseline_world_camera(dict(native,w2c=other.tolist()),arm['static']['world_to_viewer'])
                    np.testing.assert_allclose(back['w2c'],E,rtol=0,atol=1e-10)
                views[index] = dict(distance_class=label,distance_m=distance,azimuth_deg=az,pitch_deg=el,
                    world_camera=native,portable_camera=portable,camera_center_world=C.tolist(),
                    head_crop_xyxy=pixels,normal_body_crop_xyxy=crop(body,wh,36) if label=='normal' else None,
                    projection_near_m=PROJECTION_NEAR_M,raster_center_near_m=RASTER_NEAR_M,
                    head_proxy_min_depth_m=float(z.min()),head_proxy_visible=True,
                    fixed_head_proxy_bounds_xyxy=[*p.min(axis=0).tolist(),*p.max(axis=0).tolist()],
                    baseline_centers={name:center_receipt(arm['centers_world'],K,E,wh) for name,arm in arms.items()},
                    real_rgb_ground_truth_available=False)
    for index,far in views.items():
        if far['distance_class']!='normal':
            continue
        close = views[index.replace('normal_','close_',1)]
        A,B = (np.asarray(item['world_camera']['w2c']) for item in (far,close))
        np.testing.assert_allclose(A[:3,:3],B[:3,:3],rtol=0,atol=1e-14)
        move = np.asarray(close['camera_center_world'])-np.asarray(far['camera_center_world'])
        np.testing.assert_allclose(move,A[2,:3]*(far['distance_m']-close['distance_m']),rtol=0,atol=1e-12)
        require(far['world_camera']['K']==close['world_camera']['K'] and far['world_camera']['wh']==close['world_camera']['wh'],
                'A close view zoomed or resampled instead of moving the camera.')
    require(len(views)==50,'The static acceptance roster is not 25 directions at two physical distances.')
    return views


def build(subject):
    arms,proof = read_baselines(subject)
    reference = fixed_reference(arms)
    views = grid(arms,reference)
    proof.update(reference_fitted=descriptor(reference['fit_file']),reference_mesh=descriptor(reference['mesh_file']),
                 smplx_asset=descriptor(reference['asset']),camera_math=descriptor(Path(camera_math.__file__)),planner=descriptor(Path(__file__)))
    baseline = {name:dict(portable=arm['static'],world_ply=arm['world_ply'],terminal=arm['term']) for name,arm in arms.items()}
    return dict(schema='shuolee-head-support-static-views/1',subject=subject,reference_frame=reference['frame'],
        native_K=reference['row']['K'],native_wh=reference['row']['wh'],pixel_convention='pixel_corners',
        integer_center_pixel_conversion='subtract 0.5 from both projected coordinates',
        pivot_world=reference['anchor'].tolist(),front_world=reference['front'].tolist(),up_world=reference['up'].tolist(),
        head_anchor_native_landmark_slots=reference['native_anchor_points'],fixed_portable_basis=arms['R20']['static']['world_to_viewer'],
        head_proxy_lbs_joint_ids=list(HEAD_JOINTS),head_proxy_lbs_threshold=.5,normal_distance_m=camera_math.NORMAL_DISTANCE_M,
        close_distance_m=camera_math.CLOSE_DISTANCE_M,views=views,baseline=baseline,proof=proof,
        acceptance_scope='Actual shipped static PLY at fixed world cameras. Synthetic views have no matching real RGB/alpha GT.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject',choices=previous.SUBJECTS)
    parser.add_argument('--write',action='store_true')
    args = parser.parse_args()
    manifest = build(args.subject)
    path = DELIVERY/'input-audit'/f'head-support-views-{args.subject}.json'
    if args.write:
        path.parent.mkdir(parents=True,exist_ok=True)
        data = (json.dumps(manifest,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8')
        if path.exists():
            require(path.read_bytes()==data,'The already registered static camera bytes differ.')
        else:
            with path.open('xb') as stream:
                stream.write(data)
    clipped = [(name,arm,detail['positive_clipped_centers_projected_in_sensor']) for name,row in manifest['views'].items()
               for arm,detail in row['baseline_centers'].items() if detail['positive_clipped_centers_projected_in_sensor']]
    print(json.dumps(dict(subject=args.subject,state='written' if args.write else 'checked',
        manifest=str(path) if args.write else None,sha256=sha(path) if args.write else None,
        views=len(manifest['views']),in_sensor_clipped_centers=clipped,
        min_head_proxy_depth_m=min(view['head_proxy_min_depth_m'] for view in manifest['views'].values())),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
