"""Review fixed native poses and actual static PLYs for the round-four delivery."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_smpl_delivery_round3 as prior


review, round1, transfer = prior.review, prior.round1, prior.transfer
read, pin, require, sha = prior.read, prior.pin, prior.require, prior.sha
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-shape-20261111')
ROUND3_COMMIT = '967c286d023c47ae752991f5edfe4c09afc6c6f9'
ROUND3_RECEIPT = Path(__file__).resolve().parents[1]/'outputs/smpl-delivery-round3/receipt.json'
ROUND3_RECEIPT_SHA = 'ed54c1fa0e80d32b1126728f9b6b77b086a2887706198b99d27f1475e6ce3ef9'
ROUND3_REVIEWER_SHA = '9d4f98adee7b6f680e97b7f5c4ac8f578143d29a4aa6d2932fa7c4f8d3893d82'
FINAL_SNAPSHOT = prior.FINAL_SNAPSHOT+'-shape-001800'
IDENTITY_FIELDS = ('fitted_sha256','training_inputs_sha256','protected_holdouts','reference_frame_id')


def center(w2c):
    matrix = np.asarray(w2c, dtype=np.float64)
    rotation = matrix[:3,:3]
    require(matrix.shape == (4,4) and np.isfinite(matrix).all() and np.allclose(matrix[3], [0,0,0,1])
            and np.allclose(rotation.T@rotation, np.eye(3), atol=1e-6, rtol=0)
            and np.isclose(np.linalg.det(rotation),1,atol=1e-6,rtol=0), 'Invalid rigid w2c.')
    return -rotation.T@matrix[:3,3]


def unit(direction):
    norm = np.linalg.norm(direction)
    require(np.isfinite(norm) and norm > 1e-8, 'Degenerate camera/mesh direction.')
    return direction/norm


def mesh_bases(joints, vertices):
    neck, right_shoulder, left_shoulder, hips = joints[[1,2,5,8]]
    right = unit(right_shoulder-left_shoulder)
    up = unit(neck-hips-np.dot(neck-hips,right)*right)
    torso = ((neck+hips)*.5, np.stack((right,up,unit(np.cross(up,right)))))
    nose,right_ear,left_ear = vertices[[9120,616,6]]
    ears = (left_ear+right_ear)*.5
    right = unit(right_ear-left_ear)
    forward = unit(nose-ears-np.dot(nose-ears,right)*right)
    head = (ears,np.stack((right,unit(np.cross(right,forward)),forward)))
    return dict(torso=torso,head=head)


def local_direction(camera_center, basis):
    origin, axes = basis
    right,up,forward = xyz = axes@unit(camera_center-origin)
    return dict(unit_direction_local=xyz.tolist(),
                azimuth_degrees=float(np.degrees(np.arctan2(right,forward))),
                elevation_degrees=float(np.degrees(np.arcsin(np.clip(up,-1.,1.)))),
                origin_world=origin.tolist(), right_up_forward_world=axes.tolist())


def angular_support(model, views, transform, selection):
    run = Path(model.provenance['run'])
    metadata = read(run/'data/training-inputs.json')
    sources, bases = [], {}
    with np.load(run/'fit/fitted.npz',allow_pickle=False) as archive:
        joints, fitted_w2c = archive['joints_camera'], archive['w2c']
        source_w2c = archive['source_w2c']
        for frame,row in model.rows.items():
            slot = model.slots[frame]
            matrix = np.asarray(row['w2c'])
            require(np.allclose(matrix,fitted_w2c[slot],atol=1e-7,rtol=0)
                    and np.allclose(row['source_w2c'],source_w2c[slot],atol=1e-7,rtol=0),
                    'Angular proxy camera does not match the frozen fit/source.')
            world_joints = (joints[slot]-matrix[:3,3])@matrix[:3,:3]
            bases[frame] = mesh_bases(world_joints, model.fit['vertices_world'][slot])
            camera_center = center(matrix)
            sources.append(dict(frame=frame,source_frame_index=row['source_frame_index'],source_sha256=row['source_sha256'],
                camera_kind=row['camera_kind'],camera_registration=row['camera_registration'],
                fit_w2c=row['w2c'],source_w2c=row['source_w2c'],camera_center_fitted_world=camera_center.tolist(),
                mesh_direction_proxy={key:local_direction(camera_center,basis) for key,basis in bases[frame].items()}))
    origin, rotation = np.asarray(transform['origin_world']),np.asarray(transform['rotation'])
    virtual = {}
    for name,item in views.items():
        camera = item['camera']
        camera_center = center(camera['w2c'])@rotation+origin
        directions = {key:local_direction(camera_center,basis) for key,basis in bases[camera['frame']].items()}
        for key, direction in directions.items():
            axis = direction['unit_direction_local']
            matches = []
            for source in sources:
                source_axis = source['mesh_direction_proxy'][key]['unit_direction_local']
                angle = np.degrees(np.arccos(np.clip(np.dot(axis,source_axis),-1.,1.)))
                matches.append((float(angle),source['frame']))
            angle,frame = min(matches)
            direction.update(nearest_selected_source_proxy_frame=frame,proxy_gap_degrees=angle)
        virtual[name] = dict(synthetic_camera=item['synthetic'],orbit_y_degrees=item.get('orbit_y_degrees'),
            fixed_pose_frame=camera['frame'], camera_center_fitted_world=camera_center.tolist(),
            mesh_direction_proxy=directions,source_rgb_available=item['source_rgb_available'],source_rgb_compared=False)
    return dict(physical_angular_coverage='unknown', formal_calibration_accepted=metadata['formal_calibration_accepted'],
        all_fit_w2c_identity=all(np.array_equal(row['fit_w2c'],np.eye(4)) for row in sources),
        source_time_bin_coverage=selection['coverage'],source_time_bin_coverage_basis=selection['coverage_basis'],
        sources=sources,static_views=virtual,
        basis='SMPL-X fitted OpenPose shoulders/neck/hips and surface nose 9120, right ear 616, left ear 6.',
        scope='Model-conditioned camera-to-mesh directions for selected source frames. Neither time-bin occupancy nor fitted angles establish physical camera motion, true angular coverage, or visible-surface support.')


def oblique_view(front, vertices, pivot):
    angle = np.pi/4
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.array([[cosine,0.,sine],[0.,1.,0.],[-sine,0.,cosine]])
    orbit_center = pivot+rotation@(center(front['w2c'])-pivot)
    forward = unit(pivot-orbit_center)
    right = unit(np.cross(forward,[0.,1.,0.]))
    orbit = np.eye(4)
    orbit[:3,:3] = np.stack((right,np.cross(forward,right),forward))
    orbit[:3,3] = -orbit[:3,:3]@orbit_center
    camera = dict(front,w2c=orbit.tolist())
    uv, depth = prior.prior.projection(vertices,camera)
    require(np.isfinite(uv).all() and (depth > .2).all(), 'The 45-degree camera intersects the framing mesh.')
    lo,hi = uv.min(axis=0),uv.max(axis=0)
    pad = max(64.,.12*float(hi[1]-lo[1]))
    crop = round1.native_box([*(lo-pad),*(hi+pad)],front['wh'])
    return dict(camera=camera,crop_xyxy=crop,synthetic=True,source_rgb_available=False,
                orbit_y_degrees=45,pivot_viewer_world=pivot.tolist())


def preflight(subject, commit=None):
    pin(ROUND3_RECEIPT, ROUND3_RECEIPT_SHA)
    pin(Path(prior.__file__), ROUND3_REVIEWER_SHA)
    receipt = read(ROUND3_RECEIPT)
    require(receipt['commit'] == ROUND3_COMMIT and receipt['baseline_commit'] == prior.ROUND2_COMMIT,
            'The round-three delivery identity changed.')
    spec = receipt['subjects'][subject]
    baseline_path = Path(spec['review'])
    pin(baseline_path,spec['review_sha256'])
    baseline = read(baseline_path)
    previous_plan = baseline_path.with_name('plan.json')
    pin(previous_plan,baseline['plan_sha256'])
    locked = read(previous_plan)
    roi_path = previous_plan.with_name('rois.npz')
    pin(roi_path,locked['roi_archive_sha256'])
    pin(Path(prior.prior.__file__),locked['reviewer_sha256'])
    require(baseline['state'] == 'round3_delivery_views_complete' and locked['new_commit'] == ROUND3_COMMIT
            and locked['round3_reviewer_sha256'] == ROUND3_REVIEWER_SHA
            and locked['regions'] == baseline['regions']
            and tuple(pair['new']['frame'] for pair in locked['pairs']) == prior.prior.SCOPES[subject],
            'The cached native camera/crop lineage changed.')
    for item in baseline['panels']:
        pin(baseline_path.parent/item['file'],item['sha256'])
    old,old_parts,_ = prior.prior.baseline(subject)
    third_run = Path(spec['result']).parent
    third = review.planned_model(third_run,prior.PROJECT,spec['result_sha256'])
    third_parts = round1.delivery_paths(third_run,read(third_run/'result.json'))
    require(locked['old'] == old.provenance and locked['new'] == third.provenance
            and third_parts[0].name == prior.FINAL_SNAPSHOT,'A locked comparison endpoint changed.')
    roots = {'round1':(old,old_parts),'round3':(third,third_parts)}
    output = review.LAB/f'shuolee-smpl-delivery-round4-{subject}'
    candidate = None
    if commit:
        require(re.fullmatch('[0-9a-f]{40}',commit) is not None, 'Pin a full candidate Git SHA.')
        transfer.rebound(transfer.base.common.git_state,PROJECT=PROJECT)(commit)
        wrapper = read(output/'wrapper-terminal.json')
        require(wrapper['state'] == 'completed' and wrapper['candidate_commit'] == commit,'Round-four wrapper is incomplete.')
        run = output/'run'
        candidate = review.planned_model(run,PROJECT,wrapper['result_sha256'])
        terminal = read(run/'result.json')
        parts = round1.delivery_paths(run,terminal)
        require(parts[0].name == terminal['final_snapshot'] == FINAL_SNAPSHOT
                and terminal['fit_reused'] is True and terminal['fit_calls'] == 0,
                'Round four changed the frozen fit or its expected terminal snapshot.')
        round1.check_source_equivalence(prior.PROJECT,PROJECT,third_parts[2]['source_files_sha256'],parts[2]['source_files_sha256'])
        roots['round4'] = candidate,parts
    for key,(model,_) in roots.items():
        require(all(model.provenance[field] == old.provenance[field] for field in IDENTITY_FIELDS),
                f'{key} changed the frozen fit, camera or training/holdout identity.')
        for pair in locked['pairs']:
            frame = pair['new']['frame']
            require(review.pose_digest(model,frame) == pair['new']['pose_sha256'],f'{key} changed a locked review pose.')
    front = locked['views']['front']['camera']
    arms = {key:prior.portable(model.provenance,front,parts) for key,(model,parts) in roots.items()}
    front_view = locked['static_views']['static_reference_front']
    side_view = locked['static_views']['static_rotated_side']
    transform = arms['round1']['world_to_viewer']
    transform_fields = ('rotation','origin_world','scale','export_frame_mode','reference_frame_id','reference_pose_index')
    require(all(item['front'] == front_view['camera']
                and all(item['world_to_viewer'][key] == transform[key] for key in transform_fields) for item in arms.values()),
            'Portable reference camera or world transform changed.')
    with np.load(old_parts[0]/'reference-mesh.npz',allow_pickle=False) as archive:
        mesh = archive['vertices_world']
    vertices = (mesh-np.asarray(transform['origin_world']))@np.asarray(transform['rotation']).T
    pivot = np.asarray(side_view['pivot_viewer_world'])
    views = dict(static_reference_front=front_view,static_rotated_oblique=oblique_view(front_view['camera'],vertices,pivot),
                 static_rotated_side=side_view)
    for arm in arms.values():
        points = PlyData.read(arm['ply'])['vertex'].data
        xyz = np.stack([points[axis] for axis in ('x','y','z')],axis=1)
        arm['min_center_depth_m'] = {}
        for key,view in views.items():
            z = np.asarray(view['camera']['w2c'][2])
            depth = float((xyz@z[:3]+z[3]).min())
            require(np.isfinite(depth) and depth>.2,'Static portable view crosses the stock raster cutoff.')
            arm['min_center_depth_m'][key] = depth
    support = angular_support(old,views,transform,read(Path(old.provenance['run'])/'fit/selection.json'))
    labels = {name:dict(frame=item['frame'], label={'front':'native_reference_source','side':'native_oblique_source',
              'dolly':'synthetic_fixed_focal_dolly'}[name],native_rgb_truth=item['native_rgb_truth'],
              source_frame_index=old.rows[item['frame']]['source_frame_index'] if item['native_rgb_truth'] else None)
              for name,item in locked['views'].items()}
    plan = dict(locked,new=candidate.provenance if candidate else None,new_commit=commit,output=str(output/'view-comparison'),
                round4_reviewer_sha256=sha(Path(__file__)),round3_plan_sha256=baseline['plan_sha256'],
                baseline_review_sha256=spec['review_sha256'],baseline_review=str(baseline_path),
                static_views=views,static_arms=arms,source_angular_support=support,native_view_labels=labels)
    return plan,baseline,roi_path


def assemble(output,plan,previous,new,regions,static):
    panels = []
    records = {item['role']:item for item in new['records']}
    previous_root = Path(plan['baseline_review']).parent
    names = ['round1_762d6d0','round3_967c286',f'round4_{plan["new_commit"][:7]}']
    for item in previous['panels']:
        name,scale = item['view'],item['scale']
        if name.startswith('static_'):
            continue
        if name in ('front','side'):
            new_panel = records[name]['native_panel']
            actual = records[name]['native_camera']
        elif name == 'dolly':
            new_panel = records['front']['closeup']['panel']
            actual = records['front']['closeup']['camera']
        else:
            new_panel = regions['head_panels'][name.removesuffix('_head')]
            actual = records[name.removesuffix('_head')]['native_camera']
        expected = plan['views'][name.removesuffix('_head')]['camera']
        require(all(actual[key] == expected[key] for key in ('frame','K','w2c','wh')),
                'A rendered native or dolly camera changed.')
        require(item['header_pixels'] == 0 and item['crop_xyxy'] == new_panel['crop_xyxy']
                and item['rows'][0] == names[0] and item['rows'][2] == names[1],'Review crops or row identities changed.')
        old_path,new_path = previous_root/item['file'],output/new_panel['file']
        pin(old_path,item['sha256'])
        pin(new_path,new_panel['sha256'])
        with Image.open(old_path) as old,Image.open(new_path) as candidate:
            w,h = old.width,old.height//3
            require(old.height == 3*h and candidate.size == (w,2*h) and scale in (1.,.5),
                    'Matched native panel dimensions differ.')
            index = 0 if scale == 1. else 1
            sheet = Image.new('RGB',(w,3*h))
            sheet.paste(old.crop((0,0,w,h)),(0,0))
            sheet.paste(old.crop((0,2*h,w,3*h)),(0,h))
            sheet.paste(candidate.crop((0,index*h,w,(index+1)*h)),(0,2*h))
            destination = output/item['file']
            sheet.save(destination)
        panels.append(dict(item,sha256=sha(destination),rows=names))
    for name,view in plan['static_views'].items():
        tiles = []
        for arm in ('round1','round3','round4'):
            item = static[arm][name]
            path = output/item['file']
            pin(path,item['sha256'])
            require(all(item['camera'][key] == view['camera'][key] for key in ('frame','K','w2c','wh'))
                    and item['static_ply_sha256'] == plan['static_arms'][arm]['ply_sha256']
                    and item['pose_driven'] is False and item['source_rgb_compared'] is False,
                    'A static render camera or PLY changed.')
            with Image.open(path) as image:
                tiles.append(image.convert('RGB'))
        w,h = tiles[0].size
        x0,y0,x1,y1 = view['crop_xyxy']
        require((w,h) == (3*(x1-x0),y1-y0) and all(tile.size == (w,h) for tile in tiles),
                'Matched static panel dimensions differ.')
        sheet = Image.new('RGB',(w,3*h))
        for index,tile in enumerate(tiles):
            sheet.paste(tile,(0,index*h))
        path = output/f'{name}-{view["camera"]["frame"]}-scale-1.0.png'
        sheet.save(path)
        panels.append(dict(file=path.name,sha256=sha(path),frame=view['camera']['frame'],view=name,scale=1.,header_pixels=0,
            crop_xyxy=view['crop_xyxy'],rows=names,columns=['black','white','alpha'],pose_driven=False,
            source_rgb_available=view['source_rgb_available'],source_rgb_compared=False,
            synthetic_camera=view['synthetic'],orbit_y_degrees=view.get('orbit_y_degrees')))
    require(len(panels) == 13,'Unexpected finite delivery panel count.')
    return panels


def main():
    if len(sys.argv) == 4 and sys.argv[1] == '_worker':
        path,fd = Path(sys.argv[2]),int(sys.argv[3])
        plan = read(path)
        pin(Path(__file__),plan['round4_reviewer_sha256'])
        pin(Path(prior.__file__),ROUND3_REVIEWER_SHA)
        prior.prior.worker(path,'new',fd)
        prior.static_worker(path)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('check','render'))
    parser.add_argument('subject',choices=prior.prior.SCOPES)
    parser.add_argument('--commit')
    args = parser.parse_args()
    if args.action == 'check':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    elif not args.commit:
        parser.error('--commit must pin the exact completed round-four candidate')
    plan,baseline,rois = preflight(args.subject,args.commit)
    if args.action == 'check':
        support = plan['source_angular_support']
        print(json.dumps(dict(state='cpu_preflight_passed',terminal_ready=plan['new'] is not None,gpu_calls=0,
            frames=prior.prior.SCOPES[args.subject],static_views=list(plan['static_views']),
            angular_source_rows=len(support['sources']),all_fit_w2c_identity=support['all_fit_w2c_identity'],
            physical_angular_coverage=support['physical_angular_coverage'],static_direction_proxy=support['static_views'],
            static_min_depth_m={name:item['min_center_depth_m'] for name,item in plan['static_arms'].items()})),flush=True)
        return
    output = Path(plan['output'])
    with review.GPU_LEASE.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        output.mkdir(exist_ok=False)
        path = output/'plan.json'
        shutil.copyfile(rois,output/'rois.npz')
        pin(output/'rois.npz',plan['roi_archive_sha256'])
        review.save(path,plan)
        with (output/'review.log').open('x',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-I','-B',__file__,'_worker',str(path),str(lease.fileno())],
                pass_fds=(lease.fileno(),),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=420)
    new,regions,static = (read(output/file) for file in ('new-result.json','new-regions.json','static-arms.json'))
    panels = assemble(output,plan,baseline,new,regions,static)
    result = output/'result.json'
    review.save(result,dict(state='round4_delivery_views_complete',subject=args.subject,plan_sha256=sha(path),
        panels=panels,regions=plan['regions'],region_metrics=dict(baseline['region_metrics'],round4=regions),
        source_angular_support=plan['source_angular_support'],native_view_labels=plan['native_view_labels'],
        static_render_receipts=static,old=baseline['old'],round3=baseline['new'],new=new,optimizer_updates=0,refits=0,
        dolly_rgb_truth_available=False,static_rotated_oblique_rgb_truth_available=False,
        static_rotated_side_rgb_truth_available=False,quality_accepted=False))
    print(json.dumps(dict(result=str(result),sha256=sha(result),panels=len(panels))),flush=True)


if __name__ == '__main__':
    main()
