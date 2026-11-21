"""Review the two public cold-run terminals against the delivered R9 models."""

import argparse
import fcntl
from itertools import permutations
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_smpl_delivery_round9 as r9
import measure_smpl_delivery_round5_ear016 as ear016
from review_bound_alpha_ab import mask_metrics, native_box
from review_bound_scale_coverage import face_alpha, scale_pixels


review, round1 = r9.review, r9.round1
read, require, sha = r9.read, r9.require, r9.sha
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-joint-fit-20261111')
DELIVERY = Path(__file__).resolve().parents[1] / 'outputs/smpl-delivery-round9'
BASE_COMMIT = '7602f31d16dc13ba68c27ca63895d72a07d3a1dd'
BASE_SHA = '0a489c425b8ed9ea6f32f390e953330325c3123cf72130a5d4994ac1a90d805c'
R9_REVIEWER_SHA = '55994b289c1f86f642819d21986f97b475ae214c1f7fb1a16d3c52633223e5be'
R9_HOTSPOT_SHA = '8374f40ced51425b1bdb264c3106f282cd539ac1b6065ddeb1fd82822bb7808a'
SUBJECTS = {'016': ('frame_000027', 'frame_000051'), '006': ('frame_000033', 'frame_000044')}
SPECIAL_016 = ('frame_000051', 'frame_000055')
CAMERA_KEYS = ('K', 'K_integer_centers', 'w2c', 'wh')


def pin(path, digest):
    path = Path(path)
    r9.pin(path,digest)
    return path


def root_for(subject):
    return review.LAB / f'shuolee-smpl-delivery-round10-{subject}'


def verify_base(subject):
    pin(Path(r9.__file__), R9_REVIEWER_SHA)
    receipt = read(pin(DELIVERY / 'receipt.json', BASE_SHA))
    spec = receipt['subjects'][subject]
    require(receipt['commit'] == spec['commit'] == BASE_COMMIT, 'Round-nine receipt identity changed.')
    path = pin(Path(spec['review']), spec['review_sha256'])
    baseline = read(path)
    locked = read(pin(path.with_name('plan.json'), baseline['plan_sha256']))
    require(baseline['state'] == 'round9_delivery_views_complete' and locked['new_commit'] == BASE_COMMIT
            and locked['round9_reviewer_sha256'] == R9_REVIEWER_SHA, 'Round-nine reference views differ.')
    run = Path(spec['result']).parent
    old = review.planned_model(run, r9.PROJECT, spec['result_sha256'])
    require(old.provenance == locked['new'], 'The locked R9 views belong to a different terminal.')
    return old, locked, path


def completed_training(subject, commit):
    require(re.fullmatch('[0-9a-f]{40}', commit) is not None, 'Pin the exact R10 candidate.')
    r9.transfer.rebound(r9.transfer.base.common.git_state, PROJECT=PROJECT)(commit)
    root = root_for(subject)
    run = root / 'run'
    wrapper_path = root / 'wrapper-terminal.json'
    wrapper = read(wrapper_path)
    terminal_path = pin(run / 'result.json', wrapper['result_sha256'])
    terminal = read(terminal_path)
    require(wrapper['state'] == terminal['state'] == 'completed' and wrapper['candidate_commit'] == commit
            and terminal['from_scratch'] is terminal['format_passed'] is True
            and terminal['steps'] == 10000 and terminal['final_snapshot'] == 'step-010000',
            'R10 is missing the finished cold video run and static delivery.')
    integration = run / 'gaussian-run/integration.json'
    fit = read(run / 'fit/official-fit-result.json')
    joint = pin(run/'fit/joint-body-refinement.json',wrapper['joint_refinement_sha256'])
    candidates = pin(run/'fit/joint-candidates.npz',wrapper['joint_candidate_sha256'])
    refinements = read(joint)
    require(fit['fitted_candidate_frames'] == 52
            and read(integration)['optimizer_updates'] == 10000
            and refinements['candidate_sha256'] == wrapper['joint_candidate_sha256']
            and refinements['shared_betas_frozen'] is refinements['native_K_frozen'] is True
            and refinements['per_frame_scale_parameters'] == 0,
            'R10 fit or Gaussian step count differs from the registered budget.')
    return terminal, wrapper, dict(
        wrapper=dict(file=str(wrapper_path), sha256=sha(wrapper_path)),
        integration=dict(file=str(integration), sha256=sha(integration)),
        joint=dict(file=str(joint), sha256=sha(joint)),
        joint_candidates=dict(file=str(candidates), sha256=sha(candidates)))


def same_view(a, b):
    return all(a[key] == b[key] for key in (*review.NATIVE_KEYS, 'K_integer_centers'))


def selected_pairs(old, new, subject, locked):
    common = {frame for frame in old.rows.keys() & new.rows.keys() if same_view(old.rows[frame], new.rows[frame])}
    unmatched = sorted((old.rows.keys() & new.rows.keys()) - common)
    preferred = dict(zip(('front','side'),SUBJECTS[subject],strict=True))

    def distance(role,frame):
        return abs(old.rows[frame]['source_frame_index']-old.rows[preferred[role]]['source_frame_index'])

    roles = {}
    if len(common) >= 2:
        front,side = min(permutations(sorted(common),2),key=lambda pair:(
            -sum(frame == expected for frame,expected in zip(pair,preferred.values(),strict=True)),
            sum(distance(role,frame) for role,frame in zip(preferred,pair,strict=True)),pair))
        roles.update(front=front,side=side)
    elif common:
        frame = next(iter(common))
        roles[min(preferred,key=lambda role:distance(role,frame))] = frame
    if subject == '016':
        roles.update((name.removeprefix('frame_'), name) for name in SPECIAL_016 if name in common - set(roles.values()))
    pairs = []
    for role, name in roles.items():
        row = old.rows[name]
        raw = pin(Path(row['image']), row['source_sha256'])
        target = pin(Path(row['mask']), row['mask_sha256'])
        fresh = new.rows[name]
        fresh_mask = pin(Path(fresh['mask']),fresh['mask_sha256'])
        a, b, c, d = row['initial_box_xyxy']
        body_crop = native_box((a-64, b-64, c+64, d+64), row['wh'])
        exact = next((spec for spec in locked['views'].values()
                      if spec['native_rgb_truth'] and spec['frame'] == name), None)
        face = row.get('face_box_xyxy')
        head_crop = None
        if exact:
            head_crop = exact['crop_xyxy']
        elif face:
            x0, y0, x1, y1 = face
            head_crop = native_box((x0-24, y0-24, x1+24, y1+24), row['wh'])
        pairs.append(dict(role=role, frame=name, source_frame_index=row['source_frame_index'],
            source=str(raw), source_sha256=row['source_sha256'],
            score_mask=str(target), score_mask_sha256=row['mask_sha256'],
            fresh_score_mask=str(fresh_mask), fresh_score_mask_sha256=fresh['mask_sha256'],
            camera={key:row[key] for key in CAMERA_KEYS}, crop_xyxy=body_crop, head_crop_xyxy=head_crop,
            old_slot=old.slots[name], new_slot=new.slots[name],
            old_pose_sha256=review.pose_digest(old,name), new_pose_sha256=review.pose_digest(new,name),
            preferred_frame=preferred.get(role,name)))
    return pairs, sorted(common), unmatched


def baseline_world_camera(camera, transform):
    """Undo only the portable coordinate export, leaving the camera itself fixed."""
    E = np.asarray(camera['w2c'], dtype=float)
    Q = np.asarray(transform['rotation'], dtype=float)
    origin = np.asarray(transform['origin_world'], dtype=float)
    require(transform['scale'] == 1.0, 'Expected rigid metre-based portable exports.')
    restored = np.eye(4)
    restored[:3,:3] = E[:3,:3] @ Q
    restored[:3,3] = E[:3,3] - restored[:3,:3] @ origin
    return dict(camera, w2c=restored.tolist())


def static_plan(old, new, locked, old_parts, new_parts):
    from review_smpl_delivery_round3 import portable, exported_view

    arms = {}
    for side, model, parts in (('old',old,old_parts), ('new',new,new_parts)):
        reference = model.provenance['reference_frame_id']
        arm = portable(model.provenance, model.rows[reference], parts)
        arm['reference_frame'] = reference
        points = PlyData.read(arm['ply'])['vertex'].data
        arm['_xyz'] = np.stack([points[axis] for axis in ('x','y','z')], axis=1)
        arms[side] = arm
    base = arms['old']['world_to_viewer']
    views = {}
    for name, item in locked['static_views'].items():
        views[name] = dict(item, source_rgb_available=False, fixed_world_camera=baseline_world_camera(item['camera'], base))
    dolly = locked['views']['dolly']
    near = {key:dolly['camera'][key] for key in ('K','w2c','wh','frame')}
    views['dolly'] = dict(camera=exported_view(near,base), crop_xyxy=dolly['crop_xyxy'],
                          synthetic=True, source_rgb_available=False, fixed_world_camera=near)
    for name, item in views.items():
        item['arms'] = {}
        camera = item['fixed_world_camera']
        movement = 1.
        for fraction in (1., .75, .5):
            if fraction != 1. and name != 'dolly':
                break
            world = np.asarray(camera['w2c']).copy()
            if name == 'dolly' and fraction != 1.:
                front = np.asarray(views['static_reference_front']['fixed_world_camera']['w2c'])
                world[:3,3] = front[:3,3]+fraction*(world[:3,3]-front[:3,3])
            probe = dict(camera, w2c=world.tolist())
            for side, arm in arms.items():
                # Preserve the pinned R9 camera numerically when the locked view is in use.
                converted = item['camera'] if side == 'old' and fraction == 1. else exported_view(probe,arm['world_to_viewer'])
                np.testing.assert_allclose(baseline_world_camera(converted,arm['world_to_viewer'])['w2c'],
                    world, rtol=0, atol=1e-10)
                E = np.asarray(converted['w2c'])
                depth = float((arm['_xyz'] @ E[2,:3]+E[2,3]).min())
                item['arms'][side] = dict(camera=converted, min_center_depth_m=depth)
            if all(entry['min_center_depth_m'] > .2 for entry in item['arms'].values()):
                item['fixed_world_camera'] = probe
                movement = fraction
                break
        require(all(entry['min_center_depth_m'] > .2 for entry in item['arms'].values()), 'Matched camera crosses the raster cutoff.')
        item['dolly_movement_fraction'] = movement if name == 'dolly' else None
    for arm in arms.values():
        del arm['_xyz']
    return views, arms


def special_identity(run, model, old, subject, proof):
    if subject != '016':
        return {}
    candidate = proof['joint_candidates']
    candidates = pin(candidate['file'],candidate['sha256'])
    raw_manifest = run / 'data/inputs.json'
    rows = {row['frame']:row for row in read(raw_manifest)['records']}
    with np.load(candidates,allow_pickle=False) as npz:
        fit_ids = npz['frame_ids'].tolist()
    require(len(fit_ids) == len(set(fit_ids)) == 52, 'Joint candidate roster is incomplete.')
    require(set(model.provenance['protected_holdouts']).isdisjoint(fit_ids), 'A protected pose entered joint candidate fitting.')
    output = {}
    for frame in SPECIAL_016:
        row = rows[frame]
        base = old.rows[frame]
        require(frame not in model.provenance['protected_holdouts'] and same_view(base,row),
                'The f051/f055 native source or camera changed.')
        output[frame] = dict(source_frame_index=base['source_frame_index'], source_sha256=base['source_sha256'],
            R9='trained', R10='trained' if frame in model.rows else 'fit_only' if frame in fit_ids else 'source_only',
            old_mask_sha256=base['mask_sha256'], new_mask_sha256=row['mask_sha256'],
            source=row['image'], wh=row['wh'], K_integer_centers=row['K_integer_centers'])
    return output


def preflight(subject, commit=None):
    old, locked, previous = verify_base(subject)
    if not commit:
        return dict(subject=subject, baseline_review=str(previous), baseline_reference=old.provenance['reference_frame_id'],
                    baseline_selected=len(old.rows), real_R9_f051_f055_available=subject == '016')
    terminal, wrapper, proof = completed_training(subject,commit)
    root, run = root_for(subject), root_for(subject)/'run'
    new = review.planned_model(run, PROJECT, wrapper['result_sha256'])
    selection = {}
    placement = {}
    for side, model in (('old',old),('new',new)):
        model_run = Path(model.provenance['run'])
        selection[side] = read(model_run / 'fit/selection.json')
        require(set(selection[side]['selected_frames']) == set(model.rows) and len(model.rows) == 24,
                'The reported selection does not bind 24 trained source frames.')
        descriptors = read(model_run/'fit/official-fit-result.json')['official_frame_results']
        require({entry['frame_id'] for entry in descriptors} == set(model.rows),
                'Initial official receipts do not match the delivered fit roster.')
        fields = [read(pin(entry['path'],entry['sha256'])).get('body_placement_parameters')
                  for entry in descriptors]
        placement[side] = dict(reported_counts=sorted({value for value in fields if value is not None}),
                               receipt_count=len(fields),without_field=fields.count(None))
    require(old.provenance['protected_holdouts'] == new.provenance['protected_holdouts'],
            'The protected holdout identities changed.')
    before = run / 'fit/selection-before-joint.json'
    selection['pre_joint'] = read(before)
    require(selection['pre_joint']['retained_holdouts'] == selection['new']['retained_holdouts'],
            'The pre-joint selection used a different protected holdout roster.')
    old_parts = round1.delivery_paths(Path(old.provenance['run']),read(Path(old.provenance['run'])/'result.json'))
    new_parts = round1.delivery_paths(run,terminal)
    round1.check_source_equivalence(r9.PROJECT,PROJECT,old_parts[2]['source_files_sha256'],
                                   new_parts[2]['source_files_sha256'])
    for model in (old,new):
        for row in model.rows.values():
            base = next(iter(old.rows.values()))
            require(all(base[key] == row[key] for key in CAMERA_KEYS), 'The fixed native camera contract changed.')
    pairs, common, unmatched = selected_pairs(old,new,subject,locked)
    views, arms = static_plan(old,new,locked,old_parts,new_parts)
    proof['pre_joint_selection'] = dict(file=str(before),sha256=sha(before))
    special = special_identity(run,new,old,subject,proof)
    return dict(schema='shuolee-round10-native-pose-and-static-review-v1',subject=subject,
        old=old.provenance,new=new.provenance,baseline_review=str(previous),baseline_review_sha256=sha(previous),
        commit=commit,output=str(root/'view-comparison'),asset=str(review.ASSET),reviewer_sha256=sha(Path(__file__)),
        pairs=pairs,matched_frames=common,unmatched_native_identities=unmatched,static_views=views,static_arms=arms,
        selections=selection,special_016=special,proof=proof,
        official_initializer_body_placement=placement,
        shared_betas_l2_change=float(np.linalg.norm(new.fit['shared_betas']-old.fit['shared_betas'])),
        fitted_pose_sha_differs=old.provenance['fitted_sha256'] != new.provenance['fitted_sha256'],
        fixed_native_camera=True,physical_angular_coverage='unknown',
        static_RGB_truth_available=False,optimizer_updates=0,refits=0,quality_accepted=False)


def image_tile(values, crop):
    x0,y0,x1,y1 = crop
    pixels = values[y0:y1,x0:x1]
    if pixels.dtype != np.uint8:
        pixels = np.rint(np.clip(pixels,0,1)*255).astype(np.uint8)
    return Image.fromarray(pixels).convert('RGB')


def strip(path, crop, tiles):
    x0,y0,x1,y1 = crop
    w,h = x1-x0,y1-y0
    require(w > 0 and h > 0, 'Review crop is empty.')
    image = Image.new('RGB',(len(tiles)*w,h))
    for col,item in enumerate(tiles):
        tile = item if isinstance(item,Image.Image) else image_tile(item,crop)
        require(tile.size == (w,h), 'A matched review tile changed pixel dimensions.')
        image.paste(tile,(col*w,0))
    image.save(path)
    return dict(file=path.name, sha256=sha(path), crop_xyxy=list(crop))


def silhouette_metrics(vertices,faces,K,matte):
    import cv2

    projected = vertices@K.T
    require(np.isfinite(projected).all() and (projected[:,2] > .05).all(),
            'The actual fitted mesh must be projectable in the native camera.')
    triangles = np.rint(projected[:,:2]/projected[:,2:])[faces].astype(np.int32)
    silhouette = np.zeros(matte.shape,np.uint8)
    for triangle in triangles:
        cv2.fillConvexPoly(silhouette,triangle,1)
    fg = matte > .99
    covered = silhouette.astype(bool)
    return dict(iou=float((fg&covered).sum()/(fg|covered).sum()),
                uncovered_reliable_fg_pixels=int((fg&~covered).sum()),
                reliable_exterior_mesh_pixels=int(((matte < .01)&covered).sum()))


def render_worker(plan_path, side, lease_fd):
    require(os.path.samestat(os.fstat(lease_fd),review.GPU_LEASE.stat()), 'Worker did not inherit the exclusive GPU lease.')
    fcntl.flock(lease_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan = read(plan_path)
    pin(Path(__file__),plan['reviewer_sha256'])
    model_info = plan[side]
    out, root = plan_path.parent, Path(model_info['run'])
    pin(root/'result.json',model_info['terminal_sha256'])
    sys.path.insert(0,str(Path(model_info['project'])/'src'))
    import torch
    from shuolee_portrait import runtime as rt
    from shuolee_portrait.bound_replay import _load_model
    from shuolee_portrait.bound_review_panels import _features, _mesh_panel
    from shuolee_portrait.cli import gpu_lock
    from shuolee_portrait.training import camera, render_pose

    asset = Path(plan['asset'])
    rt.configure(asset.parents[1])
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel

    rows = read(pin(root/'data/training-inputs.json',model_info['training_inputs_sha256']))['records']
    saved = round1.delivered(root,asset,rows,model_info['protected_holdouts'])
    mesh_path = saved.snapshot/'posed-replay'/saved.replay['mesh_file']
    pin(mesh_path,saved.replay['mesh_sha256'])
    with np.load(mesh_path,allow_pickle=False) as archive:
        vertices,faces = archive['vertices_world'],archive['faces']
    source_by_name = {row['frame']:row for row in rows}
    native,static,panels = [],{},{}
    with gpu_lock(),torch.no_grad():
        require(torch.cuda.device_count() == 1, 'Expose exactly one CUDA device.')
        black,white = torch.zeros(3,device='cuda'),torch.ones(3,device='cuda')
        pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)

        def raster(model,row,slot):
            cam, actual = camera(row,0,pose_index=slot)
            pixels,error = scale_pixels(render,render_pose,cam,model,pipe,black,white,slot,1.)
            return pixels,error,actual

        model,parity = _load_model(saved.snapshot/'bound-local.ply',saved.snapshot,saved.fit,saved.mesh,saved.facts,asset)
        for item in plan['pairs']:
            frame,role,slot = item['frame'],item['role'],item[f'{side}_slot']
            row = source_by_name[frame]
            source = pin(Path(item['source']),item['source_sha256'])
            target = pin(Path(item['score_mask']),item['score_mask_sha256'])
            fresh_target = pin(Path(item['fresh_score_mask']),item['fresh_score_mask_sha256'])
            own = pin(Path(row['image']),row['source_sha256'])
            require(all(row[key] == value for key,value in item['camera'].items()) and sha(own) == sha(source),
                    'The reviewer must use the same native camera and source RGB.')
            with Image.open(source) as rgb,Image.open(target) as mask:
                source_rgb = np.array(rgb.convert('RGB'))
                matte = np.asarray(mask.convert('L'),dtype=np.float32)/255.
            with Image.open(fresh_target) as mask:
                fresh_matte = np.asarray(mask.convert('L'),dtype=np.float32)/255.
            require(matte.shape == fresh_matte.shape == source_rgb.shape[:2], 'A source matte is on a different grid.')
            old_fg,fresh_fg = matte >= .5,fresh_matte >= .5
            divergence = dict(
                disagreement_pixels=int(np.count_nonzero(old_fg!=fresh_fg)),
                mask_iou=float((old_fg&fresh_fg).sum()/(old_fg|fresh_fg).sum()))
            pixels,error,actual = raster(model,row,slot)
            archived = next(record for record in saved.replay['records'] if record['frame'] == frame)
            require(archived['pose_index'] == slot and all(archived['camera'][key] == actual[key] for key in ('K','w2c','wh')),
                    'The replay camera or bound pose slot changed.')
            with Image.open(pin(saved.snapshot/'posed-replay'/archived['image'],archived['image_sha256'])) as image:
                ref = np.asarray(image.convert('RGB'),dtype=np.int16)
            delta = int(np.abs(np.rint(np.clip(pixels[0],0,1)*255).astype(np.int16)-ref).max())
            require(delta <= 1,'Native render did not reproduce the actual delivered replay.')
            no_extra_roi = np.zeros(matte.shape,dtype=bool)
            stats = mask_metrics(pixels[2],matte,no_extra_roi)
            fresh_stats = mask_metrics(pixels[2],fresh_matte,no_extra_roi)
            regions = {}
            masks = Path(out/'rois.npz')
            with np.load(pin(masks,plan['roi_sha256']),allow_pickle=False) as archive:
                score_keypoints = archive[f'{role}_score_keypoints']
                for name in plan['native_regions'][role]:
                    region = archive[f'{role}_{name}']
                    require(r9.native.hash_mask(region) == plan['native_regions'][role][name], 'Native ROI changed.')
                    black_rgb = pixels[0][region]
                    alpha = pixels[2][region]
                    quantized = np.rint(np.clip(alpha,0,1)*255)
                    result = dict(alpha=face_alpha(pixels[2],region), black_rgb_mae=float(
                        np.abs(np.clip(black_rgb,0,1)-source_rgb[region].astype(np.float32)/255).mean()),
                        alpha_u8_lt_166=int((quantized<166).sum()),alpha_u8_lt_243=int((quantized<243).sum()))
                    result['fresh_mask_reliable_pixels'] = int((fresh_matte[region]>.99).sum())
                    if name == 'ear_016_f051':
                        y,x = np.nonzero(region)
                        owners = ear016.mesh_coverage(vertices[slot],faces,actual,np.column_stack((x,y)))
                        result['posed_mesh_support_pixels'] = int((owners >= 0).sum())
                    regions[name] = result
            crops = {role:item['crop_xyxy']}
            if item['head_crop_xyxy']:
                crops[f'{role}_head'] = item['head_crop_xyxy']
            for label,crop in crops.items():
                panels[label] = dict(strip(out/f'{label}-{side}.png',crop,(source_rgb,*pixels)),frame=frame,
                    columns=['source_rgb','black','white','alpha'])
            crop = item['crop_xyxy']
            original = image_tile(source_rgb,crop)
            E,K = (np.asarray(row[key]) for key in ('w2c','K_integer_centers'))
            world = vertices[slot]
            camera_vertices = world@E[:3,:3].T+E[:3,3]
            mesh = _mesh_panel(original,crop,camera_vertices,faces,K)
            observed = saved.fit['observed_native_keypoints'][slot]
            projected = saved.fit['joints_projected_native'][slot]
            joints = _features(original,crop,observed,projected)
            label = f'{role}_geometry'
            panels[label] = dict(strip(out/f'{label}-{side}.png',crop,(original,joints,mesh,pixels[0])),frame=frame,
                columns=['source_rgb','keypoints','mesh','same_frame_gaussian'])
            valid = score_keypoints[:,2] >= .5
            residual = np.linalg.norm(projected-score_keypoints[:,:2],axis=1)
            native.append(dict(role=role,frame=frame,native_camera=actual, pose_sha256=item[f'{side}_pose_sha256'],
                source_frame_index=item['source_frame_index'],source_sha256=item['source_sha256'],
                score_mask_sha256=item['score_mask_sha256'],own_mask_sha256=row['mask_sha256'],
                fresh_score_mask_sha256=item['fresh_score_mask_sha256'],
                fresh_mask_metrics=fresh_stats,mask_divergence=divergence,
                native_archive_max_abs_8bit=delta,composition_max_abs=error,mask_metrics=stats,region_metrics=regions,
                posed_mesh_silhouette=silhouette_metrics(camera_vertices,faces,K,matte),
                reliable_keypoint_count=int(valid.sum()),
                reliable_keypoint_rms_px=float(np.sqrt(np.mean(residual[valid]**2))) if valid.any() else None))
        del model
        arm = plan['static_arms'][side]
        pin(Path(arm['ply']),arm['ply_sha256'])
        static_model = GaussianModel(0)
        static_model.load_ply(arm['ply'])
        require(static_model.binding is None and static_model.optimizer is None
                and static_model.active_sh_degree == 0 and len(static_model.get_xyz) == arm['points'],
                'Expected the unbound, actual delivered static PLY.')
        for name,view in plan['static_views'].items():
            pixels,error,actual = raster(static_model,view['arms'][side]['camera'],0)
            panel = dict(strip(out/f'{name}-{side}.png',view['crop_xyxy'],pixels),
                         frame=arm['reference_frame'],columns=['black','white','alpha'])
            x0,y0,x1,y1 = view['crop_xyxy']
            covered = pixels[2] >= .5
            outside = int(covered.sum()-covered[y0:y1,x0:x1].sum())
            static[name] = dict(camera=actual, fixed_world_camera=view['fixed_world_camera'],panel=panel,
                actual_reference_frame=arm['reference_frame'],static_ply_sha256=arm['ply_sha256'],
                pose_driven=False,source_rgb_compared=False,composition_max_abs=error,
                alpha_at_least_half_pixels_outside_crop=outside)
            panels[name] = panel
    review.save(out/f'{side}-result.json',dict(native=native,static=static,panels=panels,model_parity=parity))


def setup_regions(plan, out):
    native = {}
    arrays = {}
    run = Path(plan['old']['run'])
    rows = {row['frame']:row for row in read(run/'data/training-inputs.json')['records']}
    with np.load(pin(run/'fit/fitted.npz',plan['old']['fitted_sha256']),allow_pickle=False) as archive:
        fit = {key:archive[key] for key in ('joints_camera','joints_projected_native','observed_native_keypoints')}
    for item in plan['pairs']:
        frame,role = item['frame'],item['role']
        row = rows[frame]
        with Image.open(pin(Path(item['score_mask']),item['score_mask_sha256'])) as image:
            matte = np.asarray(image.convert('L')) >= 253
        crop = item['head_crop_xyxy'] or item['crop_xyxy']
        masks,_ = r9.native.source_regions(fit,item['old_slot'],row,row,matte,crop)
        if plan['subject'] == '016' and frame == ear016.FRAME:
            audit = read(pin(ear016.ROUND4/'ear-016-source-audit.json',ear016.EAR_SHA))
            coords = audit['roi']
            require(item['source_sha256'] == audit['identity']['source_rgb']['sha256']
                    and item['score_mask_sha256'] == audit['identity']['score_mask']['sha256'], 'The fixed ear source or matte changed.')
            raster = Image.new('1',tuple(coords['wh']))
            ImageDraw.Draw(raster).polygon([tuple(point) for point in coords['polygon_xy_native_integer_centers_inclusive']],fill=1)
            mask = np.asarray(raster)
            require(int(mask.sum()) == 574 and r9.native.hash_mask(mask) == coords['roi_sha256'] and matte[mask].all(),
                    'The pinned f051 ear polygon changed.')
            masks['ear_016_f051'] = mask
        native[role] = {name:r9.native.hash_mask(mask) for name,mask in masks.items()}
        arrays.update({f'{role}_{name}':mask for name,mask in masks.items()})
        arrays[f'{role}_score_keypoints'] = fit['observed_native_keypoints'][item['old_slot']]
    path = out/'rois.npz'
    np.savez_compressed(path,**arrays)
    plan.update(native_regions=native,roi_sha256=sha(path))


def merge(out,plan,old,new):
    require(old['panels'].keys() == new['panels'].keys(), 'The review panels did not match across endpoints.')
    panels = []
    for label,a in old['panels'].items():
        b = new['panels'][label]
        require(a['crop_xyxy'] == b['crop_xyxy'] and a['columns'] == b['columns'], 'Matched pixel grids differ.')
        with Image.open(pin(out/a['file'],a['sha256'])) as left,Image.open(pin(out/b['file'],b['sha256'])) as right:
            require(left.size == right.size,'Matched panel dimensions differ.')
            sheet = Image.new('RGB',(left.width,left.height*2))
            sheet.paste(left,(0,0)); sheet.paste(right,(0,left.height))
            target = out/f'{label}-{a["frame"]}.png'
            sheet.save(target)
        panels.append(dict(view=label,file=target.name,sha256=sha(target),frame=a['frame'],new_frame=b['frame'],
            crop_xyxy=a['crop_xyxy'],columns=a['columns'],rows=['round9_7602f31',f'round10_{plan["commit"][:7]}']))
    return panels


def complete_016(out,plan):
    """Retain the exact f051/f055 record even when a fit is screened from training."""
    if not plan['special_016']:
        return []
    covered = {pair['frame'] for pair in plan['pairs']}
    old_root = Path(plan['old']['run'])
    old_terminal = read(pin(old_root/'result.json',plan['old']['terminal_sha256']))
    old_panels = pin(old_root/old_terminal['bound_panels'],old_terminal['bound_panels_sha256'])
    pin_manifest = read(old_panels)
    by_frame = {item['frame']:item for item in pin_manifest['panels']}
    candidate_record = plan['proof'].get('joint_candidates')
    candidates = {}
    if candidate_record:
        with np.load(pin(candidate_record['file'],candidate_record['sha256']),allow_pickle=False) as f:
            candidates = dict(f)
    if any(item['R10'] == 'fit_only' for item in plan['special_016'].values()):
        sys.path.insert(0,str(Path(plan['new']['project'])/'src'))
        from shuolee_portrait.bound_review_panels import _features, _mesh_panel
    panels = []
    for frame,identity in plan['special_016'].items():
        if frame in covered:
            continue
        item = by_frame[frame]
        require(item['source_frame_index'] == identity['source_frame_index']
                and item['source_sha256'] == identity['source_sha256'], 'R9 special-frame panel identity changed.')
        crop = item['crop_xyxy']
        x0,y0,x1,y1 = crop
        width,height = x1-x0,y1-y0
        with Image.open(pin(old_panels.parent/item['panel'],item['panel_sha256'])) as panel:
            require(panel.width == 4*width and panel.height == height+26, 'Unexpected R9 native geometry panel.')
            # The production panel order is original, Gaussian, mesh and keypoints.
            base_tiles = [panel.crop((i*width,26,(i+1)*width,height+26)) for i in (0,3,2,1)]
        with Image.open(pin(identity['source'],identity['source_sha256'])) as raw:
            source = raw.convert('RGB').crop(crop)
        require(np.array_equal(np.asarray(source),np.asarray(base_tiles[0])), 'Special-frame native pixels differ.')
        missing = Image.new('RGB',source.size,'#202020')
        draw = ImageDraw.Draw(missing)
        draw.text((12,12),'No matching R10 trained pose',fill='white')
        joints, mesh = missing,missing
        if identity['R10'] == 'fit_only':
            index = candidates['frame_ids'].tolist().index(frame)
            require(int(candidates['source_frames'][index]) == identity['source_frame_index'],
                    'Joint candidate source index changed.')
            for field, key in (('image_sha256','source_sha256'),('mask_sha256','new_mask_sha256')):
                if field in candidates:
                    require(candidates[field][index].item() == identity[key], 'Joint candidate source or mask hash changed.')
            projected = candidates['joints_projected_native'][index].reshape(-1,2)
            observed = candidates['observed_native_keypoints'][index].reshape(-1,3)
            vertices = candidates['vertices_camera'][index].reshape(-1,3)
            joints = _features(source,crop,observed,projected)
            mesh = _mesh_panel(source,crop,vertices,candidates['faces'],np.asarray(identity['K_integer_centers']))
            valid = observed[:,2] >= .5
            error = np.linalg.norm(projected-observed[:,:2],axis=1)
            identity['unselected_reliable_keypoint_rms_px'] = float(np.sqrt(np.mean(error[valid]**2))) if valid.any() else None
        sheet = Image.new('RGB',(4*width,2*height))
        for row,tiles in enumerate((base_tiles,(source,joints,mesh,missing))):
            for index,tile in enumerate(tiles):
                sheet.paste(tile,(width*index,height*row))
        target = out/f'available-record-{frame}.png'
        sheet.save(target)
        panels.append(dict(view='available_record',file=target.name,sha256=sha(target),frame=frame,new_frame=frame,
            crop_xyxy=crop,columns=['source_rgb','keypoints','mesh','same_frame_gaussian'],
            rows=['round9_7602f31',f'round10_{plan["commit"][:7]}'],new_gaussian_available=False))
    return panels


def missing_bin_evidence(out,plan):
    """Show one named source frame per newly covered or still-empty old/new interval."""
    sys.path.insert(0,str(Path(plan['new']['project'])/'src'))
    from shuolee_portrait.bound_review_panels import _crop, _features, _mesh_panel

    selections = plan['selections']
    missing = {side:set(selections[side]['coverage']['front']['missing_source_bins']) for side in ('old','new')}
    bins = sorted(missing['old'] | missing['new'])
    if not bins:
        return None,[]
    run = Path(plan['new']['run'])
    raw_manifest = run/'data/inputs.json'
    records = {row['frame']:row for row in read(raw_manifest)['records']}
    candidate = plan['proof']['joint_candidates']
    with np.load(pin(candidate['file'],candidate['sha256']),allow_pickle=False) as f:
        arrays = {key:f[key] for key in ('frame_ids','source_frames','image_sha256','mask_sha256','faces',
            'K_integer_centers','vertices_camera','observed_native_keypoints','joints_projected_native')}
    slots = {frame:index for index,frame in enumerate(arrays['frame_ids'].tolist())}
    evidence = []
    width,height = 265,270
    image = Image.new('RGB',(width*3,height*len(bins)),'#202020')
    draw = ImageDraw.Draw(image)
    for i,source_bin in enumerate(bins):
        group = sorted((frame for frame,reason in selections['new']['reasons'].items()
                        if reason.get('source_bin') == source_bin and frame in slots),
                       key=lambda frame:records[frame]['source_frame_index'])
        members = []
        for frame in group:
            item = {'frame':frame,'source_frame_index':records[frame]['source_frame_index']}
            for side in ('old','pre_joint','new'):
                reason = selections[side].get('reasons',{}).get(frame,{})
                item[side] = dict(status=reason.get('status'),screen_flags=reason.get('screen_flags'))
            members.append(item)
        selected = [frame for frame in group if frame in selections['new']['selected_frames']]
        frame = selected[0] if selected else group[len(group)//2] if group else None
        entry = dict(source_bin=source_bin,was_missing_R9=source_bin in missing['old'],
                     is_missing_R10=source_bin in missing['new'],frames=members,representative=frame)
        evidence.append(entry)
        if frame is None:
            draw.text((8,i*height+4),f'Bin {source_bin:02d} | no non-holdout fitted frame',fill='white')
            continue
        row,index = records[frame],slots[frame]
        require(arrays['source_frames'][index] == row['source_frame_index']
                and arrays['image_sha256'][index] == row['source_sha256']
                and arrays['mask_sha256'][index] == row['mask_sha256']
                and np.array_equal(arrays['K_integer_centers'][index],row['K_integer_centers']),
                'The candidate mesh or source pixels differ in the interval evidence.')
        source = pin(row['image'],row['source_sha256'])
        crop = _crop(row)
        with Image.open(source) as rgb:
            original = rgb.convert('RGB').crop(crop)
        tiles = (original,_features(original,crop,arrays['observed_native_keypoints'][index],
                                   arrays['joints_projected_native'][index]),
                 _mesh_panel(original,crop,arrays['vertices_camera'][index],arrays['faces'],arrays['K_integer_centers'][index]))
        description = 'R10 missing' if source_bin in missing['new'] else 'R10 covered'
        draw.text((8,i*height+3),f'Bin {source_bin:02d} | {frame} | source {row["source_frame_index"]} | {description}',fill='white')
        for column,(tile,label) in enumerate(zip(tiles,('Native RGB','Native keypoints','R10 fit mesh'),strict=True)):
            tile.thumbnail((width-10,height-48),Image.Resampling.LANCZOS)
            image.paste(tile,(column*width+(width-tile.width)//2,i*height+23))
            draw.text((column*width+8,(i+1)*height-20),label,fill='white')
    path = out/'missing-bins.png'
    image.save(path)
    frame = next((item['representative'] for item in evidence if item['representative']),plan['new']['reference_frame_id'])
    return dict(view='missing_bins',file=path.name,sha256=sha(path),frame=frame,new_frame=frame,
        rows=['interval evidence'],columns=['source_rgb','keypoints','mesh']),evidence


def static_hotspot(out,plan,old,new):
    hotspot = read(pin(DELIVERY/'hotspot-comparison.json',R9_HOTSPOT_SHA))['subjects'][plan['subject']]
    view = hotspot['view']
    crop = hotspot['crop_xyxy']
    x0,y0,x1,y1 = crop
    width,height = x1-x0,y1-y0
    mask = Image.new('1',(width,height))
    draw = ImageDraw.Draw(mask)
    for polygon in hotspot['hotspot_poly_crop_local']:
        draw.polygon([tuple(point) for point in polygon],fill=1)
    region = np.asarray(mask)
    require(int(region.sum()) == hotspot['metrics']['hotspot']['pixels'], 'Fixed static polygon changed.')
    metrics = {}
    for label,arm in (('R9',old),('R10',new)):
        item = arm['static'][view]
        panel = item['panel']
        require(panel['crop_xyxy'] == crop and item['fixed_world_camera'] == plan['static_views'][view]['fixed_world_camera'],
                'Fixed static polygon uses a different camera or crop.')
        with Image.open(pin(out/panel['file'],panel['sha256'])) as image:
            require(image.size == (3*width,height), 'Static color/alpha strip shape changed.')
            alpha = np.asarray(image.crop((2*width,0,3*width,height)).convert('L'))[region]
        metrics[label] = dict(alpha_mean=float(alpha.mean()/255.),alpha_ge_half_pixels=int((alpha>=128).sum()),
                              baked_pose=item['actual_reference_frame'])
    return dict(view=view,pixels=int(region.sum()),metrics=metrics,source_rgb_truth_available=False,
                same_baked_pose_name=metrics['R9']['baked_pose'] == metrics['R10']['baked_pose'])


def main():
    if len(sys.argv) == 5 and sys.argv[1] == '_worker':
        render_worker(Path(sys.argv[2]),sys.argv[3],int(sys.argv[4]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('check','render'))
    parser.add_argument('subject',choices=SUBJECTS)
    parser.add_argument('--commit')
    args = parser.parse_args()
    if args.action == 'check':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    elif not args.commit:
        parser.error('The real completed R10 commit is required to render.')
    plan = preflight(args.subject,args.commit)
    if args.action == 'check':
        import json
        print(json.dumps({key:value for key,value in plan.items() if key in
                         ('subject','baseline_reference','baseline_selected','matched_frames','real_R9_f051_f055_available')}))
        return
    out = Path(plan['output'])
    with review.GPU_LEASE.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        out.mkdir(exist_ok=False)
        setup_regions(plan,out)
        path = out/'plan.json'
        review.save(path,plan)
        for side in ('old','new'):
            with (out/f'{side}.log').open('x',encoding='utf-8') as log:
                subprocess.run([sys.executable,'-I','-B',__file__,'_worker',str(path),side,str(lease.fileno())],
                               pass_fds=(lease.fileno(),),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
    old,new = (read(out/f'{side}-result.json') for side in ('old','new'))
    panels = merge(out,plan,old,new)+complete_016(out,plan)
    missing_panel, missing_rows = missing_bin_evidence(out,plan)
    if missing_panel:
        panels.append(missing_panel)
    result = out/'result.json'
    review.save(result,dict(state='round10_delivery_views_complete',subject=args.subject,commit=args.commit,
        plan_sha256=sha(path),panels=panels,old=old,new=new,proof=plan['proof'],
        special_016=plan['special_016'],missing_bin_evidence=missing_rows,
        static_hotspot=static_hotspot(out,plan,old,new),quality_accepted=False))
    print(f'{result} {sha(result)}',flush=True)


if __name__ == '__main__':
    main()
