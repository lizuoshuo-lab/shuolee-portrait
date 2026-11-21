"""Render actual static PLY files at the R21 registered metric camera grid."""

import fcntl
from hashlib import sha256
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
from plyfile import PlyData

import review_smpl_delivery_round20 as R20
import review_round21_io as stage_io
from review_bound_scale_coverage import scale_pixels


read,pin,sha,require,descriptor = R20.read,R20.pin,R20.sha,R20.require,R20.descriptor
review,round1,r10,r13 = R20.review,R20.round1,R20.r10,R20.r13
ARM_NAMES = ('r15','r20','before10','r21_10','before18','r21')
PORTABLE = ('r15','r20','r21_10','r21')
PROXY_SHA = {'016':'4eeffd047d02193414cd4643e809d620b5ff83c046faa6a892f48fa26031c071',
             '006':'5b145f4e098464bc17edce2b42d105d779b914d2823a7381e1554ad72efaa53c'}


def static_model(info):
    run = Path(info['run'])
    terminal = read(pin(run/'result.json',info['terminal_sha256']))
    snapshot,_,bound = round1.delivery_paths(run,terminal)
    rows = read(pin(run/'data/training-inputs.json',info['training_inputs_sha256']))['records']
    original = next(row for row in rows if row['frame']==info['reference_frame_id'])
    arm = stage_io.portable(info,original,(snapshot,None,bound))
    local = pin(snapshot/'bound-local.ply',bound['bound_ply_sha256'])
    world = pin(snapshot/'world.ply',bound['inputs_sha256']['world_ply'])
    return dict(arm,project=info['project'],local=descriptor(local),world=descriptor(world),
                run=str(run),model_class='actual_delivered_static_portable_ply',ref=info['reference_frame_id'])


def prepare(plan, old, mid):
    import review_round21_camera_grid as cameras
    import review_round21_head_proxy as proxies

    subject = plan['subject']
    new_info = plan['new']
    run = Path(new_info['run'])
    path = proxies.camera_manifest(subject)
    staged = pin(run/'data/head-support-views.json',sha(path))
    fixed = read(path)
    proof_path = pin(path.with_name(f'head-support-proxy-{subject}.json'),PROXY_SHA[subject])
    proof = read(proof_path)
    require(proof['camera_manifest']==descriptor(path) and len(fixed['views'])==len(proof['views'])==50,
            'The registered camera or geometry proxy roster differs.')
    pins = {key:descriptor(item) for key,item in (('camera_manifest',path),('staged_camera_manifest',staged),
                                                   ('proxy_manifest',proof_path))}
    pins['proxy_masks'] = descriptor(pin(proof['mask_archive']['file'],proof['mask_archive']['sha256']))
    arm = dict(r15=static_model(old),r20=static_model(mid),r21_10=static_model(plan['ten']),r21=static_model(new_info))
    require(all(arm[name.lower()]['ply_sha256']==fixed['baseline'][name]['portable']['ply_sha256']
                and arm[name.lower()]['camera_sha256']==fixed['baseline'][name]['portable']['camera_sha256'] for name in ('R15','R20')),
            'An acceptance baseline is not its preregistered actual delivered PLY and camera.')
    for label,side,step in (('before10','ten','10000'),('before18','new','18000')):
        info = plan[side]
        branch = Path(info['run'])
        stage = plan['duration']['raw_snapshots'][step]['files']
        before = pin(stage['world.ply']['file'],stage['world.ply']['sha256'])
        pinned_mesh = pin(stage['reference-mesh.npz']['file'],stage['reference-mesh.npz']['sha256'])
        parity = read(pin(stage['native-ply-parity.json']['file'],stage['native-ply-parity.json']['sha256']))
        require(parity['passed'] and parity['sha256']==sha(before) and parity['mesh_sha256']==sha(pinned_mesh),
                'The same-run raw world snapshot did not pass native export parity.')
        arm[label] = dict(ply=str(before),ply_sha256=sha(before),project=info['project'],run=str(branch),
                         ref=info['reference_frame_id'],world_to_viewer=dict(rotation=np.eye(3).tolist(),origin_world=[0.,0.,0.],scale=1.),
                         model_class='same_run_unmodified_main_training_world_ply')
    require(all(value['ref']==fixed['reference_frame'] for value in arm.values()),
            'A static PLY had a different frozen reference pose.')
    prepared = {}
    for side,value in arm.items():
        points = PlyData.read(pin(value['ply'],value['ply_sha256']))['vertex'].data
        xyz = np.column_stack([points[axis] for axis in ('x','y','z')])
        transform = value['world_to_viewer']
        rotation = np.asarray(transform['rotation'])
        origin = np.asarray(transform['origin_world'])
        centers = xyz@rotation+origin
        per_view = {}
        for key,spec in fixed['views'].items():
            world = spec['world_camera']
            E = cameras.to_portable(world['w2c'],transform)
            portable = dict(world,w2c=E.tolist())
            if side in ('r15','r20'):
                expected = spec['baseline_centers'][side.upper()]
                observed = cameras.center_receipt(centers,world['K'],world['w2c'],world['wh'])
                require(all(observed[key]==expected[key] for key in expected if key!='min_center_depth_m')
                        and abs(observed['min_center_depth_m']-expected['min_center_depth_m'])<1e-10,
                        'Baseline Gaussian positions changed after camera registration.')
            else:
                observed = cameras.center_receipt(centers,world['K'],world['w2c'],world['wh'])
            restored = r10.baseline_world_camera(portable,transform)
            np.testing.assert_allclose(restored['w2c'],world['w2c'],rtol=0,atol=1e-10)
            per_view[key] = dict(camera=portable,centers=observed)
        prepared[side] = dict(value,points=len(xyz),views=per_view)
    return dict(schema='shuolee-r21-registered-static-ply-comparison/1',camera_roster=fixed['views'],
        proxies=proof['views'],arms=prepared,proof=pins,native_wh=fixed['native_wh'],
        synthetic_no_real_ground_truth=True)


def opacity_stats(alpha,mask):
    values = alpha[mask]
    if not len(values):
        return dict(pixels=0,low_alpha_095_fraction=None,mean_transmission=None,p05=None)
    return dict(pixels=len(values),low_alpha_095_fraction=float((values<.95).mean()),
                mean_transmission=float((1.-values).mean(dtype=np.float64)),p05=float(np.quantile(values,.05)))


def exterior_stats(alpha,band):
    values = alpha[band]
    positive = ((alpha>=.1)&band).astype(np.uint8)
    count,_,stats,_ = cv2.connectedComponentsWithStats(positive,connectivity=8)
    return dict(pixels=len(values),alpha_mass=float(values.sum(dtype=np.float64)),
        alpha_ge_010_pixels=int((values>=.1).sum()),alpha_ge_050_pixels=int((values>=.5).sum()),
        max_connected_alpha_ge_010_pixels=int(stats[1:,cv2.CC_STAT_AREA].max()) if count>1 else 0)


def head_binding(static, fixed, count):
    import torch

    if 'local' not in static:
        return None
    bound = PlyData.read(pin(static['local']['file'],static['local']['sha256']))['vertex'].data
    world = PlyData.read(pin(static['world']['file'],static['world']['sha256']))['vertex'].data
    actual = PlyData.read(pin(static['ply'],static['ply_sha256']))['vertex'].data
    require(len(bound)==len(world)==len(actual)==count and all(np.array_equal(bound[field],world[field])
                    and np.array_equal(world[field],actual[field]) for field in ('f_dc_0','f_dc_1','f_dc_2','opacity')),
            'Static PLY order does not match the exact bound rows for head-contribution diagnostics.')
    asset = fixed['proof']['smplx_asset']
    with np.load(pin(asset['file'],asset['sha256']),allow_pickle=False) as file:
        faces = file['f']
        weights = file['weights'][:,fixed['head_proxy_lbs_joint_ids']].sum(axis=1)
        head = (weights[faces]>=fixed['head_proxy_lbs_threshold']).all(axis=1)
    ids = bound['binding_0'].astype(np.int64)
    require(np.array_equal(ids,bound['binding_0']) and ((ids>=0)&(ids<len(faces))).all(),'Bad head-bound Gaussian row IDs.')
    mask = head[ids]
    require(mask.any(),'A portable PLY has no anatomical head-bound Gaussian rows.')
    return torch.as_tensor(mask,device='cuda',dtype=torch.float32)[:,None].expand(-1,3).contiguous()


def panel_image(out,key,view,pixels,silhouette):
    crop = view['normal_body_crop_xyxy'] or view['head_crop_xyxy']
    panels = [r10.image_tile(field,crop) for field in pixels]
    columns = ['black','white','alpha']
    if view['distance_class']=='close':
        outline = np.asarray(panels[2]).copy()
        mask = silhouette.astype(np.uint8)
        contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(outline,contours,-1,(38,233,247),1,lineType=cv2.LINE_8)
        panels.append(Image.fromarray(outline))
        columns.append('alpha_with_fixed_anatomical_head_outline')
    width,height = panels[0].size
    stitched = Image.new('RGB',(width*len(panels),height))
    for index,item in enumerate(panels):
        stitched.paste(item,(width*index,0))
    target = out/f'{key}.png'
    stitched.save(target,compress_level=1)
    return dict(file=str(target),sha256=sha(target),crop_xyxy=crop,columns=columns)


def render_grid(path,side,fd):
    plan = read(path)
    block = plan['static_grid']
    fixed = read(pin(block['proof']['camera_manifest']['file'],block['proof']['camera_manifest']['sha256']))
    static = block['arms'][side]
    require(os.path.samestat(os.fstat(fd),review.GPU_LEASE.stat()),'Missing inherited GPU lease.')
    fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sys.path.insert(0,str(Path(static['project'])/'src'))
    import torch
    from shuolee_portrait import runtime as rt
    from shuolee_portrait.cli import gpu_lock
    from shuolee_portrait.training import camera, render_pose

    asset = Path(plan['asset'])
    rt.configure(asset.parents[1])
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel

    directory = path.parent/'static-grid'/side
    directory.mkdir(parents=True,exist_ok=False)
    mask_path = pin(block['proof']['proxy_masks']['file'],block['proof']['proxy_masks']['sha256'])
    rows = {}
    with gpu_lock(),torch.no_grad(),np.load(mask_path,allow_pickle=False) as masks:
        require(torch.cuda.device_count()==1,'Static review must use one CUDA device.')
        model = GaussianModel(0)
        model.load_ply(str(pin(static['ply'],static['ply_sha256'])))
        require(model.optimizer is None and model.binding is None and model.active_sh_degree==0
                and len(model.get_xyz)==static['points'],'The review did not use the actual unbound SH0 PLY.')
        weights = head_binding(static,fixed,len(model.get_xyz))
        black,white = torch.zeros(3,device='cuda'),torch.ones(3,device='cuda')
        pipe = SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
        for idx,(key,view) in enumerate(block['camera_roster'].items()):
            camera_entry = static['views'][key]
            cam,record = camera(camera_entry['camera'],idx,pose_index=0)
            require(float(cam.znear)==view['projection_near_m'] and all(record[key]==camera_entry['camera'][key] for key in ('K','wh','w2c')),
                    'R21 static projector altered the registered fixed camera.')
            pixels,composition = scale_pixels(render,render_pose,cam,model,pipe,black,white,0,1.)
            x0,y0,x1,y1 = view['head_crop_xyxy']
            shape = (y1-y0,x1-x0)
            p = block['proxies'][key]
            binary = {}
            for name,meta in p['masks'].items():
                sample = masks[key+'__'+name]
                require(sample.shape==shape and sha256(np.ascontiguousarray(sample).tobytes()).hexdigest()==meta['sha256'],
                        "The fixed view's anatomical evidence differs.")
                binary[name] = sample.astype(bool)
            a = pixels[2][y0:y1,x0:x1]
            tagged = None
            if weights is not None:
                contribution = render(cam,model,pipe,black,override_color=weights)['render']
                require(torch.isfinite(contribution).all() and float((contribution-contribution[:1]).abs().max())<1e-6,
                        'Head-bound alpha-contribution readback is invalid.')
                head = contribution[0,y0:y1,x0:x1].cpu().numpy()
                require(float((head-a).max())<1e-5,'Head-bound contribution exceeded the full-scene alpha.')
                tagged = exterior_stats(head,binary['head_exterior_ring'])
            rows[key] = dict(camera=record,fixed_world_camera=view['world_camera'],actual_portable_ply=side in PORTABLE,
                gaussian_points=static['points'],center_receipt=camera_entry['centers'],
                black_white_alpha_composition_max_abs=composition,
                opacity={name:opacity_stats(a,mask) for name,mask in binary.items() if name!='head_exterior_ring'},
                anatomical_exterior_ring=exterior_stats(a,binary['head_exterior_ring']),head_bound_contribution_to_exterior=tagged,
                panel=panel_image(directory,key,view,pixels,binary['head_silhouette']),real_rgb_ground_truth_available=False)
    review.save(directory/'result.json',dict(side=side,view_count=len(rows),records=rows,ply_sha256=static['ply_sha256']))


def combine(out,plan):
    block = plan['static_grid']
    shots = out/'static-grid'
    states = {}
    for side in ARM_NAMES:
        path = shots/side/'result.json'
        values = read(path)
        require(values['side']==side and values['view_count']==50 and set(values['records'])==set(block['camera_roster'])
                and values['ply_sha256']==block['arms'][side]['ply_sha256'],'Static grid evidence missed a fixed view or PLY.')
        states[side] = dict(receipt=descriptor(path),records=values['records'])
    panels = []
    for key,view in block['camera_roster'].items():
        for names,group in ((PORTABLE,'actual'),(('before10','r21_10'),'same_run_10k'),(('before18','r21'),'same_run_18k')):
            records = [states[side]['records'][key] for side in names]
            first = records[0]['panel']
            require(all(item['fixed_world_camera']==view['world_camera'] and item['panel']['crop_xyxy']==first['crop_xyxy']
                        and item['panel']['columns']==first['columns'] and not item['real_rgb_ground_truth_available']
                        for item in records),'A paired static scene changed the native review grid.')
            images = []
            for item in records:
                file = item['panel']
                with Image.open(pin(file['file'],file['sha256'])) as image:
                    images.append(image.convert('RGB'))
            w,h = images[0].size
            require(all(image.size==(w,h) for image in images),'An actual static panel changed its review pixel grid.')
            combined = Image.new('RGB',(w,h*len(images)))
            for idx,image in enumerate(images):
                combined.paste(image,(0,h*idx))
            name = f'static_grid_{group}_{key}.png'
            file = out/name
            combined.save(file)
            panels.append(dict(view=f'static_grid_{group}_{key}',file=name,sha256=sha(file),frame=view['world_camera']['frame'],
                new_frame=view['world_camera']['frame'],crop_xyxy=first['crop_xyxy'],columns=first['columns'],rows=list(names),
                grid_id=key,distance_m=view['distance_m'],distance_class=view['distance_class'],azimuth_deg=view['azimuth_deg'],
                pitch_deg=view['pitch_deg'],real_rgb_ground_truth_available=False))
    path = shots/'result.json'
    require(not path.exists(),'Preserve the completed static-grid evidence.')
    review.save(path,dict(schema='shuolee-r21-fixed-static-grid-result/1',subject=plan['subject'],state='complete',
        per_arm_views=50,actual_portable_arms=list(PORTABLE),same_run_controls=['before10','before18'],
        cameras=block['proof']['camera_manifest'],proxy=block['proof']['proxy_manifest'],arms=states,panels=panels,
        no_real_ground_truth=True,quality_accepted=False))
    return descriptor(path),panels
