"""Compare only the four directed round-two source views and the locked dollies."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw
from plyfile import PlyData
from scipy.spatial import ConvexHull, QhullError

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_smpl_delivery_round1 as round1
from review_bound_scale_coverage import face_alpha, interior


review, transfer = round1.review, round1.transfer
pin,read,require,sha = round1.pin,round1.read,round1.require,round1.sha
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-surface-20261111')
RECEIPT = Path(__file__).resolve().parents[1]/'outputs/smpl-delivery-round1/receipt.json'
RECEIPT_SHA = '0cd5b6afaaa6bc273646ce264fdb5d39c8dfcac834940f30f77492f367bc3732'
SCOPES = {'016':('frame_000027','frame_000051'),'006':('frame_000033','frame_000044')}
REGIONS = {'face':slice(67,118),'eye_36_41':slice(86,92),'eye_42_47':slice(92,98),'nose':slice(77,86)}


def hash_mask(mask):
    return hashlib.sha256(np.ascontiguousarray(mask,dtype=np.bool_).tobytes()).hexdigest()


def projection(world,row):
    K,E = np.asarray(row['K']),np.asarray(row['w2c'])
    camera = world@E[:3,:3].T+E[:3,3]
    screen = camera@K.T
    return screen[:,:2]/screen[:,2:3]-.5,camera[:,2]


def polygon(xy,wh,hull):
    if hull:
        try:
            xy = xy[ConvexHull(xy).vertices]
        except QhullError:
            return None
    raster = Image.new('1',tuple(wh))
    ImageDraw.Draw(raster).polygon([tuple(point) for point in xy],fill=1)
    return np.asarray(raster)


def source_regions(fit,index,native,view,matte,crop):
    E = np.asarray(native['w2c'])
    world = (fit['joints_camera'][index].astype(np.float64)-E[:3,3])@E[:3,:3]
    observed = fit['observed_native_keypoints'][index]
    xy_native,z_native = projection(world,native)
    xy_view,z_view = projection(world,view)
    require(np.allclose(xy_native,fit['joints_projected_native'][index],atol=.001,rtol=0),
            'Saved fitted native projection does not match its source camera.')
    real_view = view['w2c'] == native['w2c']
    errors = np.linalg.norm(xy_native-observed[:,:2],axis=1)
    flags = np.isfinite(observed).all(axis=1)&(observed[:,2]>=.5)&(z_native>.2)&(errors<=6.)
    flags &= ((observed[:,:2]>=0)&(observed[:,:2]<native['wh'])).all(axis=1)
    masks,report = {},{}
    for name,slots in REGIONS.items():
        supported = flags[slots]
        entry = dict(supported_landmarks=int(supported.sum()),registered_landmarks=len(supported),eligible=False)
        report[name] = entry
        if supported.sum() < (8 if name == 'face' else len(supported)):
            entry['reason'] = 'native_landmarks_fail_confidence_or_projection_agreement'
            continue
        entry['max_native_landmark_disagreement_px'] = float(errors[slots][supported].max())
        native_xy = observed[slots,:2][supported]
        native_mask = interior(native_xy,native['wh']) if name == 'face' else polygon(native_xy,native['wh'],name == 'nose')
        if native_mask is None or (name != 'face' and not matte[native_mask].all()):
            entry['reason'] = 'unsupported_native_roi_shape_or_foreground'
            continue
        if int((native_mask&matte).sum()) < 32:
            entry['reason'] = 'native_roi_under_32_pixels'
            continue
        if real_view:
            roi = native_mask&matte
        else:
            xy = xy_view[slots][supported]
            x0,y0,x1,y1 = crop
            if (z_view[slots][supported]<=.2).any() or not ((xy>=[x0,y0])&(xy<[x1,y1])).all():
                entry['reason'] = 'projected_roi_outside_locked_dolly_crop'
                continue
            roi = interior(xy,view['wh']) if name == 'face' else polygon(xy,view['wh'],name == 'nose')
        if roi is None or int(roi.sum()) < 32:
            entry['reason'] = 'view_roi_under_32_pixels'
            continue
        masks[name] = roi
        entry.update(eligible=True,pixels=int(roi.sum()),mask_sha256=hash_mask(roi))
    return masks,report


def baseline(subject):
    pin(RECEIPT,RECEIPT_SHA)
    receipt = read(RECEIPT)
    entry = receipt['subjects'][subject]
    require(receipt['commit'] == entry['commit'] == round1.COMMIT,'Wrong director delivery lineage.')
    result_path = Path(entry['result'])
    run = result_path.parent
    prior_review = Path(entry['native_review'])
    pin(prior_review,entry['native_review_sha256'])
    old_result = read(prior_review)
    prior_plan = prior_review.with_name('plan.json')
    pin(prior_plan,old_result['plan_sha256'])
    cached = read(prior_plan)
    old = review.planned_model(run,round1.NEW_PROJECT,entry['result_sha256'])
    require(cached['new'] == old.provenance and old_result['state'] == 'delivery_views_complete',
            'Round-one same-camera review does not bind the director endpoint.')
    parts = round1.delivery_paths(run,read(result_path))
    return old,parts,cached


def preflight(subject,commit=None):
    old,base_parts,cached = baseline(subject)
    root = review.LAB/f'shuolee-smpl-delivery-round2-{subject}'
    new = new_parts = None
    if commit is not None:
        transfer.rebound(transfer.base.common.git_state,PROJECT=PROJECT)(commit)
        wrapper = read(root/'wrapper-terminal.json')
        require(wrapper['state'] == 'completed' and wrapper['candidate_commit'] == commit,'Round-two terminal is not completed.')
        run = root/'run'
        new = review.planned_model(run,PROJECT,wrapper['result_sha256'])
        terminal = read(run/'result.json')
        new_parts = round1.delivery_paths(run,terminal)
        require(all(new.provenance[key] == old.provenance[key] for key in ('fitted_sha256','training_inputs_sha256',
                    'protected_holdouts','reference_frame_id')) and terminal['fit_reused'] is True and terminal['fit_calls'] == 0,
                'Round-two fitted pose, camera, masks or source rows changed.')
        round1.check_source_equivalence(round1.NEW_PROJECT,PROJECT,base_parts[2]['source_files_sha256'],
                                        new_parts[2]['source_files_sha256'])
    pairs,regions,arrays,views = [],{},{},{}
    with np.load(Path(old.provenance['run'])/'fit/fitted.npz',allow_pickle=False) as archive:
        fit = {key:archive[key] for key in ('observed_native_keypoints','joints_camera','joints_projected_native')}
    for prior,frame in zip(cached['pairs'],SCOPES[subject],strict=True):
        spec = dict(prior['new'])
        require(spec['frame'] == frame and prior['exact_fitted_pose'],'Director locked a different source frame or pose.')
        row = old.rows[frame]
        pose = old.slots[frame]
        pin(Path(row['image']),row['source_sha256']); pin(Path(row['mask']),row['mask_sha256'])
        require(spec['pose_sha256'] == review.pose_digest(old,frame) and spec['score_mask_sha256'] == row['mask_sha256'],
                'Round-one camera review selected a different native fit.')
        with Image.open(row['mask']) as mask:
            matte = np.asarray(mask.convert('L'))>=253
        face_box = row['face_box_xyxy']
        require(face_box is not None,'Missing source detector head box for the directed view.')
        x0,y0,x1,y1 = face_box
        # Include the pinna and temple at the native side view. The ears are visual-only.
        if prior['label'] == 'side':
            ear = fit['observed_native_keypoints'][pose,[17,18]]
            supported = ear[(ear[:,2]>=.5)&np.isfinite(ear).all(1)]
            if len(supported):
                x0 = min(x0,float(supported[:,0].min())); x1 = max(x1,float(supported[:,0].max()))
                y0 = min(y0,float(supported[:,1].min())); y1 = max(y1,float(supported[:,1].max()))
        spec['head_crop'] = round1.native_box([x0-24,y0-24,x1+24,y1+24],row['wh'])
        requested = [(prior['label'],row,spec['head_crop'])]
        if prior['label'] == 'front':
            close = spec['dolly']
            requested.append(('dolly',dict(row,w2c=close['w2c']),close['crop']))
        for name,camera,crop in requested:
            masks,regions[name] = source_regions(fit,pose,row,camera,matte,crop)
            views[name] = dict(frame=frame,camera=camera,pose_index=pose,crop_xyxy=crop,native_rgb_truth=name != 'dolly')
            arrays.update({f'{name}_{key}':mask for key,mask in masks.items()})
        pairs.append(dict(label=prior['label'],old=dict(spec),new=dict(spec),exact_native_source_and_camera=True,exact_fitted_pose=True))
    require(all(regions['front'][key]['eligible'] for key in REGIONS),'Required native reference regions are unavailable.')
    depth = {}
    for key,parts in (('old',base_parts),('new',new_parts)):
        if parts is not None:
            points = PlyData.read(parts[0]/'world.ply')['vertex'].data
            xyz = np.stack([points[axis] for axis in ('x','y','z')],axis=1)
            dz = np.asarray(views['dolly']['camera']['w2c'][2])
            depth[key] = float((xyz@dz[:3]+dz[3]).min())
            require(np.isfinite(depth[key]) and depth[key]>.2,'A delivered Gaussian center crosses the locked dolly cutoff.')
    plan = dict(schema='shuolee-round2-delivery-views-v1',subject=subject,asset=str(review.ASSET),
        old=old.provenance,new=new.provenance if new else None,old_commit=round1.COMMIT,new_commit=commit,
        pairs=pairs,views=views,regions=regions,dolly_min_center_depth_m=depth,output=str(root/'view-comparison'),
        director_round1_receipt_sha256=RECEIPT_SHA,reviewer_sha256=sha(Path(__file__)),
        observation_scope='Native fitted-source landmarks and confident matte; ear visual-only; dolly has no measured RGB.',
        optimizer_updates=0,refits=0,quality_accepted=False)
    return plan,arrays


def worker(path,side,fd):
    plan = read(path)
    pin(Path(__file__),plan['reviewer_sha256'])
    archive = path.with_name('rois.npz')
    pin(archive,plan['roi_archive_sha256'])
    with np.load(archive,allow_pickle=False) as data:
        masks = dict(data)
    for name,regions in plan['regions'].items():
        for region,info in regions.items():
            if info['eligible']:
                require(hash_mask(masks[f'{name}_{region}']) == info['mask_sha256'],'A source-bound ROI changed.')
    sources,metrics,head_panels = {},{},{}
    for name,view in plan['views'].items():
        if view['native_rgb_truth']:
            source = view['camera']
            pin(Path(source['image']),source['source_sha256'])
            with Image.open(source['image']) as image:
                sources[name] = np.asarray(image.convert('RGB'))

    def raster(render,render_pose,cam,model,pipe,black,white,pose,scale):
        pixels,error = review.scale_pixels(render,render_pose,cam,model,pipe,black,white,pose,scale)
        matrix = cam.world_view_transform.T.detach().cpu().numpy()
        matching = [key for key,view in plan['views'].items() if view['frame'] == cam.image_name and view['pose_index'] == pose
                    and np.allclose(matrix,view['camera']['w2c'],rtol=0,atol=1e-6)]
        require(len(matching) == 1,'Renderer called an unregistered camera.')
        name = matching[0]
        region_metrics = {}
        for region,info in plan['regions'][name].items():
            if info['eligible']:
                mask = masks[f'{name}_{region}']
                rgb_mae = float(np.abs(pixels[0][mask]-sources[name][mask].astype(np.float32)/255).mean()) if name in sources else None
                region_metrics[region] = dict(alpha=face_alpha(pixels[2],mask),native_black_rgb_mae=rgb_mae)
        metrics[f'{name}:{scale:.1f}'] = region_metrics
        return pixels,error

    def record_panel(path,*args):
        item = round1.panel(path,*args)
        if path.stem.endswith('-head'):
            head_panels[path.stem.split('-')[0]] = item
        return item

    sys.path.insert(0,str(Path(plan[side]['project'])/'src'))
    from shuolee_portrait.cli import gpu_lock
    with gpu_lock():
        transfer.rebound(review.worker,delivered=round1.delivered,panel=record_panel,scale_pixels=raster)(path,side,fd)
    require(set(metrics) == {f'{name}:{scale:.1f}' for name in plan['views'] for scale in (1.,.5)}
            and set(head_panels) == {'front','side'},'A required comparison view or close crop was not rendered.')
    review.save(path.parent/f'{side}-regions.json',dict(metrics=metrics,head_panels=head_panels))


def merge_heads(output,plan,old,new):
    panels = []
    for name in ('front','side'):
        a,b = (arm['head_panels'][name] for arm in (old,new))
        require(a['crop_xyxy'] == b['crop_xyxy'],'Native head crops differ.')
        pin(output/a['file'],a['sha256']); pin(output/b['file'],b['sha256'])
        with Image.open(output/a['file']) as before, Image.open(output/b['file']) as after:
            require(before.size == after.size and before.height%2 == 0,'Invalid head panels.')
            w,h = before.width,before.height//2
            for i,scale in enumerate((1.,.5)):
                rect = (0,i*h,w,(i+1)*h)
                image = Image.new('RGB',(w,2*h))
                image.paste(before.crop(rect),(0,0)); image.paste(after.crop(rect),(0,h))
                frame = plan['views'][name]['frame']
                file = output/f'{name}-head-{frame}-scale-{scale:.1f}.png'
                image.save(file)
                panels.append(dict(file=file.name,sha256=sha(file),frame=frame,view=f'{name}_head',scale=scale,
                    header_pixels=0,crop_xyxy=a['crop_xyxy'],columns=['source_rgb','black','white','alpha']))
    return panels


def main():
    if len(sys.argv) == 5 and sys.argv[1] == '_worker':
        worker(Path(sys.argv[2]),sys.argv[3],int(sys.argv[4])); return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('check','render'))
    parser.add_argument('subject',choices=SCOPES)
    parser.add_argument('--commit')
    args = parser.parse_args()
    if args.action == 'check':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    elif not args.commit:
        parser.error('--commit must pin the exact completed common candidate')
    plan,masks = preflight(args.subject,args.commit)
    if args.action == 'check':
        print(json.dumps(dict(state='cpu_preflight_passed',terminal_ready=plan['new'] is not None,
              frames=SCOPES[args.subject],regions=plan['regions'],gpu_calls=0)),flush=True); return
    output = Path(plan['output'])
    with review.GPU_LEASE.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        output.mkdir(exist_ok=False)
        path,roi_path = output/'plan.json',output/'rois.npz'
        np.savez_compressed(roi_path,**masks)
        plan['roi_archive_sha256'] = sha(roi_path)
        review.save(path,plan)
        for side in ('old','new'):
            with (output/f'{side}.log').open('x',encoding='utf-8') as log:
                subprocess.run([sys.executable,'-I','-B',__file__,'_worker',str(path),side,str(lease.fileno())],
                               pass_fds=(lease.fileno(),),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
    old,new = (read(output/f'{side}-result.json') for side in ('old','new'))
    metrics = {side:read(output/f'{side}-regions.json') for side in ('old','new')}
    panels = round1.merge(output,old,new)+merge_heads(output,plan,metrics['old'],metrics['new'])
    for item in panels:
        item['rows'] = ['round1_762d6d0',f'round2_{args.commit[:7]}']
    path = output/'result.json'
    review.save(path,dict(state='round2_delivery_views_complete',subject=args.subject,plan_sha256=sha(output/'plan.json'),
        old=old,new=new,panels=panels,regions=plan['regions'],region_metrics=metrics,optimizer_updates=0,refits=0,
        dolly_rgb_truth_available=False,quality_accepted=False))
    print(json.dumps(dict(result=str(path),sha256=sha(path),panels=len(panels))),flush=True)


if __name__ == '__main__':
    main()
