"""Compare round three to the two exact delivered native and portable endpoints."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
from PIL import Image
from plyfile import PlyData

sys.path.insert(0,str(Path(__file__).resolve().parent))
import review_smpl_delivery_round2 as prior
from review_bound_scale_coverage import exported_view


review,round1,transfer = prior.review,prior.round1,prior.transfer
read,pin,require,sha = prior.read,prior.pin,prior.require,prior.sha
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-coverage-20261111')
ROUND2_RECEIPT = Path(__file__).resolve().parents[1]/'outputs/smpl-delivery-round2/receipt.json'
ROUND2_RECEIPT_SHA = '9598cc638c8e035e37a0472992e08c58461f961d9fc23c6653fe3bccecb8ef56'
ROUND2_REVIEWER_SHA = 'fec880bee1b15b277ad71f99280caadbca0c0a8f4a121fdb4f6fe0cb269d7f72'
ROUND2_COMMIT = '9a0ef65026e027f1ba2020fdf0a411b7091d80c6'
FINAL_SNAPSHOT = 'step-010000-head-000600-surface-001800-coverage-001500'
REVIEW_PINS = {
    '016':'b7a22673463a67232d5a009111a868480afc12d0fe0a42c144c719b4db38f8d5',
    '006':'0514710640f8ec168343efd840df481fa3bfd0dc5c02d59ca6d6870fc84a536c',
}


def portable(provenance,native,parts):
    run = Path(provenance['run'])
    terminal = read(run/'result.json')
    path,camera_path = (run/terminal[key] for key in ('ply','camera'))
    pin(path,terminal['sha256'])
    camera_receipt = read(camera_path.with_name('camera_provenance.json'))
    digest = pin(camera_path,camera_receipt['files_sha256']['cameras.json'])
    with zipfile.ZipFile(run/terminal['scene']) as bundle:
        with bundle.open('cameras.json') as packed_camera, bundle.open('point_cloud/iteration_10000/point_cloud.ply') as packed_ply:
            require(transfer.base.common.stream_sha(packed_camera) == digest
                    and transfer.base.common.stream_sha(packed_ply) == terminal['sha256'],
                    'Portable PLY and camera bytes differ inside the delivered bundle.')
    doc, = read(camera_path)
    row, = camera_receipt['records']
    require(camera_receipt['cameras_share_model_world'] is True and row['frame'] == native['frame']
            and doc['pixelConvention'] == 'pixel_corners' and doc['projectionModel'] == 'pinhole',
            'Portable PLY camera is not the exact fixed reference.')
    R = np.asarray(doc['rotation']).T
    w2c = np.eye(4)
    w2c[:3,:3],w2c[:3,3] = R,-R@np.asarray(doc['position'])
    K = [[doc['fx'],0.,doc['cx']],[0.,doc['fy'],doc['cy']],[0.,0.,1.]]
    camera = dict(frame=native['frame'],wh=[doc['width'],doc['height']],K=K,w2c=w2c.tolist())
    transform = camera_receipt['world_to_viewer']
    expected = exported_view(native,transform)
    require(camera['K'] == row['K'] == expected['K'] and camera['wh'] == row['wh'] == expected['wh']
            and np.allclose(w2c,row['w2c'],rtol=0,atol=1e-12) and np.allclose(w2c,expected['w2c'],rtol=0,atol=1e-12)
            and transform['scale'] == 1. and parts[0].is_dir(), 'Portable JSON and exact export projection disagree.')
    return dict(ply=str(path),ply_sha256=terminal['sha256'],points=terminal['points'],camera_path=str(camera_path),
                camera_sha256=digest,front=camera,world_to_viewer=transform)


def static_views(arms,reference_mesh,front_crop):
    base = arms['round1']
    front = base['front']
    require(all(item['front'] == front for item in arms.values()),'Portable reference cameras differ between rounds.')
    transform = base['world_to_viewer']
    vertices = (reference_mesh-np.asarray(transform['origin_world']))@np.asarray(transform['rotation']).T
    pivot = (vertices.min(axis=0)+vertices.max(axis=0))*.5
    Q = np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
    w2c = np.asarray(front['w2c'])
    R = w2c[:3,:3]
    C = -R.T@w2c[:3,3]
    orbit_c = pivot+Q@(C-pivot)
    forward = pivot-orbit_c
    forward /= np.linalg.norm(forward)
    right = np.cross(forward,[0.,1.,0.])
    right /= np.linalg.norm(right)
    orbit = np.eye(4)
    orbit[:3,:3] = np.stack((right,np.cross(forward,right),forward))
    orbit[:3,3] = -orbit[:3,:3]@orbit_c
    side = dict(front,w2c=orbit.tolist())
    uv,depth = prior.projection(vertices,side)
    require(np.isfinite(uv).all() and (depth>.2).all(),'Static orbit intersects the framing mesh.')
    lo,hi = uv.min(axis=0),uv.max(axis=0)
    pad = max(64.,.12*float(hi[1]-lo[1]))
    side_crop = round1.native_box([*(lo-pad),*(hi+pad)],front['wh'])
    return dict(static_reference_front=dict(camera=front,crop_xyxy=front_crop,synthetic=False,source_rgb_available=True),
                static_rotated_side=dict(camera=side,crop_xyxy=side_crop,synthetic=True,source_rgb_available=False,
                    orbit_y_degrees=90,pivot_viewer_world=pivot.tolist()))


def preflight(subject,commit=None):
    pin(ROUND2_RECEIPT,ROUND2_RECEIPT_SHA)
    pin(Path(prior.__file__),ROUND2_REVIEWER_SHA)
    delivery = read(ROUND2_RECEIPT)
    require(delivery['commit'] == ROUND2_COMMIT and delivery['baseline_commit'] == round1.COMMIT,'Unexpected round-two delivery.')
    spec = delivery['subjects'][subject]
    old,old_parts,_ = prior.baseline(subject)
    baseline_result = Path(spec['review'])
    pin(baseline_result,REVIEW_PINS[subject])
    baseline = read(baseline_result)
    cache_path = baseline_result.with_name('plan.json')
    pin(cache_path,baseline['plan_sha256'])
    locked = read(cache_path)
    require(baseline['state'] == 'round2_delivery_views_complete' and locked['reviewer_sha256'] == ROUND2_REVIEWER_SHA
            and locked['old'] == old.provenance and locked['new_commit'] == ROUND2_COMMIT
            and tuple(pair['new']['frame'] for pair in locked['pairs']) == prior.SCOPES[subject],
            'Round-two native, real side or dolly baseline identity changed.')
    cache_rois = baseline_result.with_name('rois.npz')
    pin(cache_rois,locked['roi_archive_sha256'])
    second_run = Path(spec['result']).parent
    second = review.planned_model(second_run,prior.PROJECT,spec['result_sha256'])
    require(locked['new'] == second.provenance,'Round-two receipt references another terminal model.')
    second_parts = round1.delivery_paths(second_run,read(second_run/'result.json'))
    roots = {'round1':(old,old_parts),'round2':(second,second_parts)}
    output = review.LAB/f'shuolee-smpl-delivery-round3-{subject}'
    candidate = None
    if commit:
        transfer.rebound(transfer.base.common.git_state,PROJECT=PROJECT)(commit)
        wrapper = read(output/'wrapper-terminal.json')
        require(wrapper['state'] == 'completed' and wrapper['candidate_commit'] == commit,'Round-three wrapper is incomplete.')
        run = output/'run'
        candidate = review.planned_model(run,PROJECT,wrapper['result_sha256'])
        terminal = read(run/'result.json')
        parts = round1.delivery_paths(run,terminal)
        require(parts[0].name == terminal['final_snapshot'] == FINAL_SNAPSHOT
                and terminal['fit_reused'] is True and terminal['fit_calls'] == 0
                and all(candidate.provenance[key] == old.provenance[key] for key in
                    ('fitted_sha256','training_inputs_sha256','protected_holdouts','reference_frame_id')),
                'Round-three snapshot or exact fitted observation identity differs.')
        round1.check_source_equivalence(prior.PROJECT,PROJECT,second_parts[2]['source_files_sha256'],parts[2]['source_files_sha256'])
        roots['round3'] = candidate,parts
    front = locked['views']['front']['camera']
    for key,(model,_) in roots.items():
        for pair in locked['pairs']:
            spec_frame = pair['new']
            require(review.pose_digest(model,spec_frame['frame']) == spec_frame['pose_sha256'],f'{key} changed a locked review pose.')
    arms = {key:portable(model.provenance,front,parts) for key,(model,parts) in roots.items()}
    with np.load(old_parts[0]/'reference-mesh.npz',allow_pickle=False) as archive:
        mesh = archive['vertices_world']
    views = static_views(arms,mesh,locked['pairs'][0]['new']['crop'])
    for arm in arms.values():
        points = PlyData.read(arm['ply'])['vertex'].data
        xyz = np.stack([points[axis] for axis in ('x','y','z')],axis=1)
        arm['min_center_depth_m'] = {}
        for key,view in views.items():
            z = np.asarray(view['camera']['w2c'][2])
            depth = float((xyz@z[:3]+z[3]).min())
            require(np.isfinite(depth) and depth>.2,'Static portable view crosses the stock raster cutoff.')
            arm['min_center_depth_m'][key] = depth
    plan = dict(locked,subject=subject,new=candidate.provenance if candidate else None,new_commit=commit,
                output=str(output/'view-comparison'),round3_reviewer_sha256=sha(Path(__file__)),
                baseline_review_sha256=REVIEW_PINS[subject],baseline_review=str(baseline_result),
                static_views=views,static_arms=arms)
    return plan,baseline,cache_rois


def static_worker(path):
    import torch
    from shuolee_portrait import runtime as rt
    from shuolee_portrait.cli import gpu_lock
    from shuolee_portrait.training import camera,render_pose

    plan = read(path)
    rt.configure(review.ASSET.parents[1])
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel

    receipts = {}
    with gpu_lock(),torch.no_grad():
        require(torch.cuda.device_count() == 1,'Expose exactly one CUDA device.')
        black,white = torch.zeros(3,device='cuda'),torch.ones(3,device='cuda')
        pipe = SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
        for arm,item in plan['static_arms'].items():
            pin(Path(item['ply']),item['ply_sha256']); pin(Path(item['camera_path']),item['camera_sha256'])
            model = GaussianModel(0)
            model.load_ply(item['ply'])
            require(model.binding is None and model.optimizer is None and model.active_sh_degree == 0
                    and len(model.get_xyz) == item['points'],'A static PLY unexpectedly has binding or wrong point count.')
            receipts[arm] = {}
            for name,view in plan['static_views'].items():
                cam,actual = camera(view['camera'],0,pose_index=0)
                images,error = review.scale_pixels(render,render_pose,cam,model,pipe,black,white,0,1.)
                x0,y0,x1,y1 = view['crop_xyxy']
                w,h = x1-x0,y1-y0
                sheet = Image.new('RGB',(3*w,h))
                for index,image in enumerate(images):
                    pixels = np.rint(np.clip(image[y0:y1,x0:x1],0,1)*255).astype(np.uint8)
                    sheet.paste(Image.fromarray(pixels).convert('RGB'),(index*w,0))
                panel = path.parent/f'{name}-{arm}.png'
                sheet.save(panel)
                receipts[arm][name] = dict(camera=actual,file=panel.name,sha256=sha(panel),composition_max_abs=error,
                    static_ply_sha256=item['ply_sha256'],pose_driven=False,source_rgb_available=view['source_rgb_available'],
                    source_rgb_compared=False)
            del model
    review.save(path.parent/'static-arms.json',receipts)


def assemble(output,plan,previous,new,regions,static):
    panels = []
    previous_root = Path(plan['baseline_review']).parent
    records = {item['role']:item for item in new['records']}
    names = ['round1_762d6d0','round2_9a0ef65',f'round3_{plan["new_commit"][:7]}']
    for item in previous['panels']:
        name,scale = item['view'],item['scale']
        if name in ('front','side'):
            new_panel = records[name]['native_panel']
        elif name == 'dolly':
            new_panel = records['front']['closeup']['panel']
        else:
            new_panel = regions['head_panels'][name.removesuffix('_head')]
        require(item['header_pixels'] == 0 and item['crop_xyxy'] == new_panel['crop_xyxy'],'Review crops or panel layout changed.')
        old_path = previous_root/item['file']; new_path = output/new_panel['file']
        pin(old_path,item['sha256']); pin(new_path,new_panel['sha256'])
        with Image.open(old_path) as old,Image.open(new_path) as candidate:
            require(old.size == candidate.size and old.height%2 == 0,'Matched panel dimensions differ.')
            w,h = old.width,old.height//2
            i = 0 if scale == 1. else 1
            sheet = Image.new('RGB',(w,3*h))
            sheet.paste(old,(0,0)); sheet.paste(candidate.crop((0,i*h,w,(i+1)*h)),(0,2*h))
            destination = output/item['file']
            sheet.save(destination)
        panels.append(dict(item,sha256=sha(destination),rows=names))
    for name,view in plan['static_views'].items():
        tiles = []
        for arm in plan['static_arms']:
            item = static[arm][name]
            path = output/item['file']
            pin(path,item['sha256'])
            with Image.open(path) as image:
                tiles.append(image.convert('RGB'))
        w,h = tiles[0].size
        require(all(item.size == (w,h) for item in tiles),'Portable panel sizes differ between rounds.')
        sheet = Image.new('RGB',(w,3*h))
        for index,tile in enumerate(tiles):
            sheet.paste(tile,(0,index*h))
        path = output/f'{name}-{view["camera"]["frame"]}-scale-1.0.png'
        sheet.save(path)
        panels.append(dict(file=path.name,sha256=sha(path),frame=view['camera']['frame'],view=name,scale=1.,header_pixels=0,
            crop_xyxy=view['crop_xyxy'],rows=names,columns=['black','white','alpha'],pose_driven=False,
            source_rgb_available=view['source_rgb_available'],source_rgb_compared=False,
            synthetic_camera=view['synthetic'],orbit_y_degrees=view.get('orbit_y_degrees')))
    return panels


def main():
    if len(sys.argv) == 4 and sys.argv[1] == '_worker':
        path,fd = Path(sys.argv[2]),int(sys.argv[3])
        plan = read(path)
        pin(Path(__file__),plan['round3_reviewer_sha256'])
        prior.worker(path,'new',fd)
        static_worker(path)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('check','render'))
    parser.add_argument('subject',choices=prior.SCOPES)
    parser.add_argument('--commit')
    args = parser.parse_args()
    if args.action == 'check':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    elif not args.commit:
        parser.error('--commit must pin the exact completed round-three candidate')
    plan,baseline,rois = preflight(args.subject,args.commit)
    if args.action == 'check':
        print(json.dumps(dict(state='cpu_preflight_passed',terminal_ready=plan['new'] is not None,gpu_calls=0,
            frames=prior.SCOPES[args.subject],static_views=plan['static_views'],
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
    review.save(result,dict(state='round3_delivery_views_complete',subject=args.subject,plan_sha256=sha(path),
        panels=panels,regions=plan['regions'],region_metrics=dict(round1=baseline['region_metrics']['old'],
            round2=baseline['region_metrics']['new'],round3=regions),static_render_receipts=static,
        old=baseline['old'],round2=baseline['new'],new=new,optimizer_updates=0,refits=0,
        dolly_rgb_truth_available=False,static_rotated_side_rgb_truth_available=False,quality_accepted=False))
    print(json.dumps(dict(result=str(result),sha256=sha(result),panels=len(panels))),flush=True)


if __name__ == '__main__':
    main()
