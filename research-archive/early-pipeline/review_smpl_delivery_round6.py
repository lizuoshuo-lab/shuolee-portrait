"""Compare round six against the exact frozen round-five review cameras."""

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

sys.path.insert(0,str(Path(__file__).resolve().parent))
import review_smpl_delivery_round5 as prior


review, round1, transfer = prior.review, prior.round1, prior.transfer
read, pin, require, sha = prior.read, prior.pin, prior.require, prior.sha
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-scale-20261111')
BASE_COMMIT = '9013c26bc1b0758a5798ae2c3b0cf28e4d1a72c0'
BASE_RECEIPT = Path(__file__).resolve().parents[1]/'outputs/smpl-delivery-round5/receipt.json'
BASE_RECEIPT_SHA = '6db51a1c18534fa9b7965383d6162107c69ccf3e0335fa32cb05ebafca31cf5d'
BASE_REVIEWER_SHA = '71fef93284dc2f42668a026de888f678c0fd1ebead8f7818e9303ce0a1a2f5f5'
STATIC_WORKER_SHA = prior.prior.ROUND3_REVIEWER_SHA
native,static_worker = prior.native,prior.prior.prior


def root_for(subject):
    return review.LAB/f'shuolee-smpl-delivery-round6-{subject}'


def completed_training(subject,commit):
    root = root_for(subject)
    run = root/'run'
    wrapper = read(root/'wrapper-terminal.json')
    terminal_path = run/'result.json'
    pin(terminal_path,wrapper['result_sha256'])
    terminal = read(terminal_path)
    integration_path = run/'gaussian-run/integration.json'
    integration_sha = pin(integration_path,wrapper['integration_sha256'])
    integration = read(integration_path)
    require(wrapper['state'] == terminal['state'] == integration['state'] == 'completed'
            and wrapper['candidate_commit'] == commit and terminal['final_snapshot'] == prior.FINAL_SNAPSHOT
            and terminal['fit_reused'] is True and terminal['fit_calls'] == 0
            and terminal['gaussian_from_scratch'] is True and terminal['unified_main_training'] == integration
            and integration['optimizer_updates'] == 10000 and terminal['points'] == wrapper['points'],
            'Expected exact fresh 10k round-six terminal and integration receipt.')
    scale_path = run/'gaussian-run/world-scale.json'
    scale_pin = integration['world_scale']
    scale_sha = pin(scale_path,scale_pin['sha256'])
    scale = read(scale_path)
    require(scale_pin['file'] == scale_path.name and scale['state'] == 'completed'
            and scale_pin['terminal'] == scale['terminal'],
            'The round-six training receipt does not bind the exact terminal world-scale metrics.')
    receipts = dict(integration_receipt=dict(file=str(integration_path),sha256=integration_sha),
                    world_scale_receipt=dict(file=str(scale_path),sha256=scale_sha))
    return terminal,wrapper,receipts


def preflight(subject,commit=None):
    pin(BASE_RECEIPT,BASE_RECEIPT_SHA)
    pin(Path(prior.__file__),BASE_REVIEWER_SHA)
    pin(Path(static_worker.__file__),STATIC_WORKER_SHA)
    delivery = read(BASE_RECEIPT)
    require(delivery['commit'] == BASE_COMMIT,'Unexpected round-five delivery.')
    spec = delivery['subjects'][subject]
    previous = Path(spec['review'])
    pin(previous,spec['review_sha256'])
    baseline = read(previous)
    previous_plan = previous.with_name('plan.json')
    pin(previous_plan,baseline['plan_sha256'])
    locked = read(previous_plan)
    rois = previous_plan.with_name('rois.npz')
    pin(rois,locked['roi_archive_sha256'])
    pin(Path(native.__file__),locked['reviewer_sha256'])
    require(baseline['state'] == 'round5_delivery_views_complete' and locked['new_commit'] == BASE_COMMIT
            and locked['round5_reviewer_sha256'] == BASE_REVIEWER_SHA
            and all(locked[key] == baseline[key] for key in ('regions','source_angular_support','native_view_labels'))
            and locked['source_angular_support']['physical_angular_coverage'] == 'unknown'
            and tuple(pair['new']['frame'] for pair in locked['pairs']) == native.SCOPES[subject],
            'The round-five frozen camera, ROI or angular evidence differs.')
    for panel in baseline['panels']:
        pin(previous.parent/panel['file'],panel['sha256'])
    old_run = Path(spec['result']).parent
    old = review.planned_model(old_run,prior.PROJECT,spec['result_sha256'])
    old_parts = round1.delivery_paths(old_run,read(old_run/'result.json'))
    front = locked['views']['front']['camera']
    old_static = static_worker.portable(old.provenance,front,old_parts)
    require(old.provenance == locked['new'] and old_parts[0].name == prior.FINAL_SNAPSHOT
            and old_static['ply_sha256'] == locked['static_arms']['round5']['ply_sha256'],
            'The round-five endpoint changed.')
    root = root_for(subject)
    candidate,arms,receipts = None,{},{}
    if commit:
        require(re.fullmatch('[0-9a-f]{40}',commit) is not None,'Pin the full round-six Git SHA.')
        transfer.rebound(transfer.base.common.git_state,PROJECT=PROJECT)(commit)
        terminal,wrapper,receipts = completed_training(subject,commit)
        candidate = review.planned_model(root/'run',PROJECT,wrapper['result_sha256'])
        parts = round1.delivery_paths(root/'run',terminal)
        require(parts[0].name == prior.FINAL_SNAPSHOT
                and all(candidate.provenance[key] == old.provenance[key] for key in prior.prior.IDENTITY_FIELDS),
                'The round-six training frames, frozen fit, camera or holdouts changed.')
        source_old,source_new = (part[2]['source_files_sha256'] for part in (old_parts,parts))
        trainer = 'src/shuolee_portrait/training.py'
        require(trainer in source_old and source_old.keys() == source_new.keys()
                and all(source_old[key] == source_new[key] for key in source_old if key != trainer),
                'The common replay renderer, camera or binding sources changed.')
        prior.check_frozen_mesh(old_parts,parts)
        for pair in locked['pairs']:
            frame = pair['new']['frame']
            require(review.pose_digest(candidate,frame) == pair['new']['pose_sha256'],'A matched native pose changed.')
        portable = static_worker.portable(candidate.provenance,front,parts)
        transform_fields = ('rotation','origin_world','scale','export_frame_mode','reference_frame_id','reference_pose_index')
        require(portable['front'] == locked['static_views']['static_reference_front']['camera']
                and all(portable['world_to_viewer'][key] == old_static['world_to_viewer'][key] for key in transform_fields),
                'The round-six portable camera or transform changed.')
        points = PlyData.read(portable['ply'])['vertex'].data
        xyz = np.stack([points[axis] for axis in ('x','y','z')],axis=1)
        portable['min_center_depth_m'] = {}
        for name,view in locked['static_views'].items():
            z = np.asarray(view['camera']['w2c'][2])
            depth = float((xyz@z[:3]+z[3]).min())
            require(np.isfinite(depth) and depth > .2,'The static PLY crosses the stock raster cutoff.')
            portable['min_center_depth_m'][name] = depth
        arms['round6'] = portable
    plan = dict(locked,old=old.provenance,old_commit=BASE_COMMIT,new=candidate.provenance if candidate else None,new_commit=commit,
                output=str(root/'view-comparison'),round6_reviewer_sha256=sha(Path(__file__)),baseline_review=str(previous),
                baseline_review_sha256=spec['review_sha256'],round5_plan_sha256=baseline['plan_sha256'],static_arms=arms)
    plan.update(receipts)
    return plan,baseline,rois


def assemble(output,plan,baseline,new,regions,static):
    panels = []
    records = {item['role']:item for item in new['records']}
    baseline_root = Path(plan['baseline_review']).parent
    labels = ['round5_9013c26',f'round6_{plan["new_commit"][:7]}']
    for item in baseline['panels']:
        name,scale = item['view'],item['scale']
        is_static = name in plan['static_views']
        if is_static:
            view = plan['static_views'][name]
            entry = static['round6'][name]
            prior.check_camera(entry['camera'],view['camera'])
            require(entry['static_ply_sha256'] == plan['static_arms']['round6']['ply_sha256']
                    and entry['pose_driven'] is False and entry['source_rgb_compared'] is False
                    and item['crop_xyxy'] == view['crop_xyxy'] and scale == 1.,
                    'A static review crop or actual PLY provenance changed.')
        else:
            entry,camera = prior.candidate_native(name,records,regions)
            prior.check_camera(camera,plan['views'][name.removesuffix('_head')]['camera'])
            require(item['crop_xyxy'] == entry['crop_xyxy'],'Matched native crops differ.')
        old_path,new_path = baseline_root/item['file'],output/entry['file']
        pin(old_path,item['sha256'])
        pin(new_path,entry['sha256'])
        require(item['header_pixels'] == 0 and item['rows'][-1] == labels[0],'The baseline panel layout changed.')
        with Image.open(old_path) as old,Image.open(new_path) as candidate:
            w,h = old.width,old.height//2
            require(old.height == 2*h and candidate.size == (w,h if is_static else 2*h) and scale in (1.,.5),
                    'Matched panel dimensions differ.')
            index = 0 if scale == 1. else 1
            sheet = Image.new('RGB',(w,2*h))
            sheet.paste(old.crop((0,h,w,2*h)),(0,0))
            sheet.paste(candidate.crop((0,index*h,w,(index+1)*h)),(0,h))
            destination = output/item['file']
            sheet.save(destination)
        panels.append(dict(item,sha256=sha(destination),rows=labels))
    require(len(panels) == 13,'Unexpected finite delivery panel count.')
    return panels


def main():
    if len(sys.argv) == 4 and sys.argv[1] == '_worker':
        path,fd = Path(sys.argv[2]),int(sys.argv[3])
        plan = read(path)
        pin(Path(__file__),plan['round6_reviewer_sha256'])
        pin(Path(static_worker.__file__),STATIC_WORKER_SHA)
        native.worker(path,'new',fd)
        static_worker.static_worker(path)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('check','render'))
    parser.add_argument('subject',choices=native.SCOPES)
    parser.add_argument('--commit')
    args = parser.parse_args()
    if args.action == 'check':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    elif not args.commit:
        parser.error('--commit must pin the exact completed round-six candidate')
    plan,baseline,rois = preflight(args.subject,args.commit)
    if args.action == 'check':
        print(json.dumps(dict(state='cpu_preflight_passed',terminal_ready=plan['new'] is not None,gpu_calls=0,
            frames=native.SCOPES[args.subject],static_views=list(plan['static_views']),
            physical_angular_coverage=plan['source_angular_support']['physical_angular_coverage'])),flush=True)
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
    review.save(result,dict(state='round6_delivery_views_complete',subject=args.subject,plan_sha256=sha(path),
        panels=panels,regions=plan['regions'],region_metrics=dict(baseline['region_metrics'],round6=regions),
        source_angular_support=plan['source_angular_support'],native_view_labels=plan['native_view_labels'],
        integration_receipt=plan['integration_receipt'],world_scale_receipt=plan['world_scale_receipt'],
        static_render_receipts=dict(round5=baseline['static_render_receipts']['round5'],round6=static['round6']),
        old=baseline['new'],new=new,optimizer_updates=0,refits=0,
        dolly_rgb_truth_available=False,static_rotated_oblique_rgb_truth_available=False,
        static_rotated_side_rgb_truth_available=False,quality_accepted=False))
    print(json.dumps(dict(result=str(result),sha256=sha(result),panels=len(panels))),flush=True)


if __name__ == '__main__':
    main()
