"""Compare round five with the exact matched native and static round-four views."""

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
import review_smpl_delivery_round4 as prior


review, round1, transfer = prior.review, prior.round1, prior.transfer
read, pin, require, sha = prior.read, prior.pin, prior.require, prior.sha
native = prior.prior.prior
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-main-20261111')
ROUND4_COMMIT = 'ac2cfe68379de9f16a35784f2320fefcb88843e8'
ROUND4_REVIEWER_SHA = '432dc641732942d0f1fdb949a7777b0e421741a2f9ab99d09f93ed1e837f4e5a'
ROUND4_RECEIPT = Path(__file__).resolve().parents[1]/'outputs/smpl-delivery-round4/receipt.json'
ROUND4_RECEIPT_SHA = '9526e9cf8cfe01a662c7e0b9dcd1243537a1114278adca298c154c71e1a4ba99'
FINAL_SNAPSHOT = 'step-010000'


def check_camera(actual,expected):
    require(all(actual[key] == expected[key] for key in ('frame','K','w2c','wh')),
            'A rendered native or static camera changed.')


def check_frozen_mesh(old_parts,new_parts):
    meshes = []
    for snapshot,_,bound in (old_parts,new_parts):
        path = snapshot/'reference-mesh.npz'
        pin(path,bound['inputs_sha256']['reference_mesh'])
        with np.load(path,allow_pickle=False) as archive:
            meshes.append(dict(archive))
    old,new = meshes
    require(old.keys() == new.keys() and all(np.array_equal(old[key],new[key]) for key in old),
            'Round five changed the exact mesh-only initialization reference.')


def preflight(subject,commit=None):
    pin(ROUND4_RECEIPT,ROUND4_RECEIPT_SHA)
    pin(Path(prior.__file__),ROUND4_REVIEWER_SHA)
    receipt = read(ROUND4_RECEIPT)
    require(receipt['commit'] == ROUND4_COMMIT, 'Unexpected round-four delivery.')
    spec = receipt['subjects'][subject]
    previous = Path(spec['review'])
    pin(previous,spec['review_sha256'])
    baseline = read(previous)
    path = previous.with_name('plan.json')
    pin(path,baseline['plan_sha256'])
    locked = read(path)
    rois = path.with_name('rois.npz')
    pin(rois,locked['roi_archive_sha256'])
    pin(Path(native.__file__),locked['reviewer_sha256'])
    pin(Path(prior.prior.__file__),prior.ROUND3_REVIEWER_SHA)
    require(baseline['state'] == 'round4_delivery_views_complete'
            and locked['round4_reviewer_sha256'] == ROUND4_REVIEWER_SHA and locked['new_commit'] == ROUND4_COMMIT
            and all(locked[key] == baseline[key] for key in ('regions','source_angular_support','native_view_labels'))
            and tuple(pair['new']['frame'] for pair in locked['pairs']) == native.SCOPES[subject]
            and locked['source_angular_support']['physical_angular_coverage'] == 'unknown',
            'The cached native poses, ROI or fitted angular support changed.')
    old_run = Path(spec['result']).parent
    old = review.planned_model(old_run,prior.PROJECT,spec['result_sha256'])
    old_parts = round1.delivery_paths(old_run,read(old_run/'result.json'))
    old_static = prior.prior.portable(old.provenance,locked['views']['front']['camera'],old_parts)
    require(locked['new'] == old.provenance and old_parts[0].name == prior.FINAL_SNAPSHOT
            and old_static['ply_sha256'] == locked['static_arms']['round4']['ply_sha256'],
            'The locked round-four endpoint changed.')
    for item in baseline['panels']:
        pin(previous.parent/item['file'],item['sha256'])
    output = review.LAB/f'shuolee-smpl-delivery-round5-{subject}'
    candidate, integration, arms = None, None, {}
    if commit:
        require(re.fullmatch('[0-9a-f]{40}',commit) is not None, 'Pin a full candidate Git SHA.')
        transfer.rebound(transfer.base.common.git_state,PROJECT=PROJECT)(commit)
        wrapper = read(output/'wrapper-terminal.json')
        require(wrapper['state'] == 'completed' and wrapper['candidate_commit'] == commit,'Round-five wrapper is incomplete.')
        run = output/'run'
        candidate = review.planned_model(run,PROJECT,wrapper['result_sha256'])
        terminal = read(run/'result.json')
        parts = round1.delivery_paths(run,terminal)
        integration_path = run/'gaussian-run/integration.json'
        integration_digest = sha(integration_path)
        require(bool(terminal['unified_main_training']) and terminal['unified_main_training'] == read(integration_path)
                and terminal['gaussian_from_scratch'] is True,
                'The native 10k result does not bind the completed unified main training receipt.')
        integration = dict(file=str(integration_path),sha256=integration_digest)
        require(parts[0].name == terminal['final_snapshot'] == FINAL_SNAPSHOT
                and terminal['fit_reused'] is True and terminal['fit_calls'] == 0
                and all(candidate.provenance[key] == old.provenance[key] for key in prior.IDENTITY_FIELDS),
                'Round five changed the frozen fit, camera, holdouts or expected final snapshot.')
        renderer_old = old_parts[2]['source_files_sha256']
        renderer_new = parts[2]['source_files_sha256']
        trainer = 'src/shuolee_portrait/training.py'
        require(trainer in renderer_old and renderer_old.keys() == renderer_new.keys()
                and all(renderer_old[key] == renderer_new[key] for key in renderer_old if key != trainer),
                'The common replay renderer, camera or binding sources changed.')
        check_frozen_mesh(old_parts,parts)
        for pair in locked['pairs']:
            frame = pair['new']['frame']
            require(review.pose_digest(candidate,frame) == pair['new']['pose_sha256'],'Round five changed a locked review pose.')
        portable = prior.prior.portable(candidate.provenance,locked['views']['front']['camera'],parts)
        transform_fields = ('rotation','origin_world','scale','export_frame_mode','reference_frame_id','reference_pose_index')
        require(portable['front'] == locked['static_views']['static_reference_front']['camera']
                and all(portable['world_to_viewer'][key] == old_static['world_to_viewer'][key] for key in transform_fields),
                'Round five changed the portable reference camera or transform.')
        points = PlyData.read(portable['ply'])['vertex'].data
        xyz = np.stack([points[axis] for axis in ('x','y','z')],axis=1)
        portable['min_center_depth_m'] = {}
        for name,view in locked['static_views'].items():
            z = np.asarray(view['camera']['w2c'][2])
            depth = float((xyz@z[:3]+z[3]).min())
            require(np.isfinite(depth) and depth > .2, 'The static PLY crosses the stock raster cutoff.')
            portable['min_center_depth_m'][name] = depth
        arms['round5'] = portable
    plan = dict(locked,old=old.provenance,old_commit=ROUND4_COMMIT,new=candidate.provenance if candidate else None,new_commit=commit,
                output=str(output/'view-comparison'),round5_reviewer_sha256=sha(Path(__file__)),
                baseline_review=str(previous),baseline_review_sha256=spec['review_sha256'],round4_plan_sha256=baseline['plan_sha256'],
                static_arms=arms,integration_receipt=integration)
    return plan,baseline,rois


def candidate_native(name,records,regions):
    source = name.removesuffix('_head')
    if name == 'dolly':
        entry = records['front']['closeup']
        return entry['panel'],entry['camera']
    record = records[source]
    panel = regions['head_panels'][source] if name.endswith('_head') else record['native_panel']
    return panel,record['native_camera']


def assemble(output,plan,previous,new,regions,static):
    panels = []
    records = {item['role']:item for item in new['records']}
    baseline_root = Path(plan['baseline_review']).parent
    names = ['round4_ac2cfe6',f'round5_{plan["new_commit"][:7]}']
    for item in previous['panels']:
        name,scale = item['view'],item['scale']
        is_static = name in plan['static_views']
        if is_static:
            view = plan['static_views'][name]
            entry = static['round5'][name]
            check_camera(entry['camera'],view['camera'])
            require(entry['static_ply_sha256'] == plan['static_arms']['round5']['ply_sha256']
                    and entry['pose_driven'] is False and entry['source_rgb_compared'] is False
                    and item['crop_xyxy'] == view['crop_xyxy'] and scale == 1.,
                    'A static review crop or actual PLY provenance changed.')
        else:
            entry,camera = candidate_native(name,records,regions)
            check_camera(camera,plan['views'][name.removesuffix('_head')]['camera'])
            require(item['crop_xyxy'] == entry['crop_xyxy'], 'Matched native crops differ.')
        old_path,new_path = baseline_root/item['file'],output/entry['file']
        pin(old_path,item['sha256'])
        pin(new_path,entry['sha256'])
        require(item['header_pixels'] == 0 and item['rows'][-1] == names[0], 'The baseline panel layout changed.')
        with Image.open(old_path) as old,Image.open(new_path) as candidate:
            w,h = old.width,old.height//3
            require(old.height == 3*h and candidate.size == (w,h if is_static else 2*h) and scale in (1.,.5),
                    'Matched panel dimensions differ.')
            index = 0 if scale == 1. else 1
            sheet = Image.new('RGB',(w,2*h))
            sheet.paste(old.crop((0,2*h,w,3*h)),(0,0))
            sheet.paste(candidate.crop((0,index*h,w,(index+1)*h)),(0,h))
            destination = output/item['file']
            sheet.save(destination)
        panels.append(dict(item,sha256=sha(destination),rows=names))
    require(len(panels) == 13,'Unexpected finite delivery panel count.')
    return panels


def main():
    if len(sys.argv) == 4 and sys.argv[1] == '_worker':
        path,fd = Path(sys.argv[2]),int(sys.argv[3])
        plan = read(path)
        pin(Path(__file__),plan['round5_reviewer_sha256'])
        pin(Path(prior.prior.__file__),prior.ROUND3_REVIEWER_SHA)
        native.worker(path,'new',fd)
        prior.prior.static_worker(path)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('check','render'))
    parser.add_argument('subject',choices=native.SCOPES)
    parser.add_argument('--commit')
    args = parser.parse_args()
    if args.action == 'check':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    elif not args.commit:
        parser.error('--commit must pin the exact completed round-five candidate')
    plan,baseline,rois = preflight(args.subject,args.commit)
    if args.action == 'check':
        support = plan['source_angular_support']
        print(json.dumps(dict(state='cpu_preflight_passed',terminal_ready=plan['new'] is not None,gpu_calls=0,
            frames=native.SCOPES[args.subject],static_views=list(plan['static_views']),
            physical_angular_coverage=support['physical_angular_coverage'],
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
    review.save(result,dict(state='round5_delivery_views_complete',subject=args.subject,plan_sha256=sha(path),
        panels=panels,regions=plan['regions'],region_metrics=dict(baseline['region_metrics'],round5=regions),
        source_angular_support=plan['source_angular_support'],native_view_labels=plan['native_view_labels'],
        integration_receipt=plan['integration_receipt'],
        static_render_receipts=dict(round4=baseline['static_render_receipts']['round4'],round5=static['round5']),
        old=baseline['new'],new=new,optimizer_updates=0,refits=0,
        dolly_rgb_truth_available=False,static_rotated_oblique_rgb_truth_available=False,
        static_rotated_side_rgb_truth_available=False,quality_accepted=False))
    print(json.dumps(dict(result=str(result),sha256=sha(result),panels=len(panels))),flush=True)


if __name__ == '__main__':
    main()
