"""Compare round eight against the exact frozen round-seven review cameras."""

import argparse
import ast
from hashlib import sha256
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
import review_smpl_delivery_round7 as prior


review,round1,transfer = prior.review,prior.round1,prior.transfer
read,pin,require,sha = prior.read,prior.pin,prior.require,prior.sha
native,static_worker = prior.native,prior.static_worker
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-support-20261111')
BASE_COMMIT = 'dd99e967e9c8f146b99a45df0855f3f7f4bade78'
CANDIDATE_COMMIT = '2793338736d2957325f54023d4cbabbdc2e1aa99'
BASE_RECEIPT = Path(__file__).resolve().parents[1]/'outputs/smpl-delivery-round7/receipt.json'
BASE_RECEIPT_SHA = '465196daa6a388809b8c57b2f050a6ea97f5f59141cedfd7c53e0a9a400f7f4a'
BASE_REVIEWER_SHA = '5dec43f8516561825bc0dfdc6337cd14f8cd99ed89ec534400efd0fb98531e14'
FINAL_SNAPSHOT = prior.FINAL_SNAPSHOT
base_review = prior.prior.prior
TRAINER = 'src/shuolee_portrait/training.py'
MODEL_HOOKS = {
    'src/avatar_core/scene/gaussian_model.py': ('GaussianModel',{'prune_points','densification_postfix','densify_and_split','densify_and_clone'}),
    'src/avatar_core/scene/smplx_gaussian_model.py': ('SMPLXGaussianModel',{'densify_and_split'}),
}
OPTIONAL_HOOK_FILES = {TRAINER,*MODEL_HOOKS}
HOOK_DIFF_SHA256 = {
    'src/avatar_core/scene/gaussian_model.py': 'c0bb92197e28fd1a65075ca49b93b07162d98a6faa3a22c5b6ce9ee4c2088fd6',
    'src/avatar_core/scene/smplx_gaussian_model.py': '8475385566223a9cc6154437037690ca7fb2660f752163a113fa704974c1a41c',
    TRAINER: 'b65460b194b9d36da66d5095905ba207c12dbca774e5d65550599c3c0ba97924',
}
DIFF_FLAGS = ('diff','--no-ext-diff','--no-textconv','--no-color','--no-renames','--no-indent-heuristic',
              '--diff-algorithm=myers','--full-index','--src-prefix=a/','--dst-prefix=b/')


def root_for(subject):
    return review.LAB/f'shuolee-smpl-delivery-round8-{subject}'


read_training = transfer.rebound(prior.prior.completed_training,root_for=root_for)


def completed_training(subject,commit):
    terminal,wrapper,receipts = read_training(subject,commit)
    scale = read(Path(receipts['world_scale_receipt']['file']))
    require(bool(scale['terminal']['needle']) and bool(scale['terminal']['render_support'])
            and scale['splits'] and all(row['render_support'] for row in scale['splits']),
            'The round-eight world-scale receipt lacks needle or render-support evidence.')
    return terminal,wrapper,receipts


def git_command(project):
    pointer = (project/'.git').read_text(encoding='utf-8').strip()
    require(pointer.startswith('gitdir: '),'Expected a pinned Windows worktree.')
    directory = subprocess.check_output(['/usr/bin/wslpath','-u',pointer.removeprefix('gitdir: ')],text=True).strip()
    return ['/usr/bin/git','--git-dir',directory]


def source_receipt(project,commit,name,expected):
    path = project/name
    pin(path,expected)
    git = git_command(project)
    reference = f'{commit}:{name}'
    blob = subprocess.check_output([*git,'rev-parse',reference],text=True).strip()
    content = subprocess.check_output([*git,'cat-file','blob',blob])
    require(path.read_bytes().replace(b'\r\n',b'\n') == content.replace(b'\r\n',b'\n'),
            f'Checkout bytes differ from committed source: {name}')
    return blob,content


def trainer_scope(content):
    content = content.replace(b'\r\n',b'\n')
    changed = [node for node in ast.parse(content).body if isinstance(node,ast.FunctionDef) and node.name == '_train_smpl_sh0']
    require(len(changed) == 1,'Expected the exact SH0 training entry.')
    entry, = changed
    lines = content.splitlines(keepends=True)
    return b''.join(lines[:entry.lineno-1]+lines[entry.end_lineno:]),ast.dump(entry)


def model_replay_scope(content,name):
    class_name, allowed = MODEL_HOOKS[name]
    content = content.replace(b'\r\n',b'\n')
    owner, = (node for node in ast.parse(content).body if isinstance(node,ast.ClassDef) and node.name == class_name)
    lines = content.splitlines(keepends=True)
    for entry in reversed(owner.body):
        if isinstance(entry,ast.FunctionDef) and entry.name in allowed:
            del lines[entry.lineno-1:entry.end_lineno]
    return b''.join(lines)


def replay_source_parity(old,new,commit):
    require(commit == CANDIDATE_COMMIT and old.keys() == new.keys() and OPTIONAL_HOOK_FILES.issubset(old),
            'The protected bound replay source manifest keys changed.')
    git = git_command(PROJECT)
    diff = [*git,*DIFF_FLAGS]
    changed = set(subprocess.check_output([*diff,'--name-only',BASE_COMMIT,commit,'--',*sorted(old)],text=True).splitlines())
    require(changed and changed == set(HOOK_DIFF_SHA256) and changed.issubset(OPTIONAL_HOOK_FILES),
            'The candidate changed a replay source outside the exact registered optional hook set.')
    receipts = {}
    for name in sorted(old):
        if name not in changed and old[name] == new[name]:
            continue
        left_blob,left = source_receipt(prior.PROJECT,BASE_COMMIT,name,old[name])
        right_blob,right = source_receipt(PROJECT,commit,name,new[name])
        item = dict(round7_blob=left_blob,round8_blob=right_blob,round7_raw_sha256=old[name],round8_raw_sha256=new[name])
        if name in changed:
            patch = subprocess.check_output([*diff,'--unified=0',BASE_COMMIT,commit,'--',name])
            digest = sha256(patch).hexdigest()
            require(left_blob != right_blob and digest == HOOK_DIFF_SHA256[name],f'Unregistered optional hook diff: {name}')
            if name == TRAINER:
                left_other,left_fit = trainer_scope(left)
                right_other,right_fit = trainer_scope(right)
                require(left_other == right_other and left_fit != right_fit,'The SH0 context hook changed other trainer code.')
            else:
                require(model_replay_scope(left,name) == model_replay_scope(right,name),
                        f'Optional topology hooks changed non-topology model code: {name}')
            item.update(match='registered_optional_hook',git_diff_sha256=digest)
        else:
            require(left_blob == right_blob and left == right,f'Protected source changed: {name}')
            item.update(match='git_blob_equal_checkout_line_endings')
        receipts[name] = item
    return dict(match='protected_replay_exact_optional_hooks_pinned',files=receipts,
                protected_raw_files=len(old)-len(receipts),candidate_replay_requires_native_pixel_parity=True)


def preflight(subject,commit=None):
    pin(BASE_RECEIPT,BASE_RECEIPT_SHA)
    pin(Path(prior.__file__),BASE_REVIEWER_SHA)
    pin(Path(static_worker.__file__),prior.prior.STATIC_WORKER_SHA)
    delivery = read(BASE_RECEIPT)
    require(delivery['commit'] == BASE_COMMIT,'Unexpected round-seven delivery.')
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
    require(baseline['state'] == 'round7_delivery_views_complete' and locked['new_commit'] == BASE_COMMIT
            and locked['round7_reviewer_sha256'] == BASE_REVIEWER_SHA
            and all(locked[key] == baseline[key] for key in ('regions','source_angular_support','native_view_labels'))
            and locked['source_angular_support']['physical_angular_coverage'] == 'unknown'
            and tuple(pair['new']['frame'] for pair in locked['pairs']) == native.SCOPES[subject],
            'The round-seven frozen camera, ROI or angular evidence differs.')
    for panel in baseline['panels']:
        pin(previous.parent/panel['file'],panel['sha256'])
    old_run = Path(spec['result']).parent
    old = review.planned_model(old_run,prior.PROJECT,spec['result_sha256'])
    old_parts = round1.delivery_paths(old_run,read(old_run/'result.json'))
    front = locked['views']['front']['camera']
    old_static = static_worker.portable(old.provenance,front,old_parts)
    require(old.provenance == locked['new'] and old_parts[0].name == FINAL_SNAPSHOT
            and old_static['ply_sha256'] == locked['static_arms']['round7']['ply_sha256'],
            'The round-seven endpoint changed.')
    root = root_for(subject)
    candidate,arms,receipts,source_parity = None,{},{},None
    if commit:
        require(re.fullmatch('[0-9a-f]{40}',commit) is not None,'Pin the full round-eight Git SHA.')
        transfer.rebound(transfer.base.common.git_state,PROJECT=PROJECT)(commit)
        terminal,wrapper,receipts = completed_training(subject,commit)
        candidate = review.planned_model(root/'run',PROJECT,wrapper['result_sha256'])
        parts = round1.delivery_paths(root/'run',terminal)
        require(parts[0].name == FINAL_SNAPSHOT
                and all(candidate.provenance[key] == old.provenance[key] for key in base_review.prior.IDENTITY_FIELDS),
                'The round-eight training frames, frozen fit, camera or holdouts changed.')
        source_parity = replay_source_parity(old_parts[2]['source_files_sha256'],parts[2]['source_files_sha256'],commit)
        base_review.check_frozen_mesh(old_parts,parts)
        for pair in locked['pairs']:
            frame = pair['new']['frame']
            require(review.pose_digest(candidate,frame) == pair['new']['pose_sha256'],'A matched native pose changed.')
        portable = static_worker.portable(candidate.provenance,front,parts)
        transform_fields = ('rotation','origin_world','scale','export_frame_mode','reference_frame_id','reference_pose_index')
        require(portable['front'] == locked['static_views']['static_reference_front']['camera']
                and all(portable['world_to_viewer'][key] == old_static['world_to_viewer'][key] for key in transform_fields),
                'The round-eight portable camera or transform changed.')
        points = PlyData.read(portable['ply'])['vertex'].data
        xyz = np.stack([points[axis] for axis in ('x','y','z')],axis=1)
        portable['min_center_depth_m'] = {}
        for name,view in locked['static_views'].items():
            z = np.asarray(view['camera']['w2c'][2])
            depth = float((xyz@z[:3]+z[3]).min())
            require(np.isfinite(depth) and depth > .2,'The static PLY crosses the stock raster cutoff.')
            portable['min_center_depth_m'][name] = depth
        arms['round8'] = portable
    plan = dict(locked,old=old.provenance,old_commit=BASE_COMMIT,new=candidate.provenance if candidate else None,new_commit=commit,
                output=str(root/'view-comparison'),round8_reviewer_sha256=sha(Path(__file__)),baseline_review=str(previous),
                baseline_review_sha256=spec['review_sha256'],round7_plan_sha256=baseline['plan_sha256'],static_arms=arms,
                replay_source_parity=source_parity)
    plan.update(receipts)
    return plan,baseline,rois


def assemble(output,plan,baseline,new,regions,static):
    panels = []
    records = {item['role']:item for item in new['records']}
    require(set(records) == {'front','side'} and all(item['native_archive_max_abs_8bit'] <= 1 for item in records.values()),
            'The candidate native replay did not reproduce its exact delivered poses.')
    baseline_root = Path(plan['baseline_review']).parent
    labels = ['round7_dd99e96',f'round8_{plan["new_commit"][:7]}']
    for item in baseline['panels']:
        name,scale = item['view'],item['scale']
        is_static = name in plan['static_views']
        if is_static:
            view = plan['static_views'][name]
            entry = static['round8'][name]
            base_review.check_camera(entry['camera'],view['camera'])
            require(entry['static_ply_sha256'] == plan['static_arms']['round8']['ply_sha256']
                    and entry['pose_driven'] is False and entry['source_rgb_compared'] is False
                    and item['crop_xyxy'] == view['crop_xyxy'] and scale == 1.,
                    'A static review crop or actual PLY provenance changed.')
        else:
            entry,camera = base_review.candidate_native(name,records,regions)
            base_review.check_camera(camera,plan['views'][name.removesuffix('_head')]['camera'])
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
        pin(Path(__file__),plan['round8_reviewer_sha256'])
        pin(Path(static_worker.__file__),prior.prior.STATIC_WORKER_SHA)
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
        parser.error('--commit must pin the exact completed round-eight candidate')
    plan,baseline,rois = preflight(args.subject,args.commit)
    if args.action == 'check':
        print(json.dumps(dict(state='cpu_preflight_passed',terminal_ready=plan['new'] is not None,gpu_calls=0,
            frames=native.SCOPES[args.subject],static_views=list(plan['static_views']),
            physical_angular_coverage=plan['source_angular_support']['physical_angular_coverage'],
            replay_source_parity=plan['replay_source_parity'])),flush=True)
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
    review.save(result,dict(state='round8_delivery_views_complete',subject=args.subject,plan_sha256=sha(path),
        panels=panels,regions=plan['regions'],region_metrics=dict(baseline['region_metrics'],round8=regions),
        source_angular_support=plan['source_angular_support'],native_view_labels=plan['native_view_labels'],
        replay_source_parity=plan['replay_source_parity'],
        integration_receipt=plan['integration_receipt'],world_scale_receipt=plan['world_scale_receipt'],
        static_render_receipts=dict(round7=baseline['static_render_receipts']['round7'],round8=static['round8']),
        old=baseline['new'],new=new,optimizer_updates=0,refits=0,
        dolly_rgb_truth_available=False,static_rotated_oblique_rgb_truth_available=False,
        static_rotated_side_rgb_truth_available=False,quality_accepted=False))
    print(json.dumps(dict(result=str(result),sha256=sha(result),panels=len(panels))),flush=True)


if __name__ == '__main__':
    main()
