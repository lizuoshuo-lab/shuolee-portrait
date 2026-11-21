"""Render matched stock/common front, real oblique side and dolly delivery views."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_head_appearance_frozen_fit as transfer
import review_unified_cold_views as review
from review_bound_alpha_ab import check_source_equivalence, native_box, pin, read, require, sha


COMMIT = '762d6d0b2a5db804d8d4de94c53779b2a5eb80b7'
OLD_PROJECT = transfer.base.common.PROJECT
NEW_PROJECT = transfer.PROJECT
SPECS = {
    '016': ('a1042ccba8a4399a97497bedcd3c049fb15f698e434f746213e263340bebc696',
            'cd72ea19c41ad93655fa8a8d03c2ea105bf4a22e5ed28f5c72d234723f348317', 'frame_000027', 'frame_000051'),
    '006': ('18333c3592caa278dfdd9cb8c11ee3ec2b561e94688178587824b31208c6faf7',
            '881dce78b3f7aae03182631057f4b0456f1349bc40c0761be52be8d1a031db29', 'frame_000033', 'frame_000044'),
}


def delivery_paths(run, terminal):
    replay = run/terminal['bound_replay']
    pin(replay,terminal['bound_replay_sha256'])
    snapshot = replay.parents[1]
    for key,expected in (('ply',terminal['sha256']),('scene',terminal['scene_sha256']),
                         ('bound_panels',terminal['bound_panels_sha256'])):
        path = (run/terminal[key]).resolve()
        require(path.is_relative_to(run.resolve()),'Delivery points outside the run.')
        pin(path,expected)
    manifest = snapshot/'bound-replay.json'
    pin(manifest,read(replay)['bound_replay_sha256'])
    bound = read(manifest)
    pin(snapshot/'bound-local.ply',bound['bound_ply_sha256'])
    pin(snapshot/'world.ply',bound['inputs_sha256']['world_ply'])
    return snapshot,replay,bound


def preflight(subject):
    old_sha,view_sha,front,side = SPECS[subject]
    old_root = transfer.base.common.root_for(subject)
    review_root = review.LAB/f'shuolee-smpl-delivery-round1-{subject}'
    new_root = transfer.root_for('016') if subject == '016' else review_root
    old_run,new_run = old_root/'run',new_root/'run'
    transfer.base.common.git_state(transfer.base.COMMIT)
    transfer.check_source(COMMIT)
    wrapper = read(new_root/'wrapper-terminal.json')
    require(wrapper['candidate_commit'] == COMMIT and wrapper['state'] == 'completed','Wrong common completed delivery.')
    old = review.planned_model(old_run,OLD_PROJECT,old_sha)
    new = review.planned_model(new_run,NEW_PROJECT,wrapper['result_sha256'])
    require(all(old.provenance[key] == new.provenance[key] for key in ('fitted_sha256','training_inputs_sha256',
                'protected_holdouts','reference_frame_id')) and old.provenance['reference_frame_id'] == front,
            'The old and common deliveries do not use the same fitted observations.')
    terminal = {key:read(run/'result.json') for key,run in (('old',old_run),('new',new_run))}
    require(terminal['new']['fit_reused'] is True and terminal['new']['fit_calls'] == 0,
            'Common endpoint must reuse the verified named pose and mesh.')
    artifacts = {key:delivery_paths(run,terminal[key]) for key,run in (('old',old_run),('new',new_run))}
    check_source_equivalence(OLD_PROJECT,NEW_PROJECT,*(artifact[2]['source_files_sha256'] for artifact in artifacts.values()))
    archive = old_root/'view-comparison/result.json'
    pin(archive,view_sha)
    cached = read(archive)
    archive_plan = archive.with_name('plan.json')
    pin(archive_plan,cached['plan_sha256'])
    require(read(archive_plan)['new']['terminal_sha256'] == old_sha,'Saved dolly belongs to a different stock endpoint.')
    source_records = {row['frame']:row for row in cached['new']['records']}
    close = source_records[front]['closeup']
    near = close['camera']
    require(close['kind'] == 'fixed_focal_dolly' and all(near[key] == old.rows[front][key] for key in ('K','wh','frame')),
            'Saved close camera is not the fixed-K same-pose dolly.')
    pin(archive.parent/close['panel']['file'],close['panel']['sha256'])
    pairs = []
    for label,frame in (('front',front),('side',side)):
        left,right = old.rows[frame],new.rows[frame]
        require(all(left[key] == right[key] for key in (*review.NATIVE_KEYS,'mask_sha256'))
                and all(source_records[frame]['native_camera'][key] == left[key] for key in ('frame','wh','K','w2c'))
                and review.pose_digest(old,frame) == review.pose_digest(new,frame), 'Native camera, source or fitted pose changed.')
        pin(Path(left['image']),left['source_sha256'])
        pin(Path(left['mask']),left['mask_sha256'])
        x0,y0,x1,y1 = left['initial_box_xyxy']
        common = dict(frame=frame,crop=native_box([x0-64,y0-64,x1+64,y1+64],left['wh']),score_mask=left['mask'],
                      score_mask_sha256=left['mask_sha256'],pose_sha256=review.pose_digest(old,frame))
        if label == 'front':
            common['dolly'] = dict(w2c=near['w2c'],crop=close['panel']['crop_xyxy'])
        pairs.append(dict(label=label,exact_native_source_and_camera=True,exact_fitted_pose=True,
                          old=dict(common),new=dict(common)))
    depth = {}
    for key,(snapshot,_,_) in artifacts.items():
        points = PlyData.read(snapshot/'world.ply')['vertex'].data
        xyz = np.stack([points[name] for name in ('x','y','z')],axis=1)
        depth[key] = float((xyz@np.asarray(near['w2c'][2][:3])+near['w2c'][2][3]).min())
        require(np.isfinite(depth[key]) and depth[key] > .2,'Dolly intersects the renderer center cutoff.')
    return dict(schema='shuolee-round1-delivery-views-v1',subject=subject,asset=str(review.ASSET),
        code_sha256=sha(Path(__file__)),old=old.provenance,new=new.provenance,old_commit=transfer.base.COMMIT,
        new_commit=COMMIT,output=str(review_root/'view-comparison'),pairs=pairs,
        frozen_dolly_receipt_sha256=view_sha,cpu_dolly_min_center_depth_m=depth,
        final_snapshots={key:entry[0].name for key,entry in artifacts.items()},
        new_head_appearance=terminal['new'].get('head_appearance'),dolly_rgb_truth_available=False,
        optimizer_updates=0,refits=0,quality_accepted=False)


def delivered(run,asset,rows,holdouts):
    from shuolee_portrait.bound_replay import _inputs

    terminal = read(run/'result.json')
    snapshot,replay_path,bound = delivery_paths(run,terminal)
    replay = read(replay_path)
    fit,mesh,facts = _inputs(run,snapshot,asset,bound)
    require(replay['state'] == 'completed' and replay['optimizer_updates'] == 0
            and replay['retained_holdouts'] == holdouts and len(rows) == len(replay['records']) == len(facts['frames']),
            'Native named replay does not bind the expected snapshot roster.')
    for row,record,identity in zip(rows,replay['records'],facts['frames'],strict=True):
        require(all(record[key] == value for key,value in identity.items())
                and all(record['camera'][key] == row[key] for key in ('frame','K','w2c','wh')),
                'Native archive camera or pose identity changed.')
        pin(replay_path.parent/record['image'],record['image_sha256'])
    return SimpleNamespace(snapshot=snapshot,fit=fit,mesh=mesh,facts=facts,bound=bound,replay=replay)


def panel(path,crop,pixels,source=None,mask=None):
    x0,y0,x1,y1 = crop
    width,height = x1-x0,y1-y0
    columns = 4 if source is not None else 3
    image = Image.new('RGB',(columns*width,2*height))
    for row,views in enumerate(pixels):
        cells = (source,*views) if source is not None else views
        for column,data in enumerate(cells):
            tile = data[y0:y1,x0:x1]
            if tile.dtype != np.uint8:
                tile = np.rint(np.clip(tile,0,1)*255).astype(np.uint8)
            image.paste(Image.fromarray(tile).convert('RGB'),(column*width,row*height))
    image.save(path)
    return dict(file=path.name,sha256=sha(path),crop_xyxy=list(crop),header_pixels=0)


def merge(output,left,right):
    panels = []
    for old,new in zip(left['records'],right['records'],strict=True):
        require(old['frame'] == new['frame'] and old['native_camera'] == new['native_camera']
                and old['pose_sha256'] == new['pose_sha256'],'Rendered arm camera or pose changed.')
        entries = [(old['role'],old['native_panel'],new['native_panel'],True)]
        if old['role'] == 'front':
            near_old,near_new = old['closeup'],new['closeup']
            require(near_old['kind'] == near_new['kind'] == 'fixed_focal_dolly' and near_old['camera'] == near_new['camera'],
                    'No exact delivered dolly comparison.')
            entries.append(('dolly',near_old['panel'],near_new['panel'],False))
        for view,a,b,source in entries:
            pin(output/a['file'],a['sha256']); pin(output/b['file'],b['sha256'])
            require(a['crop_xyxy'] == b['crop_xyxy'],'Arm comparison crops differ.')
            with Image.open(output/a['file']) as stock, Image.open(output/b['file']) as common:
                require(stock.size == common.size and stock.height % 2 == 0,'Wrong scale panel sizes.')
                width,height = stock.width,stock.height//2
                for index,scale in enumerate((1.,.5)):
                    crop = (0,index*height,width,(index+1)*height)
                    merged = Image.new('RGB',(width,2*height))
                    merged.paste(stock.crop(crop),(0,0)); merged.paste(common.crop(crop),(0,height))
                    path = output/f'{view}-{old["frame"]}-scale-{scale:.1f}.png'
                    merged.save(path)
                    panels.append(dict(file=path.name,sha256=sha(path),frame=old['frame'],view=view,scale=scale,
                        header_pixels=0,crop_xyxy=a['crop_xyxy'],rows=['stock_79b906f','common_762d6d0'],
                        columns=(['source_rgb'] if source else [])+['black','white','alpha']))
    return panels


def main():
    if len(sys.argv) == 5 and sys.argv[1] == '_worker':
        path,side,fd = Path(sys.argv[2]),sys.argv[3],int(sys.argv[4])
        plan = read(path)
        pin(Path(__file__),plan['code_sha256'])
        sys.path.insert(0,str(Path(plan[side]['project'])/'src'))
        from shuolee_portrait.cli import gpu_lock
        with gpu_lock():
            transfer.rebound(review.worker,delivered=delivered,panel=panel)(path,side,fd)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('check','render'))
    parser.add_argument('subject',choices=SPECS)
    args = parser.parse_args()
    if args.action == 'check':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    plan = preflight(args.subject)
    if args.action == 'check':
        print(json.dumps(dict(state='cpu_preflight_passed',gpu_calls=0,output=plan['output'],
              final_snapshots=plan['final_snapshots'],poses=[row['old']['frame'] for row in plan['pairs']],
              dolly_min_depth_m=plan['cpu_dolly_min_center_depth_m'])),flush=True)
        return
    output = Path(plan['output'])
    with review.GPU_LEASE.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        output.mkdir(parents=True,exist_ok=False)
        path = output/'plan.json'
        review.save(path,plan)
        for side in ('old','new'):
            with (output/f'{side}.log').open('x',encoding='utf-8') as log:
                subprocess.run([sys.executable,'-I','-B',__file__,'_worker',str(path),side,str(lease.fileno())],
                               pass_fds=(lease.fileno(),),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
    left,right = [read(output/f'{side}-result.json') for side in ('old','new')]
    panels = merge(output,left,right)
    result = output/'result.json'
    review.save(result,dict(state='delivery_views_complete',subject=args.subject,plan_sha256=sha(path),old=left,new=right,
                            panels=panels,optimizer_updates=0,refits=0,dolly_rgb_truth_available=False,quality_accepted=False))
    print(json.dumps(dict(result=str(result),sha256=sha(result),panels=len(panels))),flush=True)


if __name__ == '__main__':
    main()
