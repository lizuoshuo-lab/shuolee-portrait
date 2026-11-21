"""Archive both R24 driven/static endpoints, with native original-pose image records."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from PIL import Image
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
import review_smpl_delivery_round24 as review
import archive_smpl_delivery_round23 as prior


qa = review.qa
pin,checked,read,sha,require,descriptor = review.pin,review.checked,review.read,review.sha,review.require,review.descriptor
TITLE = ('Original native RGB','R23 18,000','R24 10,000','R24 18,000','Frozen SMPL-X guide','2D input / fit')
COLS = ('原图','R23 18k','R24 10k','R24 18k','冻结 Mesh 投影','2D 特征与投影')


def endpoint(out,plan,name):
    drive = plan['drives'][name]
    portable_path = checked(drive['portable'])
    snapshot = portable_path.parent
    portable = read(portable_path)
    arm = plan['static_grid']['arms'][name]
    mesh = prior.verify_scene(snapshot,dict(portable,reference_frame_id=arm['ref']))
    ply = pin(arm['ply'],arm['ply_sha256'])
    camera = pin(arm['camera_path'],arm['camera_sha256'])
    camera_provenance = camera.with_name('camera_provenance.json')
    scene = checked(drive['scene'])
    package = checked(drive['package'])
    graph = checked(plan['graphs'][name]['parameters'])
    target = out/f"duration-{drive['step']}"
    target.mkdir(parents=True,exist_ok=True)
    paths = dict(portable_ply=ply,camera=camera,camera_provenance=camera_provenance,scene=scene,
                 drive=package,graph=graph)
    files = prior.copy_proof(target,{key:(path,sha(path)) for key,path in paths.items()})
    return dict(files,step=drive['step'],points=drive['points'],free_clothing=drive['clothing_points'],
                reference_frame=arm['ref'],reference_mesh=descriptor(mesh),native_replay=drive['replay'])


def native_records(out,plan,result):
    baseline = read(checked(plan['r23_archive']))
    saved = {}
    out.mkdir(exist_ok=True)
    paste = review.rebound(prior.paste_panel,TITLE=TITLE)
    for slot,row in enumerate(plan['frozen']['rows']):
        frame = row['frame']
        previous = baseline['native_full'][frame]
        require(previous['source_sha256']==row['source_sha256']
                and previous['role']==('selected_fit' if slot<24 else 'holdout'),
                'R23 archived native reference used another frame.')
        panel = checked(previous)
        same = previous['crop_xyxy']
        bounds = [plan['drives'][name]['records'][slot]['box_xyxy'] for name in review.NEW]
        crop = prior.crop_overlap(same,*bounds)
        original,old,mesh,features = (prior.crop_file(panel,crop,same,col=index,header=26) for index in (0,3,4,5))
        source = prior.crop_file(pin(row['image'],row['source_sha256']),crop)
        require(np.array_equal(np.asarray(original),np.asarray(source)),'The native source RGB crop changed.')
        early,late = (prior.drive_image(plan,result,name,slot,crop) for name in review.NEW)
        record = paste(out/(frame+'.png'),(original,old,early,late,mesh,features))
        saved[frame] = dict(record,frame=frame,role=previous['role'],source_sha256=row['source_sha256'],
                            crop_xyxy=crop,columns=COLS)
    return saved


def evaluation(out, plan, result_path, result, native, curve, audits):
    files = {'review.json':descriptor(result_path),'plan.json':result['plan'],
             'fixed-cameras.json':plan['static_grid']['proof']['camera_manifest'],
             'garment-splits.json':plan['split_audit'],'training/curve.png':curve['image'],
             'training/curve.csv':curve['csv'],'training/loss.jsonl':curve['raw']}
    files.update({f'audit/{name}.json':entry for name,entry in audits.items()})
    for name,entry in result['panels'].items():
        directory = 'native-torso' if name.startswith('native_torso_') else 'fixed-views'
        files[f'{directory}/{name}.png'] = entry
    for frame,entry in native.items():
        files[f'native-full/{frame}.png'] = entry
    for name,entry in result.get('graph_replay',{}).items():
        files[f'graph-replay/{name}.json'] = entry
    target = out/'paired-evaluation.zip'
    if not target.exists():
        with TemporaryDirectory(prefix='.r24-evaluation-',dir=out) as temp:
            stage = Path(temp)/target.name
            with ZipFile(stage,'x') as package:
                for name,item in files.items():
                    package.write(checked(item),name)
            require(not target.exists(),'An R24 evaluation archive already exists.')
            stage.rename(target)
    with ZipFile(target) as package:
        require(len(package.namelist())==len(files) and set(package.namelist())==set(files)
                and package.testzip() is None,'The R24 evaluation archive has invalid members.')
        for name,item in files.items():
            require(sha256(package.read(name)).hexdigest()==item['sha256'],f'Changed R24 evidence: {name}')
    return dict(descriptor(target),count=len(files))


def quantitative_audits(subject,plan):
    directory = review.DELIVERY/'input-audit'
    surface = directory/f'{subject}-surface-paired.json'
    motion = directory/f'{subject}-motion-pairs-18k.json'
    opaque = directory/f'{subject}-motion-pairs-opaque-18k.json'
    mesh,moves = read(surface),read(motion)
    require(mesh['schema']=='shuolee-r24-surface-audit/1' and moves['schema']=='shuolee-r24-fixed-near-pair-transport/1'
            and mesh['subject']==moves['subject']==subject and mesh['formal_commit']==plan['commit']
            and moves['frames']==[row['frame'] for row in plan['frozen']['rows']]
            and moves['graph_sha256']==plan['graphs']['r24']['parameters']['sha256']
            and moves['r24_state_sha256']==plan['checkpoints']['r24']['checkpoint']['sha256'],
            'The independent surface/transport audit uses another graph or original pose roster.')
    for name,record in zip(review.NEW,mesh['formal'],strict=True):
        require(record['step']==review.NEW[name] and record['exact_frozen_pose_cameras']==28
                and record['checkpoint_sha256']==plan['checkpoints'][name]['checkpoint']['sha256'],
                'The numerical audit did not read both exact R24 duration checkpoints.')
    filtered = read(opaque)
    require(all(filtered[key]==moves[key] for key in ('schema','subject','frames','graph_sha256',
                'r23_state_sha256','r24_state_sha256')) and filtered['opaque_proxy'],
            'The separate opacity-proxy near-pair audit uses another run.')
    return dict(surface=descriptor(surface),transport=descriptor(motion),transport_opacity_proxy=descriptor(opaque))


def archive(subject):
    view = review.root_for(subject)/'view-comparison'
    result_path = view/'result.json'
    result = read(result_path)
    plan = read(checked(result['plan']))
    commit = plan['commit']
    require(result['state']=='paired_driven_and_static_views_rendered' and result['static_camera_count']==50
            and result['native_poses']==28 and result['commit']==commit
            and result['subject']==plan['subject']==subject and result['quality_accepted'] is False,
            'Archive requires the independent review of both exact R24 durations.')
    out = review.DELIVERY/subject
    out.mkdir(parents=True,exist_ok=True)
    endpoints = {name:endpoint(out,plan,name) for name in review.NEW}
    native = native_records(out/'native-full',plan,result)
    curve = prior.training_curve(out,plan['training_log'])
    audits = quantitative_audits(subject,plan)
    review_package = evaluation(out,plan,result_path,result,native,curve,audits)
    record_dir = prior.common.RECORDS/subject
    require(record_dir.is_dir() and record_dir.resolve(strict=True)==record_dir,'Use the existing subject record folder.')
    saved = {}
    tag = f'交付24_{commit[:7]}'
    with TemporaryDirectory(prefix='shuolee-r24-record-') as scratch:
        todo,pins = [],{}

        def add(key,item,filename):
            source = checked(item)
            pins[source] = item['sha256']
            stage,target = Path(scratch)/filename,record_dir/filename
            with Image.open(source) as image:
                image.convert('RGB').save(stage,quality=95,subsampling=0)
            digest = sha(stage)
            if target.exists():
                pin(target,digest)
            else:
                todo.append((stage,target))
            saved[key] = dict(file=str(target),sha256=digest)

        for frame,item in native.items():
            if item['role']=='selected_fit':
                add(frame,item,f'3DGS_{subject}_{tag}_原图-高斯对照-Mesh-特征点_{frame}.jpg')
            else:
                add(frame,item,f'分析_{subject}_{tag}_留出原图-高斯对照-Mesh-特征点_{frame}.jpg')
        for key in prior.prior.PITCH_KEYS:
            add(key,result['panels'][key],f'分析_{subject}_{tag}_{key}.jpg')
        ref = 'native_torso_'+plan['frozen']['ref']
        if ref in result['panels']:
            add(ref,result['panels'][ref],f'分析_{subject}_{tag}_{ref}.jpg')
        add('training_curve',curve['image'],f'分析_{subject}_{tag}_10000-18000_主训练曲线.jpg')
        prior.common.publish_images(todo,pins)
    receipt = dict(schema='shuolee-r24-paired-delivery/1',subject=subject,commit=commit,quality_accepted=False,
                   review=descriptor(result_path),plan=result['plan'],endpoints=endpoints,curve=curve,records=saved,
                   native_full=native,evaluation=review_package,split_audit=plan['split_audit'],
                   split_summary=plan['split_summary'],graph_replay=result['graph_replay'],
                   quantitative_audits=audits,
                   selected_count=24,heldout_count=4)
    item = prior.prior.write_receipt(out/'receipt.json',receipt)
    print(json.dumps(dict(receipt=item,record_images=len(saved),selected_images=24,heldouts=4)),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject',choices=qa.prior.SUBJECTS)
    archive(parser.parse_args().subject)
