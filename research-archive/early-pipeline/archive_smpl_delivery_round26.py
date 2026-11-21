"""Archive paired R26 face reconstructions and the exact named motion packages."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parent))
import archive_smpl_delivery_round22 as old
import archive_smpl_delivery_round23 as common_archive
import review_smpl_delivery_round26 as review


pin,checked,read,sha,require,descriptor = review.pin,review.checked,review.read,review.sha,review.require,review.descriptor
TITLE = ('Original native RGB','R22 18,000','R26 10,000','R26 18,000','Frozen SMPL-X','Native 2D features')
COLS = ('原图','R22 18k','R26 10k','R26 18k','冻结 Mesh 投影','2D 特征与投影')


def endpoint(out,plan,name):
    current = plan['snapshots'][name]
    arm = plan['static_grid']['arms'][name]
    portable = checked(current['portable'])
    mesh = old.verify_scene(portable.parent,dict(read(portable),reference_frame_id=arm['ref']))
    ply = pin(arm['ply'],arm['ply_sha256'])
    camera = pin(arm['camera_path'],arm['camera_sha256'])
    files = dict(portable_ply=ply,camera=camera,camera_provenance=camera.with_name('camera_provenance.json'),
                 scene=checked(current['scene']),drive=checked(current['package']))
    target = out/f'duration-{current["step"]}'
    target.mkdir(exist_ok=True)
    saved = old.copy_proof(target,{key:(path,sha(path)) for key,path in files.items()})
    return dict(saved,step=current['step'],points=current['points'],reference_frame=arm['ref'],
                reference_mesh=descriptor(mesh),native_replay=current['replay_receipts'])


def selected_panels(out,plan,result):
    previous = read(checked(plan['baseline_archive']))['selected_panels']
    paste = review.rebound(common_archive.paste_panel,TITLE=TITLE)
    crop = common_archive.crop_file
    out.mkdir(exist_ok=True)
    panels = {}
    for row in plan['frozen']['rows'][:24]:
        frame = row['frame']
        prior = previous[frame]
        original = checked(prior)
        box = prior['crop_xyxy']
        source,mesh,features = (crop(original,box,box,col=col,header=26) for col in (0,3,4))
        require(prior['source_sha256']==row['image']['sha256']
                and np.array_equal(np.asarray(source),np.asarray(crop(checked(row['image']),box))),
                'The archived R22 overlay did not use the exact same source crop.')
        renderings = []
        for name in review.ARMS:
            body = result['native'][name][frame]['body']
            require(body['crop_xyxy']==box,'The independent selected render uses a different native crop.')
            renderings.append(crop(checked(body['files']['black']),box,box))
        panel = paste(out/f'{frame}.png',(source,*renderings,mesh,features))
        panels[frame] = dict(panel,frame=frame,crop_xyxy=box,columns=COLS,source_sha256=row['image']['sha256'])
    return panels


def review_zip(out,plan,result_path,result,selected,material):
    files = {'review.json':descriptor(result_path),'plan.json':result['plan'],
             'fixed-cameras.json':plan['static_grid']['proof']['camera_manifest'],
             'training-config.json':plan['training_config'],'training-progress.json':plan['training_progress'],
             'split-events.json':plan['split_events'],'source-material-evidence.json':material}
    for key,item in result['panels'].items():
        folder = 'native-face' if key.startswith('native_') else 'fixed-views'
        files[f'{folder}/{key}.png'] = item
    files.update({f'selected-views/{frame}.png':item for frame,item in selected.items()})
    target = out/'paired-face-evaluation.zip'
    if not target.exists():
        with TemporaryDirectory(prefix='.r26-eval-',dir=out) as temporary:
            stage = Path(temporary)/target.name
            with ZipFile(stage,'x') as archive:
                for name,item in files.items():
                    archive.write(checked(item),name)
            require(not target.exists(),'An R26 paired archive already exists.')
            stage.rename(target)
    with ZipFile(target) as archive:
        require(len(archive.namelist())==len(files) and set(archive.namelist())==set(files)
                and archive.testzip() is None,'The R26 face evaluation archive is incomplete.')
        for name,item in files.items():
            require(sha256(archive.read(name)).hexdigest()==item['sha256'],f'Changed R26 review member {name}.')
    return dict(descriptor(target),files=len(files))


def save_records(subject,commit,result,selected,rows):
    folder = old.common.RECORDS/subject
    require(folder.is_dir() and folder.resolve(strict=True)==folder,'Use the existing subject record folder.')
    saved = {}
    tag = f'交付26_{commit[:7]}'
    with TemporaryDirectory(prefix='shuolee-r26-record-') as temp:
        pending,pins = [],{}

        def image_only(key,item,filename):
            source = checked(item)
            pins[source] = item['sha256']
            staged,target = Path(temp)/filename,folder/filename
            with Image.open(source) as image:
                image.convert('RGB').save(staged,quality=95,subsampling=0)
            digest = sha(staged)
            if target.exists():
                pin(target,digest)
            else:
                pending.append((staged,target))
            saved[key] = dict(file=str(target),sha256=digest)

        for frame,item in selected.items():
            image_only(frame,item,f'3DGS_{subject}_{tag}_原图-高斯对照-Mesh-特征点_{frame}.jpg')
        for row in rows[24:]:
            frame = row['frame']
            key = f'native_{frame}_face'
            image_only(key,result['panels'][key],f'分析_{subject}_{tag}_{row["role"]}_原图-面部高斯_{frame}.jpg')
        for key in old.PITCH_KEYS:
            image_only(key,result['panels'][key],f'分析_{subject}_{tag}_{key}.jpg')
        old.common.publish_images(pending,pins)
    return saved


def archive(subject):
    result_path = review.root_for(subject)/'view-comparison/result.json'
    result = read(result_path)
    plan = read(checked(result['plan']))
    commit = plan['commit']
    require(result['state']=='paired_native_face_and_static_views_rendered' and result['commit']==commit
            and result['subject']==plan['subject']==subject and not result['quality_accepted']
            and (result['static_camera_count'],result['native_poses'],result['auxiliary_face_poses'])==(50,28,4),
            'Archive requires both independently reviewed R26 durations.')
    config = read(checked(plan['training_config']))
    material = descriptor(pin(review.DELIVERY/'input-audit'/f'{subject}-material-evidence.json',
                              config['pinned_inputs']['material_receipt']))
    out = review.DELIVERY/subject
    out.mkdir(parents=True,exist_ok=True)
    endpoints = {name:endpoint(out,plan,name) for name in review.NEW}
    selected = selected_panels(out/'selected-views',plan,result)
    package = review_zip(out,plan,result_path,result,selected,material)
    records = save_records(subject,commit,result,selected,plan['frozen']['rows'])
    receipt = old.write_receipt(out/'receipt.json',dict(schema='shuolee-r26-face-delivery/1',subject=subject,
        commit=commit,review=descriptor(result_path),plan=result['plan'],endpoints=endpoints,material=material,
        selected=selected,evaluation=package,records=records,selected_count=24,heldout_count=4,
        auxiliary_count=4,quality_accepted=False))
    print(json.dumps(dict(receipt=receipt,selected_images=len(selected),all_record_images=len(records))),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject',choices=review.qa.prior.SUBJECTS)
    archive(parser.parse_args().subject)
