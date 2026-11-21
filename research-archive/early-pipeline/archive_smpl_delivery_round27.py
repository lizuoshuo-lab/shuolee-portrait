"""Archive verified R27 static scenes, native motion and original-pixel records."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from zipfile import ZipFile

from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parent))
import archive_smpl_delivery_round26 as prior
import review_smpl_delivery_round27 as review


pin,checked,read,sha,require,descriptor,rebound = review.pin,review.checked,review.read,review.sha,review.require,review.descriptor,review.rebound
TITLE = ('Original native RGB','R26 18,000','R27 10,000','R27 18,000','Frozen SMPL-X','Native 2D features')
COLS = ('原图','R26 18k','R27 10k','R27 18k','冻结 Mesh 投影','2D 特征与投影')


def numerical_evidence(subject,plan):
    root = review.DELIVERY/'input-audit'
    path = root/f'{subject}-face-invariants-paired.json'
    paired = read(path)
    require(paired['schema']=='shuolee-r27-independent-face-invariants-paired/1'
            and paired['subject']==subject and paired['formal_commit']==plan['commit'],
            'The independent R27 physical audit uses a different subject or production commit.')
    endpoints = {}
    for (name,step),row in zip(review.NEW.items(),paired['formal'],strict=True):
        state = plan['checkpoints'][name]['checkpoint']['sha256']
        item = pin(root/f'{subject}-face-invariants-{step:06d}.json',row['independent_receipt_sha256'])
        raw = read(item)
        require(row['step']==step==raw['step'] and row['checkpoint_sha256']==state==raw['checkpoint_sha256']
                and raw['pass_'] and raw['material_npz_sha256']==plan['material']['arrays']['sha256']
                and raw['native_gradient_audit_sha256']==plan['gradients']['sha256']
                and raw['frozen_pose_and_camera']['camera_count']==32,
                'A physical R27 audit does not reference the exact reviewed endpoint or native pose roster.')
        endpoints[name] = descriptor(item)
    return dict(paired=descriptor(path),**endpoints)


def review_zip(out,plan,result_path,result,selected,numerical):
    files = {'review.json':descriptor(result_path),'plan.json':result['plan'],
             'fixed-cameras.json':plan['static_grid']['proof']['camera_manifest'],
             'training-config.json':plan['training_config'],'training-progress.json':plan['training_progress'],
             'split-events.json':plan['split_events'],'source-material-visibility.json':plan['material']['receipt'],
             'source-material-visibility.npz':plan['material']['arrays'],'native-gradient-audit.json':plan['gradients']}
    files.update({f'physical-audit/{key}.json':item for key,item in numerical.items()})
    for key,item in result['panels'].items():
        folder = 'native-face' if key.startswith('native_') else 'fixed-views'
        files[f'{folder}/{key}.png'] = item
    files.update({f'selected-views/{frame}.png':item for frame,item in selected.items()})
    target = out/'paired-face-evaluation.zip'
    if not target.exists():
        with TemporaryDirectory(prefix='.r27-eval-',dir=out) as temporary:
            stage = Path(temporary)/target.name
            with ZipFile(stage,'x') as package:
                for name,item in files.items():
                    package.write(checked(item),name)
            require(not target.exists(),'Preserve the existing R27 paired archive.')
            stage.rename(target)
    with ZipFile(target) as package:
        require(len(package.namelist())==len(files) and set(package.namelist())==set(files)
                and package.testzip() is None,'R27 paired evaluation archive is incomplete.')
        for name,item in files.items():
            require(sha256(package.read(name)).hexdigest()==item['sha256'],f'Changed R27 evidence {name}.')
    return dict(descriptor(target),files=len(files))


def save_records(subject,commit,result,selected,rows):
    common = prior.old.common
    folder = common.RECORDS/subject
    require(folder.is_dir() and folder.resolve(strict=True)==folder,'Use the existing subject record folder.')
    saved = {}
    tag = f'交付27_{commit[:7]}'
    with TemporaryDirectory(prefix='shuolee-r27-record-') as temp:
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
        for key in prior.old.PITCH_KEYS:
            image_only(key,result['panels'][key],f'分析_{subject}_{tag}_{key}.jpg')
        common.publish_images(pending,pins)
    return saved


def archive(subject):
    result_path = review.root_for(subject)/'view-comparison/result.json'
    result = read(result_path)
    plan = read(checked(result['plan']))
    commit = plan['commit']
    require(result['state']=='paired_native_face_and_static_views_rendered' and result['commit']==commit
            and result['subject']==plan['subject']==subject and not result['quality_accepted']
            and (result['static_camera_count'],result['native_poses'],result['auxiliary_face_poses'])==(50,28,4),
            'Archive requires both independently reviewed R27 durations.')
    config = read(checked(plan['training_config']))
    material = plan['material']['receipt']
    pin(checked(material),config['pinned_inputs']['material_receipt'])
    numerical = numerical_evidence(subject,plan)
    out = review.DELIVERY/subject
    out.mkdir(parents=True,exist_ok=True)
    endpoints = {name:prior.endpoint(out,plan,name) for name in review.NEW}
    record_view = SimpleNamespace(ARMS=('r26',*review.NEW),rebound=rebound)
    selected = rebound(prior.selected_panels,review=record_view,TITLE=TITLE,COLS=COLS)(
        out/'selected-views',plan,result)
    package = review_zip(out,plan,result_path,result,selected,numerical)
    records = save_records(subject,commit,result,selected,plan['frozen']['rows'])
    receipt = prior.old.write_receipt(out/'receipt.json',dict(schema='shuolee-r27-face-delivery/1',subject=subject,
        commit=commit,review=descriptor(result_path),plan=result['plan'],endpoints=endpoints,material=material,
        material_arrays=plan['material']['arrays'],physical_audit=numerical,
        selected=selected,evaluation=package,records=records,selected_count=24,heldout_count=4,
        auxiliary_count=4,quality_accepted=False))
    print(json.dumps(dict(receipt=receipt,selected_images=len(selected),all_record_images=len(records))),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject',choices=review.qa.prior.SUBJECTS)
    archive(parser.parse_args().subject)
