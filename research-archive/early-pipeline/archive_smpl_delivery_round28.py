"""Archive checked R28 static scenes, native motion, and original-pixel image records."""

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
import archive_smpl_delivery_round27 as prior
import review_smpl_delivery_round28 as review


pin,checked,read,sha,require,descriptor,rebound = review.pin,review.checked,review.read,review.sha,review.require,review.descriptor,review.rebound
TITLE = ('Original native RGB','R27 18,000','R28 10,000','R28 18,000','Frozen SMPL-X','Native 2D features')
COLS = ('原图','R27 18k','R28 10k','R28 18k','冻结 Mesh 投影','2D 特征与投影')
SUBJECTS = ('001','005','006','008','016')


def review_zip(out,plan,result_path,result,selected):
    files = {'review.json':descriptor(result_path),'plan.json':result['plan'],
             'fixed-cameras.json':plan['static_grid']['proof']['camera_manifest']}
    for key,item in result['panels'].items():
        folder = 'native-face' if key.startswith('native_') else 'fixed-views'
        files[f'{folder}/{key}.png'] = item
    files.update({f'selected-views/{frame}.png':item for frame,item in selected.items()})
    target = out/'paired-face-evaluation.zip'
    if not target.exists():
        with TemporaryDirectory(prefix='.r28-eval-',dir=out) as temporary:
            stage = Path(temporary)/target.name
            with ZipFile(stage,'x') as package:
                for name,item in files.items():
                    package.write(checked(item),name)
            require(not target.exists(),'Preserve the existing R28 paired archive.')
            stage.rename(target)
    with ZipFile(target) as package:
        require(len(package.namelist())==len(files) and set(package.namelist())==set(files)
                and package.testzip() is None,'R28 paired evaluation archive is incomplete.')
        for name,item in files.items():
            require(sha256(package.read(name)).hexdigest()==item['sha256'],f'Changed R28 evidence {name}.')
    return dict(descriptor(target),files=len(files))


def save_records(subject,commit,result,selected,rows):
    common = prior.prior.old.common
    folder = common.RECORDS/subject
    require(folder.is_dir() and folder.resolve(strict=True)==folder,'Use the existing subject record folder.')
    saved = {}
    tag = f'交付28_{commit[:7]}'
    with TemporaryDirectory(prefix='shuolee-r28-record-') as temp:
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
        for row in rows:
            if row['role']=='selected_fit':
                continue
            frame = row['frame']
            key = f'native_{frame}_face'
            image_only(key,result['panels'][key],f'分析_{subject}_{tag}_{row["role"]}_原图-面部高斯_{frame}.jpg')
        for key in prior.prior.old.PITCH_KEYS:
            image_only(key,result['panels'][key],f'分析_{subject}_{tag}_{key}.jpg')
        common.publish_images(pending,pins)
    return saved


def archive(subject):
    result_path = review.root_for(subject)/'view-comparison/result.json'
    result = read(result_path)
    plan = read(checked(result['plan']))
    commit = plan['commit']
    rows = plan['frozen']['rows']
    selected_count = sum(row['role']=='selected_fit' for row in rows)
    heldout_count = sum(row['role']=='holdout' for row in rows)
    auxiliary_count = sum(row['role']=='auxiliary_face' for row in rows)
    require(result['state']=='paired_native_face_and_static_views_rendered' and result['commit']==commit
            and result['subject']==plan['subject']==subject and not result['quality_accepted']
            and len(rows)==selected_count+heldout_count+auxiliary_count
            and (result['static_camera_count'],result['native_poses'],result['auxiliary_face_poses'])
            ==(50,selected_count+heldout_count,auxiliary_count),
            'Archive requires both independently reviewed R28 durations.')
    out = review.DELIVERY/subject
    out.mkdir(parents=True,exist_ok=True)
    endpoints = {name:prior.prior.endpoint(out,plan,name) for name in review.NEW}
    if 'legacy_source' in plan:
        selected = result['selected']
    else:
        record_view = SimpleNamespace(ARMS=('r27',*review.NEW),rebound=rebound)
        selected = rebound(prior.prior.selected_panels,review=record_view,TITLE=TITLE,COLS=COLS)(
            out/'selected-views',plan,result)
    require(set(selected)=={row['frame'] for row in rows if row['role']=='selected_fit'},
            'R28 selected panel names differ from the actual training roster.')
    package = review_zip(out,plan,result_path,result,selected)
    records = save_records(subject,commit,result,selected,rows)
    details = dict(schema='shuolee-r28-face-delivery/1',subject=subject,
        commit=commit,review=descriptor(result_path),plan=result['plan'],endpoints=endpoints,
        selected=selected,evaluation=package,records=records,
        selected_count=selected_count,heldout_count=heldout_count,auxiliary_count=auxiliary_count,
        unavailable_holdout_frames=plan.get('unavailable_holdout_frames',[]),quality_accepted=False)
    if 'legacy_source' in plan:
        details.update(legacy_source=plan['legacy_source'],
                       unavailable_holdout_reasons=plan['unavailable_holdout_reasons'])
    receipt = prior.prior.old.write_receipt(out/'receipt.json',details)
    print(json.dumps(dict(receipt=receipt,selected_images=len(selected),all_record_images=len(records))),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject',choices=SUBJECTS)
    archive(parser.parse_args().subject)
