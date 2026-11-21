"""Archive the two R30 durations and their same-camera comparisons."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from PIL import Image

import archive_smpl_delivery_round29 as prior
import review_smpl_delivery_round30 as review


old=prior.prior.prior.prior.old
checked,read,descriptor,require,rebound=review.checked,review.read,review.descriptor,review.require,review.rebound


def evaluation_zip(out,path,result,plan):
    visibility=checked(result['face_visibility'])
    files={
        'review.json':descriptor(path),'plan.json':result['plan'],
        'fixed-cameras.json':plan['static_grid']['proof']['camera_manifest'],
        'face-visibility.json':result['face_visibility'],'face-visibility.npz':read(visibility)['arrays'],
    }
    for key,item in result['panels'].items():
        folder='native-face' if key.startswith('native_') else 'fixed-views'
        files[f'{folder}/{key}.png']=item
    files.update({f'selected-views/{frame}.png':item for frame,item in result['selected'].items()})
    target=out/'paired-face-evaluation.zip'
    if not target.exists():
        with TemporaryDirectory(prefix='.r30-eval-',dir=out) as temporary:
            staged=Path(temporary)/target.name
            with ZipFile(staged,'x') as package:
                for name,item in files.items():
                    package.write(checked(item),name)
            require(not target.exists(),'Preserve existing paired R30 evidence.')
            staged.rename(target)
    with ZipFile(target) as package:
        require(len(package.namelist())==len(files) and set(package.namelist())==set(files)
                and package.testzip() is None,'The R30 comparison ZIP is incomplete.')
        for name,item in files.items():
            require(sha256(package.read(name)).hexdigest()==item['sha256'],f'Changed R30 evidence: {name}')
    return dict(descriptor(target),files=len(files))


def save_records(subject,commit,rows,result):
    folder=old.common.RECORDS/subject
    require(folder.is_dir() and folder.resolve(strict=True)==folder,'Use the existing image-only record folder.')
    tag=f'交付30_{commit[:7]}'
    sources={frame:(item,f'3DGS_{subject}_{tag}_原图-高斯对照-Mesh-特征点_{frame}.jpg')
             for frame,item in result['selected'].items()}
    for row in rows:
        frame=row['frame']; key=f'native_{frame}_face'
        if row['role']!='selected_fit' and key in result['panels']:
            scope='面部' if frame in result['native_face_scored'] else '头部未评分'
            sources[key]=(result['panels'][key],f'分析_{subject}_{tag}_{row["role"]}_原图-{scope}高斯_{frame}.jpg')
    for key in old.PITCH_KEYS:
        sources[key]=(result['panels'][key],f'分析_{subject}_{tag}_{key}.jpg')
    saved={}
    with TemporaryDirectory(prefix='shuolee-r30-record-') as temp:
        pending=[]; pins={}
        for key,(item,name) in sources.items():
            source=checked(item); pins[source]=item['sha256']
            staged,target=Path(temp)/name,folder/name
            with Image.open(source) as image:
                image.convert('RGB').save(staged,quality=95,subsampling=0)
            digest=review.prior.sha(staged)
            if target.exists():
                review.pin(target,digest)
            else:
                pending.append((staged,target))
            saved[key]=dict(file=str(target),sha256=digest)
        old.common.publish_images(pending,pins)
    return saved


def archive(subject):
    path=review.root_for(subject)/'view-comparison/result.json'
    result=read(path); plan=read(checked(result['plan']))
    rows=plan['frozen']['rows']; selected=result['selected']
    counts={role:sum(row['role']==role for row in rows) for role in ('selected_fit','holdout','auxiliary_face')}
    faces=sum(key.startswith('native_') for key in result['panels'])
    require(result['state']=='paired_native_face_and_static_views_rendered' and result['subject']==plan['subject']==subject
            and result['commit']==plan['commit'] and result['static_camera_count']==50
            and faces==result['native_face_comparisons'] and len(rows)==result['native_poses']
            and set(selected)=={row['frame'] for row in rows if row['role']=='selected_fit'},
            'R30 archival requires completed fixed and native paired comparisons.')
    out=review.DELIVERY/subject; out.mkdir(parents=True,exist_ok=True)
    endpoints={name:prior.prior.prior.prior.endpoint(out,plan,name) for name in review.NEW}
    comparison=evaluation_zip(out,path,result,plan)
    # Keep the established image-only naming and atomic D: publication.
    records=save_records(subject,plan['commit'],rows,result)
    receipt=old.write_receipt(out/'receipt.json',dict(schema='shuolee-r30-face-delivery/1',
        subject=subject,commit=plan['commit'],review=descriptor(path),plan=result['plan'],
        baseline=plan['baseline_archive'],endpoints=endpoints,selected=selected,evaluation=comparison,
        face_visibility=result['face_visibility'],records=records,selected_count=counts['selected_fit'],
        heldout_count=counts['holdout'],auxiliary_count=counts['auxiliary_face'],
        native_face_comparisons=faces,native_face_scored=result['native_face_scored'],
        unavailable_holdout_frames=result['unavailable_holdout_frames'],
        unavailable_holdout_reasons=result['unavailable_holdout_reasons'],quality_accepted=False))
    print(json.dumps(dict(receipt=receipt,selected=len(selected),faces=faces,images=len(records))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject',choices=review.SUBJECTS)
    archive(parser.parse_args().subject)
