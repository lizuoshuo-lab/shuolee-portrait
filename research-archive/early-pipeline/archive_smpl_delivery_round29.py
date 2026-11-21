"""Archive the two R29 model durations and their same-camera image records."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

import archive_smpl_delivery_round28 as prior
import review_smpl_delivery_round29 as review


pin,checked,read,sha,require,descriptor=review.pin,review.checked,review.read,review.sha,review.require,review.descriptor


def save_records(subject,commit,rows,result):
    common=prior.prior.prior.old.common
    folder=common.RECORDS/subject
    require(folder.is_dir() and folder.resolve(strict=True)==folder,'Use the original image record folder.')
    tag=f'交付29_{commit[:7]}'
    sources={frame:(item,f'3DGS_{subject}_{tag}_原图-高斯对照-Mesh-特征点_{frame}.jpg')
             for frame,item in result['selected'].items()}
    for row in rows:
        frame=row['frame']; key=f'native_{frame}_face'
        if row['role']!='selected_fit' and key in result['panels']:
            sources[key]=(result['panels'][key],f'分析_{subject}_{tag}_{row["role"]}_原图-面部高斯_{frame}.jpg')
    for key in prior.prior.prior.old.PITCH_KEYS:
        sources[key]=(result['panels'][key],f'分析_{subject}_{tag}_{key}.jpg')
    saved={}
    with TemporaryDirectory(prefix='shuolee-r29-record-') as temp:
        pending=[]; pins={}
        for key,(item,name) in sources.items():
            source=checked(item); pins[source]=item['sha256']
            staged,target=Path(temp)/name,folder/name
            with Image.open(source) as image:
                image.convert('RGB').save(staged,quality=95,subsampling=0)
            digest=sha(staged)
            if target.exists():
                pin(target,digest)
            else:
                pending.append((staged,target))
            saved[key]=dict(file=str(target),sha256=digest)
        common.publish_images(pending,pins)
    return saved


def archive(subject):
    path=review.root_for(subject)/'view-comparison/result.json'
    result=read(path); plan=read(checked(result['plan']))
    rows=plan['frozen']['rows']; selected=result['selected']
    native_count=sum(key.startswith('native_') for key in result['panels'])
    counts={role:sum(row['role']==role for row in rows) for role in ('selected_fit','holdout','auxiliary_face')}
    require(result['state']=='paired_native_face_and_static_views_rendered' and
            result['subject']==plan['subject']==subject and result['commit']==plan['commit'] and
            len(rows)==sum(counts.values())==result['native_poses'] and
            native_count==result['native_face_comparisons'] and result['static_camera_count']==50 and
            set(selected)=={row['frame'] for row in rows if row['role']=='selected_fit'},
            'R29 archival requires matched originals and complete fixed views.')
    out=review.DELIVERY/subject; out.mkdir(parents=True,exist_ok=True)
    endpoints={name:prior.prior.prior.endpoint(out,plan,name) for name in review.NEW}
    evaluation=prior.review_zip(out,plan,path,result,selected)
    records=save_records(subject,plan['commit'],rows,result)
    receipt=prior.prior.prior.old.write_receipt(out/'receipt.json',dict(schema='shuolee-r29-face-delivery/1',
        subject=subject,commit=plan['commit'],review=descriptor(path),plan=result['plan'],
        baseline=plan['baseline_archive'],endpoints=endpoints,selected=selected,evaluation=evaluation,records=records,
        selected_count=counts['selected_fit'],heldout_count=counts['holdout'],auxiliary_count=counts['auxiliary_face'],
        native_face_comparisons=native_count,unavailable_holdout_frames=result['unavailable_holdout_frames'],
        unavailable_holdout_reasons=result['unavailable_holdout_reasons'],quality_accepted=False))
    print(json.dumps(dict(receipt=receipt,selected=len(selected),faces=native_count,images=len(records))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject',choices=review.SUBJECTS)
    archive(parser.parse_args().subject)
