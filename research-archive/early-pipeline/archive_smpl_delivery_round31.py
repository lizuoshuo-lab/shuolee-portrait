"""Archive R31 duration packages, comparison masks and image-only records."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from archive_smpl_delivery_round26 import endpoint as archive_endpoint
import archive_smpl_delivery_round30 as prior
import review_smpl_delivery_round31 as review


old=prior.old
pin,checked,read,require,descriptor=review.pin,review.checked,review.read,review.require,review.descriptor


def save_records(subject,commit,rows,result):
    folder=old.common.RECORDS/subject
    require(folder.is_dir() and folder.resolve(strict=True)==folder,'Use the existing image-only record folder.')
    tag=f'交付31_{commit[:7]}'
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
    with TemporaryDirectory(prefix='shuolee-r31-record-') as temp:
        pending=[]; pins={}
        for key,(item,name) in sources.items():
            source=checked(item); pins[source]=item['sha256']
            staged,target=Path(temp)/name,folder/name
            with Image.open(source) as image:
                image.convert('RGB').save(staged,quality=95,subsampling=0)
            digest=review.r29.sha(staged)
            if target.exists():
                pin(target,digest)
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
            and result['native_face_scored']==result['native_full_roi_scored']
            and set(selected)=={row['frame'] for row in rows if row['role']=='selected_fit'},
            'R31 archival requires both native metrics and complete fixed cameras.')
    out=review.DELIVERY/subject; out.mkdir(parents=True,exist_ok=True)
    endpoints={name:archive_endpoint(out,plan,name) for name in review.NEW}
    comparison=prior.evaluation_zip(out,path,result,plan)
    context={}
    if subject=='008':
        frame='frame_000047'
        for arm in review.ARMS:
            regions=result['native_scores'][arm][frame]['regions']
            require(all(regions[key]['score_state']=='unscored' and regions[key]['reason']=='rear_head'
                        for key in ('face','full_foreground')),
                    '008/047 must remain an unscored rear context image.')
        context[frame]=result['panels'][f'native_{frame}_face']
    records=save_records(subject,plan['commit'],rows,result)
    receipt=old.write_receipt(out/'receipt.json',dict(schema='shuolee-r31-face-delivery/1',
        subject=subject,commit=plan['commit'],review=descriptor(path),plan=result['plan'],
        controls={'r29':plan['baseline_review'],'r30':plan['baseline_r30_review']},
        endpoints=endpoints,selected=selected,evaluation=comparison,face_visibility=result['face_visibility'],
        unscored_context=context,
        records=records,selected_count=counts['selected_fit'],heldout_count=counts['holdout'],
        auxiliary_count=counts['auxiliary_face'],native_face_comparisons=faces,
        native_face_scored=result['native_face_scored'],native_full_roi_scored=result['native_full_roi_scored'],
        unavailable_holdout_frames=result['unavailable_holdout_frames'],
        unavailable_holdout_reasons=result['unavailable_holdout_reasons'],quality_accepted=False))
    print(json.dumps(dict(receipt=receipt,selected=len(selected),faces=faces,images=len(records))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject',choices=review.SUBJECTS)
    archive(parser.parse_args().subject)
