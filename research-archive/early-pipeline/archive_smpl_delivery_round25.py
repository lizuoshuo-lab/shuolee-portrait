"""Archive both R25 native drive and static models with original-pose review images."""

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parent))
import archive_smpl_delivery_round24 as prior
import review_smpl_delivery_round25 as review


qa = review.qa
pin,checked,read,sha,require,descriptor,rebound = review.pin,review.checked,review.read,review.sha,review.require,review.descriptor,review.rebound
TITLE = ('Original native RGB','R24 18,000','R25 10,000','R25 18,000','Frozen SMPL-X guide','2D input / fit')
COLS = ('原图','R24 18k','R25 10k','R25 18k','冻结 Mesh 投影','2D 特征与投影')


def endpoint(out,plan,name):
    item = prior.endpoint(out,plan,name)
    drive = plan['drives'][name]
    sidecars = plan['graphs'][name]
    anchor = checked(sidecars['anchors'])
    target = out/f"duration-{drive['step']}"
    item.update(prior.prior.copy_proof(target,dict(anchors=(anchor,sidecars['anchors']['sha256']))))
    package = checked(drive['package'])
    with ZipFile(package) as zip_file:
        require(zip_file.read('drive/'+review.ANCHOR_FILE)==anchor.read_bytes(),
                'The exported V3 archive has a different garment anchor sidecar.')
    snapshot = checked(drive['portable']).parent
    metrics = snapshot/'garment-anchors.json'
    numbers = read(metrics)
    require(numbers['total_rows']==sidecars['clothing_rows'] and numbers['trusted_rows']==sidecars['trusted_rows'],
            'The saved anchor statistics differ from the exported V3 rows.')
    return dict(item,anchor_metrics=descriptor(metrics),trusted_rows=sidecars['trusted_rows'])


def records(subject,commit,plan,result,native,curve):
    common = prior.prior.common
    record_dir = common.RECORDS/subject
    require(record_dir.is_dir() and record_dir.resolve(strict=True)==record_dir,'Use the existing subject record folder.')
    saved = {}
    tag = f'交付25_{commit[:7]}'
    with TemporaryDirectory(prefix='shuolee-r25-record-') as scratch:
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
        for key in prior.prior.prior.PITCH_KEYS:
            add(key,result['panels'][key],f'分析_{subject}_{tag}_{key}.jpg')
        ref = 'native_torso_'+plan['frozen']['ref']
        if ref in result['panels']:
            add(ref,result['panels'][ref],f'分析_{subject}_{tag}_{ref}.jpg')
        add('training_curve',curve['image'],f'分析_{subject}_{tag}_10000-18000_主训练曲线.jpg')
        common.publish_images(todo,pins)
    return saved


def numerical_audit(subject,plan):
    path = review.DELIVERY/'input-audit'/f'{subject}-anchors-paired.json'
    result = read(path)
    require(result['schema']=='shuolee-r25-independent-anchors/1' and result['subject']==subject
            and result['formal_commits']['25']==plan['commit']
            and result['same_run']['exact_draw_and_topology_prefix'],
            'The independent R25 anchor audit uses another run or invalid checkpoint history.')
    for (name,step),record in zip(review.NEW.items(),result['formal'],strict=True):
        require(record['step']==step and record['r25']['checkpoint_sha256']==plan['checkpoints'][name]['checkpoint']['sha256'],
                'The numerical audit did not read both exact R25 duration checkpoints.')
    return descriptor(path)


def archive(subject):
    result_path = review.root_for(subject)/'view-comparison'/'result.json'
    result = read(result_path)
    plan = read(checked(result['plan']))
    commit = plan['commit']
    require(result['state']=='paired_driven_and_static_views_rendered' and result['static_camera_count']==50
            and result['native_poses']==28 and result['commit']==commit
            and result['subject']==plan['subject']==subject and result['quality_accepted'] is False,
            'Archive requires the independent review of both exact R25 durations.')
    out = review.DELIVERY/subject
    out.mkdir(parents=True,exist_ok=True)
    endpoints = {name:endpoint(out,plan,name) for name in review.NEW}
    native = rebound(prior.native_records,review=review,TITLE=TITLE,COLS=COLS)(
        out/'native-full',dict(plan,r23_archive=plan['baseline_archive']),result)
    curve = prior.prior.training_curve(out,plan['training_log'])
    audits = {name+'_anchor_metrics':item['anchor_metrics'] for name,item in endpoints.items()}
    audits['numerical_anchors'] = numerical_audit(subject,plan)
    review_package = prior.evaluation(out,plan,result_path,result,native,curve,audits)
    saved = records(subject,commit,plan,result,native,curve)
    receipt = dict(schema='shuolee-r25-paired-delivery/1',subject=subject,commit=commit,quality_accepted=False,
                   review=descriptor(result_path),plan=result['plan'],endpoints=endpoints,curve=curve,records=saved,
                   native_full=native,evaluation=review_package,split_audit=plan['split_audit'],
                   split_summary=plan['split_summary'],graph_replay=result['graph_replay'],
                   quantitative_audits=audits,selected_count=24,heldout_count=4)
    item = prior.prior.prior.write_receipt(out/'receipt.json',receipt)
    print(json.dumps(dict(receipt=item,record_images=len(saved),selected_images=24,heldouts=4)),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject',choices=qa.prior.SUBJECTS)
    archive(parser.parse_args().subject)
