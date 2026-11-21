"""Bind the two verified R21 subjects, paired duration exports and diagnostics."""

import argparse
import json
from pathlib import Path
import sys
from urllib.parse import quote
from zipfile import ZipFile

sys.path.insert(0,str(Path(__file__).resolve().parent))
import publish_smpl_delivery_round21_report as publisher


reviewer = publisher.reviewer
pin,read,require,descriptor = reviewer.pin,reviewer.read,reviewer.require,reviewer.descriptor


def verify_branch(branch,face_views,steps):
    packages = branch['packages']
    for key in ('scene','bound_replay'):
        pin(packages[key],packages[key+'_sha256'])
    for key in ('portable_ply','portable_camera','camera_provenance'):
        item = branch[key]
        pin(item['path'],item['sha256'])
    with ZipFile(packages['scene']) as bundle:
        names = bundle.namelist()
        require(bundle.testzip() is None and 'cameras.json' in names
                and f'point_cloud/iteration_{steps}/point_cloud.ply' in names,
                'A duration scene is missing its expected PLY iteration or paired camera.')
    with ZipFile(packages['bound_replay']) as bundle:
        names = bundle.namelist()
        required = {f'gaussian-run/step-{steps:06d}/'+name for name in ('bound-local.ply','bound-replay.json','world.ply')}
        bound_count = 14+4*face_views if face_views else 10
        require(len(names)==bound_count and bundle.testzip() is None and required.issubset(names),
                'A duration bound replay package differs from the standard pose schema.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit',required=True)
    parser.add_argument('--findings',type=Path,required=True)
    parser.add_argument('--delivery',type=Path,default=reviewer.DELIVERY)
    parser.add_argument('--page',type=Path,default=publisher.shared.PAGE)
    args = parser.parse_args()
    anchor = f'smpl-delivery-round21-{args.commit[:7]}'
    require(args.page.read_text(encoding='utf-8').count(f'id="{anchor}"')==1,'The completed report anchor is missing or duplicate.')
    findings = read(args.findings)
    require(findings['commit']==args.commit and findings['quality_accepted'] is False,
            'The findings are not the matching visual review.')
    report = dict(descriptor(args.page),anchor=anchor,url='http://127.0.0.1:8870/diagnostics/video10-global-initializer/index.html#'+quote(anchor))
    receipt = dict(schema='shuolee-round21-independent-review-v1',state='native_static_and_paired_archives_verified',
        commit=args.commit,baseline_round15=reviewer.R15_COMMIT,baseline_round20=reviewer.R20_COMMIT,
        quality_accepted=False,findings=descriptor(args.findings),report=report,subjects={})
    for subject in reviewer.SUBJECTS:
        archive = args.delivery/subject/'receipt.json'
        loaded = publisher.load(subject,args.commit,args.delivery)
        data,review = loaded['receipt'],loaded['review']
        packages = data['packages']
        require(findings['comparison_receipt_sha256'][subject]==data['review_sha256']
                and packages['static_camera_count']==50 and packages['static_portable_arms']==4
                and packages['same_run_static_control_arms']==2 and len(review['same_run_10k_pose_readback'])==8,
                'The exact two-duration visual and native pose readbacks are incomplete.')
        verify_branch(data,packages['face_only'],18000)
        ten = data['secondary_endpoint']
        verify_branch(ten,packages['face_only'],10000)
        for key in ('pose_conditioned_evaluation','named_pose_roster'):
            pin(packages[key],packages[key+'_sha256'])
        with ZipFile(packages['pose_conditioned_evaluation']) as bundle:
            require(len(bundle.namelist())==packages['evaluation_member_count'] and bundle.testzip() is None,
                    'The pose-conditioned evaluation ZIP inventory is incomplete.')
        publisher.statuses(findings['subjects'][subject])
        receipt['subjects'][subject] = dict(archive=descriptor(archive),points=data['points'],reference=data['reference'],
            review=dict(file=data['review'],sha256=data['review_sha256']),packages=packages,
            portable_ply=data['portable_ply'],portable_camera=data['portable_camera'],camera_provenance=data['camera_provenance'],
            secondary_endpoint=ten,duration=data['duration'],training_curve=data['training_curve'],
            surface_attachment=data['surface_attachment'],matched_geometry=data['matched_geometry'],
            delivered_comparison=findings['delivered_comparison'][subject],native_frames=packages['primary_full_body'],
            supplemental_views=packages['face_only'],heldout_views=packages['pose_conditioned_holdout'],
            fixed_actual_portable_views=50,static_camera_gt=False,
            selected_image_records=len(data['selected_frame_panels']),original_records_image_only=all(
                Path(item['archive']['path']).suffix.lower()=='.jpg' for item in data['images']))
    destination = args.delivery/'review-receipt.json'
    if destination.exists():
        require(read(destination)==receipt,'Preserve the existing combined R21 QA receipt.')
    else:
        with destination.open('x',encoding='utf-8') as output:
            json.dump(receipt,output,indent=2,ensure_ascii=False,allow_nan=False)
            output.write('\n')
    print(json.dumps(descriptor(destination),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
