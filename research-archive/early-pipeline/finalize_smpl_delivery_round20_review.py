"""Bind the two finished R20 independent archives and their diagnostics page."""

import argparse
import json
from pathlib import Path
from urllib.parse import quote
from zipfile import ZipFile

import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import publish_smpl_delivery_round20_report as publisher


reviewer = publisher.reviewer
pin,read,require,sha,descriptor = reviewer.pin,reviewer.read,reviewer.require,reviewer.sha,reviewer.descriptor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit',required=True)
    parser.add_argument('--findings',type=Path,required=True)
    parser.add_argument('--delivery',type=Path,default=reviewer.DELIVERY)
    parser.add_argument('--page',type=Path,default=publisher.shared.PAGE)
    args = parser.parse_args()
    receipt = dict(schema='shuolee-round20-independent-review-v1',state='native_review_and_archives_verified',
                   commit=args.commit,baseline_round15=reviewer.r19.BASE_COMMIT,baseline_round18=reviewer.r19.R18_COMMIT,
                   baseline_round19=reviewer.R19_COMMIT,
                   quality_accepted=False,findings=descriptor(args.findings),report=descriptor(args.page),subjects={})
    anchor = f'smpl-delivery-round20-{args.commit[:7]}'
    html = args.page.read_text(encoding='utf-8')
    require(html.count(f'id="{anchor}"')==1,'The completed report anchor is missing or duplicate.')
    receipt['report']['anchor'] = anchor
    receipt['report']['url'] = 'http://127.0.0.1:8870/diagnostics/video10-global-initializer/index.html#'+quote(anchor)
    findings = read(args.findings)
    require(findings['commit']==args.commit and findings['quality_accepted'] is False,
            'The findings are not the matching visual review.')
    for subject in reviewer.SUBJECTS:
        archive = args.delivery/subject/'receipt.json'
        loaded = publisher.load(subject,args.commit,args.delivery)
        data,review = loaded['receipt'],loaded['review']
        require(findings['comparison_receipt_sha256'][subject]==data['review_sha256'],
                'The findings refer to a different visual review.')
        packages = data['packages']
        for key in ('scene','bound_replay','pose_conditioned_evaluation','named_pose_roster'):
            pin(packages[key],packages[key+'_sha256'])
        bound_count = 14+4*packages['face_only'] if packages['face_only'] else 10
        with ZipFile(packages['bound_replay']) as bundle:
            require(len(bundle.namelist())==bound_count and bundle.testzip() is None,
                    'The standard bound package or admitted face sidecar differs.')
        with ZipFile(packages['pose_conditioned_evaluation']) as bundle:
            require(len(bundle.namelist())==packages['evaluation_member_count'] and bundle.testzip() is None,
                    'The pose-conditioned evaluation package is incomplete.')
        publisher.older.statuses(findings['subjects'][subject])
        receipt['subjects'][subject] = dict(archive=descriptor(archive),points=data['points'],reference=data['reference'],
            review=dict(file=data['review'],sha256=data['review_sha256']),packages=packages,
            portable_ply=data['portable_ply'],portable_camera=data['portable_camera'],camera_provenance=data['camera_provenance'],
            comparison_with_delivered_r15=findings['delivered_r15_comparison'][subject],
            native_frames=packages['primary_full_body'],supplemental_views=packages['face_only'],
            heldout_views=packages['pose_conditioned_holdout'],pose_conditioned_independent_replay=True,
            fully_unobserved_prediction=False,head_fit=review['fit_audit']['summary'],
            selected_image_records=len(data['selected_frame_panels']),protected_face=data['protected_face'],
            original_records_image_only=all(Path(item['archive']['path']).suffix.lower()=='.jpg' for item in data['images']))
    destination = args.delivery/'review-receipt.json'
    if destination.exists():
        require(read(destination)==receipt,'Preserve the existing combined QA receipt.')
    else:
        with destination.open('x',encoding='utf-8') as file:
            json.dump(receipt,file,indent=2,ensure_ascii=False,allow_nan=False)
            file.write('\n')
    print(json.dumps(descriptor(destination),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
