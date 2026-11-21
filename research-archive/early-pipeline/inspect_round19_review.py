"""Summarize measured source, opacity, side and protected-stage metrics before visual sign-off."""

import argparse
import json
from pathlib import Path
from statistics import mean

import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import review_smpl_delivery_round19 as reviewer


read = reviewer.read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject',choices=reviewer.SUBJECTS)
    args = parser.parse_args()
    path = reviewer.root_for(args.subject)/'view-comparison/result.json'
    value = read(path)
    stage = value['protected_stage']
    audit = read(reviewer.pin(stage['optimizer_audit']['file'],stage['optimizer_audit']['sha256']))
    arms = [value['old'],value['mid'],stage['main'],value['new']]
    records = dict(
        subject=args.subject,decision=audit['state'],attempted=audit['attempted_updates'],committed=audit['committed_updates'],
        protection_failures=audit['unprotected']['measurement']['failed_frames'],
        objectives={key: [part[key] for part in (audit['baseline'],audit['unprotected']['measurement'],audit['final'])]
                    for key in ('objective','original_objective','supplemental_objective')},
        full_native={key:[mean(row[key] for row in arm['float_native'].values()) for arm in arms]
                     for key in ('foreground_rgb_mae_raw','foreground_hole_fraction_at_half','exterior_spill_pixels_at_half')},
        regions={row['frame']:{name:[[arm['native'][i]['region_metrics'][name][key] for arm in arms]
                                    for key in ('black_rgb_mae','alpha_u8_lt_243')]
                               for name in ('face','ear_016_f051','dark_source_eye','dark_source_hair','nose','mouth')
                               if name in row['region_metrics']}
                 for i,row in enumerate(arms[0]['native'])},
        supplemental={frame:[[arm['supplement_native'][frame]['foreground_rgb_mae'] for arm in arms],
                             [arm['supplement_native'][frame]['foreground_alpha']['fraction_below_095'] for arm in arms]]
                      for frame in arms[0]['supplement_native']},
        physical_near={role:[arm['physical_near'][role]['face_proxy_alpha']['fraction_below_095'] for arm in arms]
                       for role in arms[0]['physical_near']},
        common_side_bright={key:[value['brightness_diagnostic']['common_static_rotated_side'][side][key]
                                for side in ('old','mid','main','new')]
                           for key in ('bright_pixels','largest_bright_region_pixels')},
        native_torso={role:{key:[record['arms'][side][key] for side in ('old','mid','main','new')]
                           for key in ('rgb_mae','alpha_half_hole_pixels','alpha_half_spill_pixels','excess_bright_pixels')}
                      for role,record in value['native_torso_band'].items()},
        stage_pixel_changes={key:sum(row[key] for row in stage['native_pair_deltas'].values()) for key in ('rgb_changed','alpha_changed')},
        image_paths={panel['view']:str(path.parent/panel['file']) for panel in value['panels']})
    print(json.dumps(records,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()

