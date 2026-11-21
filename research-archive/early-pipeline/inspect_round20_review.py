"""Summarize R20 native quality and pose-conditioned measurements for visual review."""

import argparse
import json
from pathlib import Path
from statistics import mean
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import review_smpl_delivery_round20 as reviewer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject',choices=reviewer.SUBJECTS)
    args = parser.parse_args()
    path = reviewer.root_for(args.subject)/'view-comparison/result.json'
    q = reviewer.read(path)
    arms = [q[side] for side in reviewer.SIDES]
    fit = q['fit_audit']
    retained = q['heldout']['frames']
    regions = ('face','ear_016_f051','dark_source_eye','dark_source_hair','nose','mouth')
    data = dict(subject=args.subject,fit=fit['summary'],heldout=dict(frames=[x['frame'] for x in retained],
             scores={row['frame']:[[(arm['heldout_native'][row['frame']][key] if key!='alpha' else
                          arm['heldout_native'][row['frame']][key]['fraction_below_095']) for arm in arms]
                          for key in ('foreground_rgb_mae','alpha','exterior_half_alpha_pixels')] for row in retained}),
        full_native={key:[mean(item[key] for item in arm['float_native'].values()) for arm in arms]
            for key in ('foreground_rgb_mae_raw','foreground_hole_fraction_at_half','exterior_spill_pixels_at_half')},
        native_regions={row['frame']:{name:[[arm['native'][i]['region_metrics'][name][key] for arm in arms]
                         for key in ('black_rgb_mae','alpha_u8_lt_243')] for name in regions if name in row['region_metrics']}
                         for i,row in enumerate(arms[0]['native'])},
        face_supplement={row['frame']:[[arm['supplement_native'][row['frame']]['foreground_rgb_mae'] for arm in arms],
                        [arm['supplement_native'][row['frame']]['foreground_alpha']['fraction_below_095'] for arm in arms]]
                        for row in q['supplement']['frames']},
        physical_near={role:[arm['physical_near'][role]['face_proxy_alpha']['fraction_below_095'] for arm in arms]
                       for role in arms[0]['physical_near']},
        common_side_bright={key:[q['brightness_diagnostic']['common_static_rotated_side'][side][key] for side in reviewer.SIDES]
                            for key in ('bright_pixels','largest_bright_region_pixels')},
        native_torso={role:{key:[item['arms'][side][key] for side in reviewer.SIDES]
                            for key in ('rgb_mae','alpha_half_hole_pixels','alpha_half_spill_pixels','excess_bright_pixels')}
                            for role,item in q['native_torso_band'].items()},
        fit_ear_source={frame:{name:[item[stage][name]['rms_px'] for stage in ('before','after')] for name in ('right_ear','left_ear')}
                        for frame in ('frame_000051','frame_000055')
                        if (item:=fit['frames'].get(frame,fit['candidates'].get(frame))) is not None},
        image_paths={panel['view']:str(path.parent/panel['file']) for panel in q['panels']})
    print(json.dumps(data,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
