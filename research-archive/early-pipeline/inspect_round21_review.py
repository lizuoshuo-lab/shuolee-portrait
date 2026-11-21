"""Inspect exact completed R21 view statistics without assigning human quality labels."""

import argparse
import json
from statistics import mean

import review_smpl_delivery_round21 as review


def inspect(subject):
    path = review.root_for(subject)/'view-comparison/result.json'
    result = review.read(path)
    plan = review.read(review.pin(path.with_name('plan.json'),result['plan_sha256']))
    item = result['static_grid']
    grid = review.read(review.pin(item['file'],item['sha256']))
    cameras = plan['static_grid']['camera_roster']
    scopes = {}
    for distance in ('normal','close'):
        for tilt,select in (('all',lambda p:True),('flat',lambda p:p==0),('tilted',lambda p:p!=0),
                            ('high',lambda p:p==30),('low',lambda p:p==-30)):
            scopes[f'{distance}_{tilt}'] = [key for key,row in cameras.items()
                                            if row['distance_class']==distance and select(row['pitch_deg'])]
    def static_value(row):
        outer = row['anatomical_exterior_ring']
        return dict(face_transmission=row['opacity']['visible_face_proxy']['mean_transmission'],
                    face_alpha_below_095=row['opacity']['visible_face_proxy']['low_alpha_095_fraction'],
                    exterior_alpha_mass_per_pixel=outer['alpha_mass']/outer['pixels'],
                    exterior_alpha_ge_01=outer['alpha_ge_010_pixels'],
                    largest_exterior_alpha_ge_01=outer['max_connected_alpha_ge_010_pixels'])
    values = {arm:{key:static_value(row) for key,row in payload['records'].items()} for arm,payload in grid['arms'].items()}
    aggregates = {scope:{arm:{name:mean(v[key][name] for key in keys) for name in v[keys[0]]}
                         for arm,v in values.items()} for scope,keys in scopes.items()}
    worst = {}
    for before,after in (('r15','r21_10'),('r15','r21'),('r20','r21_10'),('r20','r21'),
                         ('before10','r21_10'),('before18','r21'),('r21_10','r21')):
        key = f'{before}__{after}'
        change = {camera:{metric:v[metric]-values[before][camera][metric] for metric in v}
                  for camera,v in values[after].items()}
        worst[key] = {metric:dict(improved=sum(row[metric]<0 for row in change.values()),
            regressed=sum(row[metric]>0 for row in change.values()),
            largest_increase=max((row[metric],camera) for camera,row in change.items()),
            largest_reduction=min((row[metric],camera) for camera,row in change.items()))
            for metric in next(iter(change.values()))}
    native = {}
    for side in review.SIDES:
        row = result[side]
        original = row['float_native'].values()
        native[side] = dict(full_body={key:mean(item[key] for item in original) for key in (
            'foreground_rgb_mae_raw','foreground_hole_fraction_at_half','exterior_spill_pixels_at_half')},
            face_only={name:{key:item[key] for key in ('foreground_rgb_mae','foreground_alpha')}
                       for name,item in row['supplement_native'].items()},
            holdout={name:dict(rgb_mae=item['foreground_rgb_mae'],alpha_below_095=item['alpha']['fraction_below_095'])
                     for name,item in row['heldout_native'].items()})
    return dict(subject=subject,receipt=review.descriptor(path),static=aggregates,paired_direction=worst,native=native,
                garment=result['native_torso_band'],head_fit=result['fit_audit']['summary'],
                checkpoints=result['training_curve']['checkpoints'])


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject',choices=review.SUBJECTS)
    args = parser.parse_args()
    print(json.dumps(inspect(args.subject),ensure_ascii=False,allow_nan=False,indent=2))
