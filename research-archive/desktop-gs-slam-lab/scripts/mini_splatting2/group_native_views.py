"""Pair metrics with observed camera distance/pitch, without inventing new GT."""
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2]/'outputs/mini-splatting2-mip-depth-001'

def main():
    audit=json.loads((ROOT/'capture-audit.json').read_text());cams=audit['cameras']
    train=[c for c in cams if c['split']=='train'];test={c['name']:c for c in cams if c['split']=='test'}
    thresholds={axis:statistics.median(c[axis] for c in train) for axis in ['radius_sfm','optical_pitch_deg']}
    models={}
    for p in sorted(ROOT.glob('native-*-18000-s*/evaluation-summary.json')):
        r=json.loads(p.read_text());groups={}
        for axis,threshold in thresholds.items():
            for label,less in [('lower',True),('upper',False)]:
                rows=[row for row in r['native_rows'] if (test[row['frame']][axis]<threshold)==less]
                if rows:
                    groups[axis+'_'+label]=dict(frames=[x['frame'] for x in rows],count=len(rows),
                        actual_range=[min(test[x['frame']][axis] for x in rows),max(test[x['frame']][axis] for x in rows)],
                        full_psnr=statistics.mean(x['full']['psnr'] for x in rows),crop_psnr=statistics.mean(x['subject_crop']['psnr'] for x in rows))
        models[p.parent.name]=groups
    out=dict(status='observed_view_diagnostic',thresholds_from_training_cameras=thresholds,models=models,
        reference='estimated SfM subject focus and mean camera up; not metric depth or measured gravity',
        coverage=dict(radius_ratio=audit['radius_ratio'],pitch_range=audit['optical_pitch_min_max_deg'],elevation_range=audit['elevation_min_max_deg']),
        limitations=['only eight development holdouts on one orbit','distance, pitch and capture time are correlated; do not infer a causal distance/pitch improvement from grouped scores',
            'all cameras share SfM; no independent calibration holdout','no ground truth for the extrapolated +/-25degree and0.5x..1.5x radius viewer paths'])
    (ROOT/'native-observed-view-groups.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out))

if __name__=='__main__':main()
