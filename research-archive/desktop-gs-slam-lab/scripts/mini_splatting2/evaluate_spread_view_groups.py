"""Freeze observed distance/pitch groups before reading variance candidate scores."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]/'outputs/mini-splatting2-mip-depth-001'
OUT=ROOT/'native-spread-observed-groups-001'


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def prepare():
    OUT.mkdir(exist_ok=False)
    audit_path=ROOT/'capture-audit.json';camera_path=ROOT/'additional-frames-001/cameras.json'
    audit=read(audit_path);train=[c for c in audit['cameras'] if c['split']=='train']
    axes=['radius_sfm','optical_pitch_deg']
    thresholds={axis:statistics.median(c[axis] for c in train) for axis in axes}
    focus=np.asarray(audit['focus']);up=np.asarray(audit['approximate_up']);cameras={}
    for c in read(camera_path):
        forward=np.asarray(c['rotation'])[:,2]
        cameras[c['name']]=dict(source_index=c['source_index'],
            radius_sfm=float(np.linalg.norm(np.asarray(c['position'])-focus)),
            optical_pitch_deg=float(np.degrees(np.arcsin(np.clip(forward@up,-1,1)))))
    groups={}
    for axis,threshold in thresholds.items():
        for side,lower in [('lower',True),('upper',False)]:
            frames=[name for name,c in cameras.items() if (c[axis]<threshold)==lower]
            assert frames,'Empty observed group; do not invent a mean'
            values=[cameras[name][axis] for name in frames]
            groups[axis+'_'+side]=dict(axis=axis,side=side,frames=frames,count=len(frames),
                actual_range=[min(values),max(values)],threshold=threshold)
    protocol=dict(created=time.time(),thresholds_from_training_cameras=thresholds,
        groups=groups,cameras=cameras,seed=0,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [audit_path,camera_path]},
        fixed_before_inspecting_candidate_scores=True,
        concurrent_training_and_evaluation_may_already_be_running=True,
        caveats=['Observed distance and pitch groups from the same video, not wide-view extrapolation',
            'SfM scale is arbitrary; pitch uses mean camera up, not measured gravity',
            'Distance, pitch, scene content and time co-vary; grouped differences are descriptive',
            'Image-resolution divisors measure downsampling behavior, not physically moving the camera'])
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    print(json.dumps(dict(thresholds=thresholds,groups=groups),indent=2))


def evaluate():
    target=OUT/'summary.json';assert not target.exists()
    protocol=read(OUT/'protocol.json')
    for name,sha in protocol['source_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha
    extra=read(ROOT/'additional-frame-spread-004/summary.json');assert extra['status']=='complete'
    models={}
    for key,result in extra['models'].items():
        groups={}
        for group,spec in protocol['groups'].items():
            scales={}
            for divisor in [1,2,4,8]:
                rows=[r for r in result['rows'] if r['divisor']==divisor and r['frame'] in spec['frames']]
                assert len(rows)==spec['count'] and {r['frame'] for r in rows}==set(spec['frames'])
                value=dict(full_psnr=statistics.mean(r['full']['psnr'] for r in rows),
                    person_psnr=statistics.mean(r['subject_crop']['psnr'] for r in rows))
                if divisor==1:value.update(ssim=statistics.mean(r['ssim'] for r in rows),
                    lpips_alex=statistics.mean(r['lpips_alex'] for r in rows))
                scales[str(divisor)]=value
            groups[group]=scales
        models[key]=groups
    changes={}
    for key in ['reference0','spread0030','spread0100']:
        changes[key]={group:{scale:{metric:value-models['control0'][group][scale][metric]
            for metric,value in values.items()} for scale,values in scales.items()}
            for group,scales in models[key].items()}
    out=dict(status='complete',created=time.time(),protocol=protocol,models=models,
        paired_changes_from_zero_control=changes,independent_extrapolation_validated=False)
    target.write_text(json.dumps(out,indent=2));print(json.dumps(changes,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    prepare() if args.prepare else evaluate()
