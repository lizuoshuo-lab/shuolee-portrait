"""Use pre-existing training thresholds for actual observed distance/pitch groups."""
import json
from pathlib import Path
import statistics
import numpy as np

ROOT=Path(__file__).resolve().parents[2]/'outputs/mini-splatting2-mip-depth-001'


def main():
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    audit=read(ROOT/'capture-audit.json');train=[c for c in audit['cameras'] if c['split']=='train']
    thresholds={a:statistics.median(c[a] for c in train) for a in ['radius_sfm','optical_pitch_deg']}
    up=np.array(audit['approximate_up']);focus=np.array(audit['focus']);camera={}
    for c in read(ROOT/'additional-frames-001/cameras.json'):
        v=np.array(c['rotation'])[:,2]
        camera[c['name']]=dict(radius_sfm=float(np.linalg.norm(np.array(c['position'])-focus)),
            optical_pitch_deg=float(np.degrees(np.arcsin(np.clip(v@up,-1,1)))),source_index=c['source_index'])
    current=read(ROOT/'additional-frame-depth-evaluation-002/summary.json')['models']
    original=read(ROOT/'additional-frame-evaluation-001/summary.json')['models']
    metrics={'combo0':original['depthmask'],**current};models={}
    for key,result in metrics.items():
        groups={};rows=[r for r in result['rows'] if r['divisor']==1]
        for axis,threshold in thresholds.items():
            for label,less in [('lower',True),('upper',False)]:
                selected=[r for r in rows if (camera[r['frame']][axis]<threshold)==less]
                assert selected,'Empty group; report insufficient coverage instead of inventing means'
                groups[axis+'_'+label]=dict(frames=[r['frame'] for r in selected],count=len(selected),
                    actual_range=[min(camera[r['frame']][axis] for r in selected),max(camera[r['frame']][axis] for r in selected)],
                    full_psnr=statistics.mean(r['full']['psnr'] for r in selected),
                    crop_psnr=statistics.mean(r['subject_crop']['psnr'] for r in selected))
        models[key]=groups
    out=dict(status='complete',thresholds_from_original_training_views=thresholds,cameras=camera,models=models,
        caveats=['SfM units are arbitrary; orientation uses mean camera up, not measured gravity',
            'distance, pitch and video time are correlated; groups are descriptive, not causal tests',
            'same-video already inspected development frames; not wide-pitch or independent capture evidence'])
    dest=ROOT/'native-depth-observed-groups.json'
    assert not dest.exists(),'Preserve earlier group analysis'
    dest.write_text(json.dumps(out,indent=2));print(json.dumps(models,indent=2))


if __name__=='__main__':main()
