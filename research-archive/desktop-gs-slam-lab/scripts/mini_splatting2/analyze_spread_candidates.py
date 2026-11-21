"""Read-only native/head/extra-frame summary for the predeclared variance stage.

All candidates use seed0. A directional result is a reason to inspect and repeat,
never proof of generalization, true surface geometry or a solved human model.
"""
import ast
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
from PIL import Image
from evaluate_head_detail import metric

SCRIPTS = Path(__file__).parent
ROOT = SCRIPTS.parents[1] / 'outputs/mini-splatting2-mip-depth-001'
NAMES = {'reference0':'native-depthguide003-18000-s0',
    'control0':'native-depthspread000-18000-s0',
    'spread0030':'native-depthspread003-18000-s0',
    'spread0100':'native-depthspread010-18000-s0'}


def mean_dict(rows):
    return {key:statistics.mean(row[key] for row in rows) for key in rows[0]}


def delta(a, b):
    return {key:a[key]-b[key] for key in a}


def rgb(path):
    return np.asarray(Image.open(path).convert('RGB'), dtype='float32') / 255


def main():
    target = ROOT/'native-spread-candidate-review.json'
    assert not target.exists()
    assert json.loads((ROOT/'native-spread-queue.json').read_text())['status'] == 'complete'
    assignment=next(n for n in ast.parse((SCRIPTS/'local_refinement.py').read_text()).body
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SUBJECT_BOX' for t in n.targets))
    boxes=ast.literal_eval(assignment.value)
    manifests={k:json.loads((ROOT/v/'manifest.json').read_text()) for k,v in NAMES.items()}
    native={k:json.loads((ROOT/v/'evaluation-summary.json').read_text()) for k,v in NAMES.items()}
    frames=[r['frame'] for r in native['control0']['native_rows']]
    assert all([r['frame'] for r in v['native_rows']]==frames for v in native.values())
    rows={k:[] for k in NAMES};fingerprints={}
    for i,frame in enumerate(frames):
        rel=Path('model/test/ours_18000');filename=f'{i:05d}.png'
        gtpath=ROOT/NAMES['control0']/rel/'gt'/filename
        gtsha=hashlib.sha256(gtpath.read_bytes()).hexdigest();gt=rgb(gtpath)
        h,w=gt.shape[:2];b=boxes[int(frame)]
        x0,y0,x1,y1=[round(v*s) for v,s in zip(b,[w,h,w,h])]
        y1=y0+round((y1-y0)/3)
        for key,name in NAMES.items():
            assert hashlib.sha256((ROOT/name/rel/'gt'/filename).read_bytes()).hexdigest()==gtsha
            path=ROOT/name/rel/'renders'/filename
            pred=rgb(path);assert pred.shape==gt.shape
            fingerprints[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            rows[key].append(dict(frame=frame,box=[x0,y0,x1,y1],
                head=metric(pred[y0:y1,x0:x1],gt[y0:y1,x0:x1])))
    head={k:mean_dict([r['head'] for r in rr]) for k,rr in rows.items()}
    extra=json.loads((ROOT/'additional-frame-spread-004/summary.json').read_text())
    geometry=json.loads((ROOT/'native-spread-geometry-001/summary.json').read_text())
    spread=json.loads((ROOT/'depth-spread-candidates-002/summary.json').read_text())
    assert extra['status']==geometry['status']==spread['status']=='complete'
    models={}
    for key,name in NAMES.items():
        models[key]=dict(run=name,seed=0,gaussians=native[key]['gaussians'],
            native=native[key]['native_mean'],head=head[key],head_rows=rows[key],
            additional24=extra['models'][key]['aggregate'],
            geometry=geometry['models'][name],spread=spread['aggregate'][key],
            opacity_coverage=mean_dict([r['opacity_coverage'] for r in spread['rows'][key]]),
            unmasked_rgb_mae=mean_dict([r['unmasked_rgb_mae'] for r in spread['rows'][key]]))
    comparisons={}
    control=models['control0']
    for key,model in models.items():
        if key=='control0':continue
        comparisons[key]=dict(native=delta(model['native'],control['native']),
            head=delta(model['head'],control['head']),
            additional24={div:delta(model['additional24'][div],control['additional24'][div]) for div in control['additional24']},
            common_head_relative_spread=model['spread']['head']['mean_view_median']-control['spread']['head']['mean_view_median'],
            opacity_coverage=delta(model['opacity_coverage'],control['opacity_coverage']),
            per_frame_head_psnr=[dict(frame=a['frame'],delta=a['head']['psnr']-b['head']['psnr'])
                for a,b in zip(model['head_rows'],control['head_rows'])])
    replay=dict(compared=['reference0','control0'],
        raw_ply_identical=manifests['reference0']['ply_sha256']==manifests['control0']['ply_sha256'],
        baked_ply_identical=manifests['reference0']['baked_ply_sha256']==manifests['control0']['baked_ply_sha256'],
        native_difference=delta(control['native'],models['reference0']['native']),
        head_difference=delta(control['head'],models['reference0']['head']),
        interpretation='Zero variance control tests the new wrapper. Repeated GPU training may not be byte-deterministic; report its measured discrepancy.')
    result=dict(status='complete',created=time.time(),models=models,compared_to_control=comparisons,
        zero_weight_replay=replay,render_sha256=fingerprints,
        protocol=dict(head_region='same top third of frozen person rectangle as prior audit; includes shoulder/background',
            scope='one seed, one work video, repeatedly inspected development views; no independent final test',
            native_reference='identical real RGB reference files verified by hash',
            gradient='Sobel difference from real reference; lower is more faithful',
            spread='alpha-normalized camera-z at Gaussian centers; shared opacity intersection plus per-model coverage',
            geometry='shared sparse SfM and predicted-prior consistency, not measured depth ground truth',
            comparison='all three variance weights fixed before candidate results; mean-depth weight fixed at .003',
            next='inspect real crops and fixed camera-path videos; repeat a useful candidate before acceptance'),
        accepted=False,repeat_required=True,independent_depth_gt=False,visual_review_pending=True)
    target.write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(replay=replay,compared_to_control=comparisons),indent=2))


if __name__=='__main__':
    main()
