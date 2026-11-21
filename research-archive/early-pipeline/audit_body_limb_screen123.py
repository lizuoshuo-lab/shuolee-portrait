"""CPU-only limb screening with the fitter's existing score contract.

Immutable snapshot, all registered frames, fixed rules, no frame exclusion.
Raw RTMW heatmap peaks are neither probabilities nor visibility labels.
"""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import label
from smplify_score_contract123 import relative_confidence

ROOT = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/demo-vhap-initialization/body-gaussian-upstream-offset')
OBS = Path('/opt/gs-slam-lab/shuolee-local-surface-123/demo-vhap-fps3/data/observations.npz')
OUT = ROOT/'limb-screen-v2'
STEP = 60000
CHAINS = {'left_arm': [5, 7, 9], 'right_arm': [6, 8, 10],
          'left_leg': [11, 13, 15], 'right_leg': [12, 14, 16]}
POLICY = dict(version=2, step=STEP, minimum_relative_confidence=.5,
    line_width_over_person_height=.03, minimum_target_pixels=32,
    maximum_soft_uncovered=.15, connectivity_alpha=.5,
    relative_score_source='Exact existing smplify_score_contract123.relative_confidence; not calibrated visibility',
    threshold_source='Keep previous .5 score gate, .03 height tube, 32 pixels and .15 missing limit; alpha .5 defines connectivity, not quality acceptance',
    scope='One all59 read-only screen of the already delivered 60000-step snapshot',
    actions='Warnings and unknowns only. No exclusion, retraining, threshold search or model promotion.',
    limitations='2D mask/alpha connectivity cannot detect wrong depth, distinguish overlap from anatomical connection, or establish visibility. Human review remains necessary.')


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, value):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
def stamp(): return datetime.now(timezone.utc).isoformat()


def support(shape, points, width):
    canvas = Image.new('L', (shape[1], shape[0]), 0)
    ImageDraw.Draw(canvas).line([tuple(p) for p in points], fill=255, width=width, joint='curve')
    return np.asarray(canvas) > 0


def connected(mask, endpoints, radius):
    """Both endpoint neighborhoods must touch the same 8-connected component."""
    components, _ = label(mask, np.ones((3, 3), dtype=np.uint8))
    groups = []
    h, w = mask.shape
    for x, y in endpoints:
        x0, x1 = max(0, int(np.floor(x-radius))), min(w, int(np.ceil(x+radius))+1)
        y0, y1 = max(0, int(np.floor(y-radius))), min(h, int(np.ceil(y+radius))+1)
        if x0 >= x1 or y0 >= y1: return False
        yy, xx = np.ogrid[y0:y1, x0:x1]
        disk = (xx-x)**2 + (yy-y)**2 <= radius**2
        ids = set(np.unique(components[y0:y1, x0:x1][disk])) - {0}
        groups.append(ids)
    return bool(groups[0] & groups[1])


def controls():
    s = np.zeros((40, 80), bool); s[15:26, 5:75] = True
    endpoints = np.array([[8, 20], [71, 20]])
    assert connected(s, endpoints, 5)
    broken = s.copy(); broken[:, 35:40] = False
    assert not connected(broken, endpoints, 5)
    assert not connected(np.zeros_like(s), endpoints, 5)
    # Brightness is deliberately absent: black clothing with alpha=1 stays connected.
    assert connected(s & (np.ones(s.shape) >= POLICY['connectivity_alpha']), endpoints, 5)
    scores = np.full((2, 133), 5.0); boxes = np.array([[0, 0, 200, 1000]]*2)
    flip = np.zeros_like(scores); flip[1, 7] = 80
    c, _ = relative_confidence(scores, boxes, flip)
    assert scores[1, 7] > .5 and c[1, 7] < .5 and np.all((c >= 0) & (c <= 1))
    return dict(intact_connected=True, severed_not_connected=True, empty_not_connected=True,
                black_opaque_not_a_hole=True, raw_score_cannot_establish_reliability=True)


def plan():
    inputs = [ROOT/'data/training-inputs.json', OBS,
        ROOT/'gaussian-run/step-060000/result.json', ROOT/'gaussian-run/step-060000/visual-review.json',
        ROOT/'gaussian-run/step-030000/render-coverage/result.json',
        Path(__file__).with_name('smplify_score_contract123.py')]
    value = dict(updated=stamp(), policy=POLICY, input_hashes={str(p): sha(p) for p in inputs},
        script_sha256=sha(__file__), cpu_controls=controls(), frames_budget=59,
        optimization_steps=0, current_training_changed=False,
        note='Legacy30k result retained. New60k values are not attributed solely to screening because checkpoint differs.')
    if (OUT/'plan.json').exists():
        old = read(OUT/'plan.json')
        assert old['policy'] == value['policy'] and old['input_hashes'] == value['input_hashes']
        assert old['script_sha256'] == value['script_sha256']
    else: save(OUT/'plan.json', value)
    print('LIMB_SCREEN_V2_PLAN_REGISTERED', flush=True)


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    p = read(OUT/'plan.json'); assert p['policy'] == POLICY and p['script_sha256'] == sha(__file__)
    for path, h in p['input_hashes'].items(): assert sha(path) == h, path
    assert not (OUT/'result.json').exists(), 'Review existing result; do not repeat.'
    snap = ROOT/'gaussian-run/step-060000'; replay = snap/'render-coverage/native-replay.json'
    assert read(replay)['passed'] and read(replay)['frames'] == 59
    rows = sorted(read(ROOT/'data/training-inputs.json')['records'], key=lambda r: r['time_seconds'])
    with np.load(OBS, allow_pickle=False) as z: obs = dict(z)
    conf, _ = relative_confidence(obs['scores'], obs['boxes'], obs['flip_disagreement'])
    slots = {str(f): i for i, f in enumerate(obs['frame_ids'])}
    assert len(rows) == len(slots) == 59 and set(slots) == {r['frame'] for r in rows}
    records = []
    for r in rows:
        i = slots[r['frame']]; points = obs['keypoints'][i]; raw = obs['scores'][i]
        ap = snap/'render-coverage'/(r['frame']+'_alpha.png')
        mask = np.asarray(Image.open(r['mask']).convert('L')) > 127
        alpha = np.asarray(Image.open(ap), dtype=float)/65535
        assert alpha.shape == mask.shape and sha(r['mask']) == r['mask_sha256']
        width = max(6, round(POLICY['line_width_over_person_height']*(r['initial_box_xyxy'][3]-r['initial_box_xyxy'][1])))
        regions = {}; warnings = []; unknowns = []
        for name, ids in CHAINS.items():
            groups = [(name, ids)] + [(name+'_'+part, pair) for part, pair in zip(('proximal','distal'), (ids[:2],ids[1:]))]
            for key, joint_ids in groups:
                pts = points[joint_ids]; score = float(conf[i, joint_ids].min())
                row = dict(raw_score_min=float(raw[joint_ids].min()), relative_confidence_min=score,
                           observation_visibility='unknown', target_pixels=None, soft_uncovered=None)
                if not np.isfinite(pts).all() or score < POLICY['minimum_relative_confidence']:
                    row.update(status='unknown', reason='unreliable_observation'); unknowns.append(key); regions[key]=row; continue
                tube = support(mask.shape, pts, width); target = tube & mask
                count = int(target.sum()); row['target_pixels'] = count
                if count < POLICY['minimum_target_pixels']:
                    row.update(status='unknown', reason='insufficient_target_support');unknowns.append(key);regions[key]=row;continue
                missing = float((1-alpha[target]).mean()); row['soft_uncovered'] = missing
                reasons = []
                if missing > POLICY['maximum_soft_uncovered']: reasons.append('segment_undercoverage')
                if len(joint_ids) == 2:
                    yy, xx = np.where(tube)
                    x0,x1,y0,y1=int(xx.min()),int(xx.max())+1,int(yy.min()),int(yy.max())+1
                    end = pts - [x0, y0]
                    target_connected = connected(target[y0:y1,x0:x1], end, width/2)
                    pred_connected = connected((target & (alpha >= POLICY['connectivity_alpha']))[y0:y1,x0:x1], end, width/2)
                    row.update(target_mask_connected=target_connected, rendered_mask_connected=pred_connected)
                    if target_connected and not pred_connected: reasons.append('rendered_connection_missing')
                    if not target_connected:
                        row['connectivity_measurement'] = 'unknown_target_support'; unknowns.append(key+':connectivity')
                row.update(status='review_required' if reasons else 'no_configured_alarm', reasons=reasons)
                warnings.extend(key+':'+reason for reason in reasons); regions[key] = row
        records.append(dict(frame=r['frame'],alpha_sha256=sha(ap),regions=regions,
            warnings=warnings,unknowns=unknowns,quality_accepted=False))
    result = dict(updated=stamp(), frames=59,step=STEP, policy=POLICY,plan_sha256=sha(OUT/'plan.json'),
        replay_sha256=sha(replay),records=records, cpu_controls=controls(),
        raw_score_range=[float(obs['scores'].min()), float(obs['scores'].max())],
        relative_confidence_range=[float(conf.min()),float(conf.max())],
        warning_frames=[r['frame'] for r in records if r['warnings']],
        unknown_frames=[r['frame'] for r in records if r['unknowns']],
        quality_accepted=False,auto_excluded_frames=[],current_training_changed=False)
    save(OUT/'result.json',result)
    print(json.dumps({k:result[k] for k in ('frames','warning_frames','unknown_frames','raw_score_range')},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['plan','run'])
    args=p.parse_args();plan() if args.mode=='plan' else main()
