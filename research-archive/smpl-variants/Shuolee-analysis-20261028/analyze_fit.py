"""Read-only, per-frame SMPL-X projection diagnostics in decoded-image pixels."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import shutil

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from smplx.joint_names import JOINT_NAMES


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(values):
    x = np.asarray(values, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    return dict(count=len(x), median=float(np.median(x)), p90=float(np.quantile(x, .9)),
                maximum=float(x.max()), above_5px=int((x > 5).sum()), above_10px=int((x > 10).sum())) if len(x) else dict(count=0)


def analyze(root, output, label):
    output.mkdir(parents=True, exist_ok=False)
    before = {p: sha(root / p) for p in ['fit/fitted.npz', 'fit/observation-weights.npz',
                                      'data/observations.npz', 'data/training-inputs.json']}
    fit = dict(np.load(root / 'fit/fitted.npz'))
    obs = dict(np.load(root / 'data/observations.npz'))
    weights = dict(np.load(root / 'fit/observation-weights.npz'))
    rows = read(root / 'data/training-inputs.json')['records']
    ids = [r['frame'] for r in rows]
    assert np.array_equal(fit['frame_ids'], ids) and np.array_equal(obs['frame_ids'], ids)
    E = np.asarray([r['w2c'] for r in rows], np.float64)
    K = np.asarray([r['K'] for r in rows], np.float64)
    def project(x):
        cam = np.einsum('nij,nkj->nki', E[:, :3, :3], np.asarray(x, np.float64)) + E[:, None, :3, 3]
        q = np.einsum('nij,nkj->nki', K, cam)
        assert (q[..., 2] > 0).all()
        return q[..., :2] / q[..., 2:] - .5
    template = np.load('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets/smplx/SMPLX_NEUTRAL.npz')
    triangles = fit['faces'][template['lmk_faces_idx']]
    face_world = (fit['vertices_world'][:, triangles] * template['lmk_bary_coords'][None, :, :, None]).sum(2)
    independent = {'coco17': project(fit['coco17_world']),
                   'joints': project(fit['joints_world']), 'face51': project(face_world)}
    parity = {key: float(np.max(np.abs(value - fit['projected_' + key]))) for key, value in independent.items()}
    feet_ids = [JOINT_NAMES.index(k) for k in ['left_big_toe', 'left_small_toe', 'left_heel', 'right_big_toe', 'right_small_toe', 'right_heel']]
    hand_names = ['wrist', 'thumb1', 'thumb2', 'thumb3', 'thumb', 'index1', 'index2', 'index3', 'index',
                  'middle1', 'middle2', 'middle3', 'middle', 'ring1', 'ring2', 'ring3', 'ring', 'pinky1', 'pinky2', 'pinky3', 'pinky']
    hand_ids = [JOINT_NAMES.index(side + '_' + name) for side in ['left', 'right'] for name in hand_names]
    regions = {
        'body': (independent['coco17'], weights['body_target'], weights['body'] * weights['visibility']),
        'face': (independent['face51'], obs['keypoints'][:, 40:91], weights['face']),
        'hands': (independent['joints'][:, hand_ids], obs['keypoints'][:, 91:133], weights['hands']),
        'feet': (independent['joints'][:, feet_ids], obs['keypoints'][:, 17:23], weights['feet']),
    }
    height = obs['boxes'][:, 3] - obs['boxes'][:, 1]
    metrics, frame_metrics = {}, []
    for name, (projected, target, w) in regions.items():
        error = np.linalg.norm(projected - target, axis=-1)
        keep = w > (.25 if name == 'body' else 0)
        metrics[name] = dict(used_observations=summary(error[keep]), stronger_weight_observations=summary(error[w > .25]),
                             normalized_by_person_height=summary((error / height[:, None])[keep]),
                             pixel_domain='decoded image; detector residual, not independent material truth')
        for i, row in enumerate(rows):
            stats = summary(error[i, keep[i]])
            frame_metrics.append(dict(frame=row['frame'], region=name, width=row['wh'][0], height=row['wh'][1], **stats))
    selected = np.unique(np.linspace(0, len(rows)-1, 6).astype(int))
    fig, axes = plt.subplots(len(selected), 3, figsize=(10, 2.2 * len(selected)), squeeze=False)
    for line, i in enumerate(selected):
        row = rows[i]
        image = np.asarray(Image.open(row['image']).convert('RGB'))
        p, y, w = regions['face']
        keep = w[i] > 0
        points = np.concatenate([p[i, keep], y[i, keep]])
        low = np.floor(points.min(0) - 20).astype(int)
        high = np.ceil(points.max(0) + 20).astype(int)
        low = np.maximum(low, 0)
        high = np.minimum(high, row['wh'])
        for col, title in enumerate(['Original', 'Detector observations used', 'SMPL-X projection']):
            ax = axes[line, col]
            ax.imshow(image)
            if col == 1:
                ax.scatter(y[i, keep, 0], y[i, keep, 1], s=12, facecolors='none', edgecolors='#00ef98', linewidths=.8)
            elif col == 2:
                ax.scatter(p[i, keep, 0], p[i, keep, 1], s=17, marker='+', c='#ff7d24', linewidths=.9)
            ax.set_xlim(low[0], high[0]); ax.set_ylim(high[1], low[1]); ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([])
            if line == 0: ax.set_title(title, fontsize=10)
        err = np.linalg.norm(p[i, keep] - y[i, keep], axis=1)
        axes[line, 0].set_ylabel(f'{row["frame"]}\nmedian {np.median(err):.2f} px', fontsize=9)
    fig.suptitle(f'{label} | fixed cameras and unchanged fitted model | face alignment', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, .98]); fig.savefig(output / 'face-alignment.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 10))
    edges = [(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
    for ax, i in zip(axes.flat, selected):
        row = rows[i]; image = np.asarray(Image.open(row['image']).convert('RGB'))
        ax.imshow(image)
        p, y, w = regions['body']
        for points, color in [(y[i], '#00ef98'), (p[i], '#ff7d24')]:
            for a,b in edges:
                ax.plot(points[[a,b],0], points[[a,b],1], c=color, linewidth=.8)
            ax.scatter(points[:,0],points[:,1],c=color,s=5)
        box = obs['boxes'][i]; margin = .06 * max(box[2:] - box[:2])
        ax.set_xlim(max(0,box[0]-margin), min(row['wh'][0],box[2]+margin))
        ax.set_ylim(min(row['wh'][1],box[3]+margin), max(0,box[1]-margin))
        ax.set_aspect('equal'); ax.set_title(row['frame'],fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f'{label} | green: 2D body targets; orange: SMPL-X projection',fontsize=12)
    fig.tight_layout(); fig.savefig(output/'body-alignment.png',dpi=150); plt.close(fig)
    with (output/'per-frame-errors.csv').open('w') as stream:
        keys=['frame','region','width','height','count','median','p90','maximum','above_5px','above_10px']
        writer=csv.DictWriter(stream,keys); writer.writeheader(); writer.writerows(frame_metrics)
    report=dict(label=label,run=str(root),video_sha256=read(root/'run.json')['video_sha256'],
                source_fit_sha256=read(root/'fit/protocol.json')['source_sha256'],frames=len(rows),
                decoded_wh=rows[0]['wh'],source_files_sha256=before,frame_identity_passed=True,
                independent_projection_max_abs_px=parity,regions=metrics,
                shared_static_mesh_max_delta=float(np.abs(fit['vertices_world']-fit['vertices_world'][0]).max()),
                gate=read(root/'fit/geometry-gate.json'),
                coarse_median_limit_px_range=[float(.03*height.min()),float(.03*height.max())],
                selected_temporal_frames=[rows[i]['frame'] for i in selected],
                source_files_unchanged=all(sha(root/p)==h for p,h in before.items()),
                physical_quality_accepted=False,measurements_are_detector_residuals=True,
                no_camera_or_pose_optimization=True)
    (output/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(label=label,frames=len(rows),wh=rows[0]['wh'],projection_parity=parity,
                         region_medians={k:v['used_observations']['median'] for k,v in metrics.items()})),flush=True)
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--label',required=True)
    args=parser.parse_args()
    analyze(args.run,args.output,args.label)
