"""Read frozen room associations and inspect inputs; no training or causal claims."""
from __future__ import annotations
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import uuid

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

LAB = Path(__file__).resolve().parents[1]
RUN_NAME = 'freiburg1_room_full_320x240_rgbd_s0_20261010T125931Z_0d3dc918'
RUN = LAB / 'outputs/splatam-official-runs-001' / RUN_NAME
TRAJ = LAB / 'outputs/splatam-repeat-evaluations-001' / RUN_NAME / 'trajectory'
DATA = LAB / 'data/tum/rgbd_dataset_freiburg1_room'
EVENTS = ((836, 842), (1131, 1135), (1158, 1160))
NATIVE_LOADER = Path(r'\\wsl.localhost\Ubuntu-20.04\opt\gs-slam-lab\splatam\repos\SplaTAM\datasets\gradslam_datasets\basedataset.py')

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()

def rows(path):
    with path.open(encoding='utf8', newline='') as f:
        return list(csv.DictReader(f))

def quantile(values):
    x = np.asarray(values, dtype=float)
    assert np.isfinite(x).all()
    return dict(zip(('min', 'p05', 'median', 'p95', 'max'), map(float, np.quantile(x, (0, .05, .5, .95, 1)))))

def main():
    out = LAB / 'outputs/splatam-room-input-diagnostic-001' / (dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    bindings = {str(p.relative_to(LAB)) if p.is_relative_to(LAB) else str(p): sha(p) for p in (
        Path(__file__), TRAJ/'associations.csv', TRAJ/'errors.csv', TRAJ/'reference-independent.tum',
        RUN/'execution.json', RUN/'effective-config.json', DATA/'rgb.txt', DATA/'depth.txt', DATA/'groundtruth.txt', NATIVE_LOADER)}
    associations, errors = rows(TRAJ/'associations.csv'), rows(TRAJ/'errors.csv')
    assert len(associations) == len(errors) == 1315
    raw_gt = np.loadtxt(DATA/'groundtruth.txt')
    gt = raw_gt[[int(r['gt_index']) for r in associations]]
    assert np.allclose(gt[:, 0], [float(r['gt_timestamp']) for r in associations], rtol=0, atol=1e-7)
    gt_steps = np.r_[0., np.linalg.norm(np.diff(gt[:, 1:4], axis=0), axis=1)]
    rotations = Rotation.from_quat(gt[:, 4:8])
    gt_rot_steps = np.r_[0., np.rad2deg((rotations[:-1].inv() * rotations[1:]).magnitude())]
    times = np.asarray([float(r['rgb_timestamp']) for r in associations])
    elapsed = times - times[0]
    records = []
    image_bindings = hashlib.sha256()
    saved_images = {}
    sample_sets = ((830, 834, 836, 838, 840, 842, 846), (1126, 1130, 1131, 1133, 1134, 1135, 1140))
    wanted = {i for sample in sample_sets for i in sample}
    previous_gray = None
    for i, (a, e) in enumerate(zip(associations, errors)):
        assert int(a['retained_index']) == int(e['frame']) == i
        assert abs(float(e['timestamp']) - times[i]) < 1e-7
        paths = [DATA/a['rgb_path'], DATA/a['depth_path']]
        for p in paths:
            assert p.resolve().is_relative_to(DATA.resolve())
            image_bindings.update((str(p.relative_to(DATA))+'\0'+sha(p)+'\n').encode())
        bgr = cv2.imread(str(paths[0]), cv2.IMREAD_COLOR)
        depth_raw = cv2.imread(str(paths[1]), cv2.IMREAD_UNCHANGED)
        assert bgr.shape == (480,640,3) and depth_raw.shape == (480,640) and depth_raw.dtype == np.uint16
        bgr = cv2.resize(bgr, (320,240), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth_raw.astype(float), (320,240), interpolation=cv2.INTER_NEAREST) / 5000.
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(float)/255.
        gx, gy = cv2.Sobel(gray, cv2.CV_64F,1,0,ksize=3), cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
        valid = depth > 0
        assert valid.any() and np.isfinite(depth).all()
        record = {'frame':i, 'rgb_timestamp':times[i], 'elapsed_seconds':elapsed[i],
            'rgb_path':a['rgb_path'], 'depth_path':a['depth_path'],
            'depth_minus_rgb_ms':float(a['depth_minus_rgb_seconds'])*1000,
            'gt_minus_rgb_ms':float(a['gt_minus_rgb_seconds'])*1000,
            'rgb_dt_seconds':float(times[i]-times[i-1]) if i else 0.,
            'depth_valid_fraction':float(valid.mean()),
            'depth_valid_median_m':float(np.median(depth[valid])),
            'depth_valid_p95_m':float(np.quantile(depth[valid], .95)),
            'gray_mean_0_1':float(gray.mean()), 'gray_std_0_1':float(gray.std()),
            'sobel3_mean_gradient_0_1':float(np.hypot(gx,gy).mean()),
            'laplacian3_variance_0_1':float(cv2.Laplacian(gray,cv2.CV_64F,ksize=3).var()),
            'unwarped_adjacent_gray_mae_0_1':float(np.abs(gray-previous_gray).mean()) if i else 0.,
            'gt_adjacent_translation_m':float(gt_steps[i]), 'gt_adjacent_rotation_deg':float(gt_rot_steps[i]),
            'first_camera_gauge_error_m':float(e['ate_first_camera_gauge_m']), 'se3_error_m':float(e['ate_se3_m']),
            'adjacent_rpe_translation_m':float(e['rpe_1frame_trans_m']) if i else 0.,
            'adjacent_rpe_rotation_deg':float(e['rpe_1frame_rot_deg']) if i else 0.}
        records.append(record)
        previous_gray = gray
        if i in wanted:
            saved_images[i] = (cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB), depth)
        if (i+1)%300 == 0:
            print(json.dumps({'processed':i+1,'total':1315}), flush=True)
    table = out/'per-frame-inputs.csv'
    with table.open('w', encoding='utf8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    fields = ('depth_valid_fraction','depth_valid_median_m','sobel3_mean_gradient_0_1','laplacian3_variance_0_1',
        'unwarped_adjacent_gray_mae_0_1','depth_minus_rgb_ms','gt_minus_rgb_ms','gt_adjacent_translation_m','gt_adjacent_rotation_deg')
    distributions = {key:quantile([r[key] for r in records[1:]]) for key in fields}
    event_summaries = []
    for first,last in EVENTS:
        context = records[max(0,first-5):min(len(records),last+6)]
        event_summaries.append({'first_frame':first,'last_frame':last,
            'elapsed_start_s':elapsed[first], 'elapsed_end_s':elapsed[last],
            'core_frames':records[first:last+1],
            'context_first_frame':context[0]['frame'],'context_last_frame':context[-1]['frame'],
            'context_statistics':{key:quantile([r[key] for r in context]) for key in fields}})
    plt.rcParams.update({'font.size':9, 'axes.spines.top':False,'axes.spines.right':False})
    fig,axs = plt.subplots(3,2,figsize=(13,10),constrained_layout=True)
    panels = (
      (('first_camera_gauge_error_m','se3_error_m'), 'Position error (m)', ('First-camera gauge','Whole-run SE3 alignment')),
      (('adjacent_rpe_translation_m','gt_adjacent_translation_m'), 'Adjacent translation (m)', ('RPE error','Raw matched GT motion')),
      (('depth_valid_fraction',), 'Sensor nonzero depth fraction', ('320x240 nearest resized depth',)),
      (('sobel3_mean_gradient_0_1',), 'Mean Sobel3 gradient, gray in [0,1]', ('Input texture proxy, not observability rank',)),
      (('depth_minus_rgb_ms','gt_minus_rgb_ms'), 'Association signed delta (ms)', ('Depth minus RGB','GT minus RGB')),
      (('adjacent_rpe_rotation_deg','gt_adjacent_rotation_deg'), 'Adjacent rotation (deg)', ('RPE error','Raw matched GT motion')))
    for ax,(keys,label,names) in zip(axs.flat,panels):
        for key,name in zip(keys,names): ax.plot(elapsed,[r[key] for r in records],label=name,lw=.8)
        for first,last in EVENTS: ax.axvspan(elapsed[first],elapsed[last],color='#da7540',alpha=.18)
        ax.set(xlabel='Selected RGB elapsed time (s)',ylabel=label); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Frozen full room run: input observations around retrospectively selected error events\nInput texture / sensor validity are proxies; historical online map coverage is not measured here')
    fig.savefig(out/'input-and-error-timeline.png',dpi=150); plt.close(fig)
    for n,indices in enumerate(sample_sets,1):
        fig,axes = plt.subplots(2,len(indices),figsize=(18,5.6),constrained_layout=True)
        for j,i in enumerate(indices):
            rgb,depth = saved_images[i]; r=records[i]
            axes[0,j].imshow(rgb); axes[0,j].set_title(f"f{i}, {elapsed[i]:.3f}s\nRPE {r['adjacent_rpe_translation_m']:.3f}m / {r['adjacent_rpe_rotation_deg']:.2f}deg")
            axes[1,j].imshow(np.ma.masked_equal(depth,0),cmap='viridis',vmin=0,vmax=6)
            axes[1,j].set_title(f"Valid depth {r['depth_valid_fraction']:.1%}\ndepth-RGB {r['depth_minus_rgb_ms']:+.2f}ms")
            axes[0,j].axis('off'); axes[1,j].axis('off')
        fig.suptitle(f'Actual RGB and depth inputs: event window {n}; RGB linear, depth nearest 320x240\nDepth color uses fixed 0..6m; zero depth masked; this is input evidence, not an online tracking mask')
        fig.savefig(out/f'event-{n}-rgb-depth.png',dpi=150); plt.close(fig)
    for p,expected in bindings.items():
        assert sha(LAB/p if not p.startswith('\\\\') else Path(p)) == expected, p
    outputs = {p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    result = {'status':'completed_read_only_input_diagnostic','created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
      'run_name':RUN_NAME,'frames':1315,'local_only':True,'source_bindings_sha256':bindings,
      'ordered_rgb_depth_file_digest_sha256':image_bindings.hexdigest(),'outputs_sha256':outputs,
      'preprocessing':{'RGB':'OpenCV INTER_LINEAR 640x480 to 320x240, gray BGR2GRAY /255',
          'depth':'OpenCV INTER_NEAREST to 320x240, uint16 /5000m, no crop; validity=depth>0',
          'GT_motion':'Consecutive selected nearest-GT absolute body poses, translation norm and SO3 relative angle',
          'statistics':'All retained noninitial frames, unwarped image changes confounded by camera motion'},
      'selection_policy':'Events and display frames chosen retrospectively from prior error diagnosis, not preregistered causal tests',
      'full_sequence_input_statistics':distributions,'events':event_summaries,
      'limits':['Texture and Laplacian are scene-dependent proxies, not a physical blur measurement or pose observability proof',
        'Sensor valid-depth fraction differs from rendered map silhouette and the online tracking loss mask',
        'Nearest timestamp deltas are association observations, not measured hardware synchronization error',
        'No causal intervention, source/model/trajectory writes, GT parameter fitting, or new algorithm accuracy claim']}
    (out/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps({'status':result['status'],'output':str(out),'frames':1315,'source_bindings_unchanged':True}),flush=True)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
