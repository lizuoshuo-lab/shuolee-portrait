"""Freeze observed street time windows and their original images, CPU only."""
from pathlib import Path
import hashlib
import itertools
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

LAB = Path(__file__).resolve().parents[2]
SCENE = LAB / 'data/dynamic-human/argoverse2/processed/training/000'
OUT = LAB / 'outputs/dynamic-human-street-temporal-012'
WINDOWS = [('pose-onset', 0, 48, 65), ('approach', 2, 100, 125),
           ('side-occlusion', 4, 124, 141), ('rear-occlusion', 6, 138, 155)]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (OUT / 'protocol.json').exists(), 'Preserve the frozen protocol'
    (OUT / 'ground-truth').mkdir(parents=True, exist_ok=True)
    annotations = json.loads((SCENE / 'instances/instances_info.json').read_text())['79']['frame_annotations']
    stamps = json.loads((SCENE / 'capture_timestamps.json').read_text())['frames']
    signs = np.array(list(itertools.product((-1., 1.), repeat=3)))
    windows, images = [], []
    for name, camera, start, end in WINDOWS:
        entries, boxes, captured_times = [], [], []
        for frame in range(start, end + 1):
            camera_entry = next(x for x in stamps[frame]['cameras'] if int(x['camera_id']) == camera)
            original_path = SCENE / 'images' / f'{frame:03d}_{camera}.jpg'
            original = Image.open(original_path).convert('RGB')
            size = tuple(int(x / 3) for x in original.size)
            target = original.resize(size, Image.Resampling.BILINEAR)
            stem = f'frame{frame:03d}-camera{camera}'
            target.save(OUT / 'ground-truth' / (stem + '.png'))
            existing = LAB / 'outputs/dynamic-human-street-joint-009/resume-10000-expandable/ground-truth' / (stem + '.png')
            if existing.exists():
                assert np.array_equal(np.asarray(target), np.asarray(Image.open(existing)))
            index = annotations['frame_idx'].index(frame)
            camera_to_city = np.loadtxt(SCENE / camera_entry['capture_pose'])
            local = np.linalg.inv(camera_to_city) @ np.asarray(annotations['obj_to_world'][index])
            corners = signs * np.asarray(annotations['box_size'][index])[None] / 2
            points = corners @ local[:3, :3].T + local[:3, 3]
            fx, fy, cx, cy = np.loadtxt(SCENE / 'intrinsics' / f'{camera}.txt')[:4]
            scale = np.array(size) / np.array(original.size)
            box = None
            if points[:, 2].min() > .1:
                uv = (points[:, :2] / points[:, 2:] * [fx, fy] + [cx, cy]) * scale
                lo = np.maximum(np.floor(uv.min(0)), 0).astype(int)
                hi = np.minimum(np.ceil(uv.max(0)), size).astype(int)
                if np.all(hi > lo):
                    box = [*lo.tolist(), *hi.tolist()]
                    boxes.append(box)
            row = dict(frame=frame, camera=camera, image_size=list(size), original_sha256=sha(original_path),
                resized_sha256=sha(OUT / 'ground-truth' / (stem + '.png')),
                timestamp_ns=int(camera_entry['timestamp_ns']), heldout=frame % 10 == 0,
                projected_person_box=box)
            images.append(row)
            entries.append(dict(frame=frame, camera=camera))
            captured_times.append(row['timestamp_ns'])
        bounds = np.array(boxes)
        assert len(bounds)
        union = [int(bounds[:, 0].min()), int(bounds[:, 1].min()), int(bounds[:, 2].max()), int(bounds[:, 3].max())]
        intervals = np.diff(np.array(captured_times, dtype=np.int64)) * 1e-9
        assert np.all(intervals > 0)
        windows.append(dict(name=name, camera=camera, start=start, end=end, images=entries,
            fixed_person_box_union=union, median_capture_interval_seconds=float(np.median(intervals)),
            interval_range_seconds=[float(intervals.min()), float(intervals.max())],
            purpose='Missing-pose fallback to articulated handoff' if name == 'pose-onset'
                    else 'Observed clothing, gait, occlusion and camera-boundary behavior'))
        # Real first/middle/last observations; no drawn scene or enhanced content.
        font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 20)
        panels = [Image.open(OUT / 'ground-truth' / f'frame{f:03d}-camera{camera}.png').crop(union)
                  for f in (start, (start + end) // 2, end)]
        scale = min(2., 310 / panels[0].width, 440 / panels[0].height)
        display_size = tuple(round(x * scale) for x in panels[0].size)
        canvas = Image.new('RGB', (960, display_size[1] + 55), 'white')
        draw = ImageDraw.Draw(canvas)
        for k, (f, panel) in enumerate(zip((start, (start + end) // 2, end), panels)):
            draw.text((320 * k + 12, 8), f'Frame {f} / Camera {camera}', font=font, fill='#182d39')
            canvas.paste(panel.resize(display_size, Image.Resampling.NEAREST), (320 * k + (320 - display_size[0]) // 2, 42))
        canvas.save(OUT / (name + '-observations.png'))
    assert len(images) == len({(x['frame'], x['camera']) for x in images}) == 80
    protocol = dict(status='time_windows_and_original_targets_frozen_before_temporal_model_rendering',
        main_person=79, scene='00a6ffc1-6ce9-3bc3-a060-6006e9893a1a', windows=windows, images=images,
        evaluation_updates=30000, views_per_model=80,
        validation='Re-render overlapping saved heldout views and require pixel agreement before accepting a reloaded model.',
        measurements=['Per-frame image and fixed-union-region PSNR/MAE.',
            'Temporal difference residual: mean absolute difference between reconstructed and observed consecutive image differences, in a fixed pixel region.',
            'Original/reconstruction videos with explicit train/heldout labels and the recorded capture intervals.'],
        boundaries=['Contains training and heldout times; supplementary continuity diagnostic, not80unseen frames.',
            'Fixed union region contains background and occluders; no instance-mask or silhouette GT.',
            'Temporal difference residual includes camera motion, object motion and photometric error; it is not isolated flicker or flow-compensated accuracy.',
            'A blurry image may score deceptively on temporal changes; actual clothing/limb visuals remain required.',
            'Handoff atframe56 is far from the camera; cannot certify near-view clothes from that window.',
            'Known camera and trajectory priors; no forecasting.'])
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    print(json.dumps(dict(status=protocol['status'], views=80, windows=windows), indent=2))


if __name__ == '__main__':
    main()
