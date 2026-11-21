"""Audit frozen street sequences and compose unenhanced, time-ordered comparisons.

CPU only. PNGs are the metric source; MP4s are explicitly lossy viewing copies.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import shutil
import statistics
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from audit_street_joint_images import check_quantization

LAB = Path(__file__).resolve().parents[2]
OUT = LAB / 'outputs/dynamic-human-street-temporal-012'
SCENE = LAB / 'data/dynamic-human/argoverse2/processed/training/000'
AUDIT = OUT / 'independent-audit'
BRANCHES = ('deformable', 'articulated')
TITLES = ('原始观测', '可变形模型', '人体关节模型')
FOOTNOTE = '均为 30,000 次更新；同倍率，未锐化；10 Hz 播放；评测使用原 PNG'
PANEL_MAX_WIDTH = 630


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_image(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB'))


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def stem(item):
    return f"frame{item['frame']:03d}-camera{item['camera']}"


def validate_sources(protocol):
    lookup = {(r['frame'], r['camera']): r for r in protocol['images']}
    assert len(lookup) == len(protocol['images']) == 80
    checked = []
    assigned = []
    for window in protocol['windows']:
        items = window['images']
        assert [r['frame'] for r in items] == list(range(window['start'], window['end'] + 1))
        assert all(r['camera'] == window['camera'] for r in items)
        x0, y0, x1, y1 = window['fixed_person_box_union']
        timestamps = []
        for item in items:
            key = item['frame'], item['camera']
            assigned.append(key)
            row = lookup[key]
            assert row['heldout'] == (item['frame'] % 10 == 0)
            original_path = SCENE / 'images' / f"{item['frame']:03d}_{item['camera']}.jpg"
            png = OUT / 'ground-truth' / (stem(item) + '.png')
            assert sha(original_path) == row['original_sha256']
            assert sha(png) == row['resized_sha256']
            with Image.open(original_path) as original:
                expected_size = tuple(int(x / 3) for x in original.size)
                expected = np.asarray(original.convert('RGB').resize(expected_size, Image.Resampling.BILINEAR))
            gt = read_image(png)
            assert list(expected_size) == row['image_size']
            assert np.array_equal(expected, gt), key
            assert 0 <= x0 < x1 <= expected_size[0] and 0 <= y0 < y1 <= expected_size[1]
            timestamps.append(row['timestamp_ns'])
            checked.append(dict(**item, source_sha256=sha(original_path), target_sha256=sha(png)))
        intervals = np.diff(np.array(timestamps, dtype=np.int64)) / 1e9
        assert np.all(intervals > 0)
        assert math.isclose(float(np.median(intervals)), window['median_capture_interval_seconds'], abs_tol=1e-9)
        # Both videos run at the acquisition cadence, not a favorable slowed sequence.
        assert float(np.max(np.abs(intervals - .1))) < .002
    assert len(assigned) == len(set(assigned)) == 80 and set(assigned) == set(lookup)
    return dict(status='all80_original_hashes_resizing_windows_and_capture_intervals_verified',
                protocol_sha256=sha(OUT / 'protocol.json'), images=checked)


def compose(window, item, arrays, view):
    if view == 'fixed-union':
        x0, y0, x1, y1 = window['fixed_person_box_union']
        arrays = [a[y0:y1, x0:x1] for a in arrays]
    size = arrays[0].shape[1], arrays[0].shape[0]
    assert all(a.shape == arrays[0].shape for a in arrays)
    scale = min(3., PANEL_MAX_WIDTH / size[0], 680 / size[1])
    display = max(1, round(size[0] * scale)), max(1, round(size[1] * scale))
    width = max(240, display[0] + 20)
    canvas_size = width * len(arrays), display[1] + 105
    # H.264 yuv420p requires even dimensions. Pad rather than distort the images.
    canvas_size = tuple(v + v % 2 for v in canvas_size)
    canvas = Image.new('RGB', canvas_size, 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 19)
    small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 17)
    assert len(TITLES) == len(arrays)
    for col, (title, pixels) in enumerate(zip(TITLES, arrays)):
        draw.text((col * width + 10, 8), title, font=font, fill='#182d39')
        panel = Image.fromarray(pixels).resize(display, Image.Resampling.NEAREST)
        canvas.paste(panel, (col * width + (width - display[0]) // 2, 39))
    heldout = '留出时刻' if item['heldout'] else '训练时刻'
    mode = '固定区域；包含背景与遮挡者' if view == 'fixed-union' else '全图'
    draw.text((10, display[1] + 47), f"帧 {item['frame']} / 相机 {item['camera']} · {heldout} · {mode}", font=small, fill='#182d39')
    draw.text((10, display[1] + 76), FOOTNOTE, font=small, fill='#182d39')
    return canvas


def encode_video(frames, directory, output):
    ffmpeg = shutil.which('ffmpeg')
    assert ffmpeg, 'ffmpeg is required for viewing copies'
    width, height = frames[0].size
    assert all(f.size == (width, height) for f in frames)
    command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-n', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', f'{width}x{height}', '-r', '10', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '16',
               '-preset', 'medium', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)]
    with (directory / (output.stem + '-encoder.log')).open('wb') as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log, stdout=subprocess.DEVNULL)
        try:
            for frame in frames:
                process.stdin.write(frame.tobytes())
            process.stdin.close()
            code = process.wait(timeout=180)
            assert code == 0, ('ffmpeg failed', code, output)
        except BaseException:
            process.kill()
            process.wait()
            raise
    ffprobe = str(Path(ffmpeg).with_name('ffprobe.exe'))
    information = json.loads(subprocess.check_output([ffprobe, '-v', 'error', '-count_frames', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate,nb_read_frames,duration', '-of', 'json', str(output)], text=True))
    stream = information['streams'][0]
    assert int(stream['nb_read_frames']) == len(frames) and stream['r_frame_rate'] == '10/1'
    assert (stream['width'], stream['height']) == (width, height)
    assert math.isclose(float(stream['duration']), len(frames) / 10, abs_tol=.001)
    # Decode every frame so a valid header alone cannot pass this check.
    subprocess.run([ffmpeg, '-v', 'error', '-i', str(output), '-f', 'null', '-'], check=True, timeout=180,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return dict(file=output.name, sha256=sha(output), stream=stream, all_frames_decode=True)


def main():
    global AUDIT
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources-only', action='store_true')
    parser.add_argument('--output-name', default='independent-audit')
    args = parser.parse_args()
    assert args.output_name.startswith('independent-audit') and all(c.isalnum() or c in '-_' for c in args.output_name)
    AUDIT = OUT / args.output_name
    protocol = json.loads((OUT / 'protocol.json').read_text())
    sources = validate_sources(protocol)
    if args.sources_only:
        path = OUT / 'source-independent-audit.json'
        if path.exists():
            assert json.loads(path.read_text()) == sources
        else:
            save_json(path, sources)
        print(json.dumps(dict(status=sources['status'], images=len(sources['images']))))
        return
    required = [OUT / branch / 'completion.json' for branch in BRANCHES]
    if not all(p.exists() for p in required):
        print(json.dumps(dict(status='not_ready_no_rendered_comparison_created', missing=[str(p) for p in required if not p.exists()])))
        return
    assert not (AUDIT / 'independent-audit.json').exists(), 'Preserve completed independent audits'
    AUDIT.mkdir(exist_ok=True)
    reports = {b: json.loads((OUT / b / 'completion.json').read_text()) for b in BRANCHES}
    item_lookup = {(r['frame'], r['camera']): r for r in protocol['images']}
    verified_images, verified_transitions, summaries = [], [], {}
    for branch, report in reports.items():
        assert report['optimized_steps'] == 0
        saved_rows = {(r['frame'], r['camera']): r for r in report['rows']}
        assert len(saved_rows) == len(report['rows']) == 80 and saved_rows.keys() == item_lookup.keys()
        saved_changes = {(r['window'], r['previous_frame'], r['frame']): r for r in report['temporal_differences']}
        assert len(saved_changes) == len(report['temporal_differences']) == 76
        branch_summary = []
        used_changes = set()
        for window in protocol['windows']:
            previous = None
            region_scores, changes, heldout_scores = [], [], []
            x0, y0, x1, y1 = window['fixed_person_box_union']
            for entry in window['images']:
                item = item_lookup[(entry['frame'], entry['camera'])]
                target = read_image(OUT / 'ground-truth' / (stem(item) + '.png'))
                path = OUT / branch / (stem(item) + '-prediction.png')
                pred = read_image(path)
                assert pred.shape == target.shape
                row = saved_rows[(item['frame'], item['camera'])]
                assert row['window'] == window['name'] and row['heldout'] == item['heldout']
                full_score, full_difference = check_quantization(target, pred, row['full_image']['psnr_db'])
                region_score, region_difference = check_quantization(target[y0:y1, x0:x1], pred[y0:y1, x0:x1], row['fixed_union_region']['psnr_db'])
                region_scores.append(region_score)
                if item['heldout']:
                    heldout_scores.append(region_score)
                verified_images.append(dict(branch=branch, **entry, heldout=item['heldout'], window=window['name'],
                    prediction_sha256=sha(path), png_full_psnr_db=full_score, png_fixed_union_psnr_db=region_score,
                    float_png_full_difference_db=full_difference, float_png_union_difference_db=region_difference))
                if previous is not None:
                    previous_item, previous_target, previous_pred = previous
                    key = window['name'], previous_item['frame'], item['frame']
                    used_changes.add(key)
                    assert previous_item['frame'] + 1 == item['frame']
                    actual = np.abs((pred.astype(np.float64) - previous_pred.astype(np.float64)) / 255 -
                                    (target.astype(np.float64) - previous_target.astype(np.float64)) / 255)
                    scores = float(actual.mean()), float(actual[y0:y1, x0:x1].mean())
                    saved = saved_changes[key]
                    assert saved['camera'] == window['camera']
                    assert saved['heldout_transition'] == (item['heldout'] or previous_item['heldout'])
                    # Two independently rounded predictions contribute at most 1/255.
                    for value, field in zip(scores, ('full_image_temporal_difference_residual', 'fixed_union_temporal_difference_residual')):
                        assert abs(value - saved[field]) <= 1 / 255 + 3e-7, (branch, key, value, saved[field])
                    changes.append(scores[1])
                    verified_transitions.append(dict(branch=branch, window=window['name'], previous_frame=previous_item['frame'],
                        frame=item['frame'], camera=item['camera'], png_full_residual=scores[0], png_fixed_union_residual=scores[1]))
                previous = item, target, pred
            branch_summary.append(dict(window=window['name'], frames=len(region_scores),
                fixed_union_png_mean_psnr_db=statistics.mean(region_scores), heldout_frames=len(heldout_scores),
                heldout_fixed_union_png_mean_psnr_db=statistics.mean(heldout_scores),
                consecutive_transitions=len(changes), png_fixed_union_mean_temporal_difference_residual=statistics.mean(changes)))
        assert used_changes == saved_changes.keys()
        summaries[branch] = branch_summary
    assert len(verified_images) == 80 * len(BRANCHES) and len(verified_transitions) == 76 * len(BRANCHES)
    videos, sheets = [], []
    for window in protocol['windows']:
        for view in ('full-image', 'fixed-union'):
            frames = []
            for entry in window['images']:
                item = item_lookup[(entry['frame'], entry['camera'])]
                arrays = [read_image(OUT / 'ground-truth' / (stem(item) + '.png'))] + [
                    read_image(OUT / b / (stem(item) + '-prediction.png')) for b in BRANCHES]
                frames.append(compose(window, item, arrays, view))
            # Sampled viewing sheets keep exact rendered pixels; every frame remains in the video.
            indices = sorted({0, len(frames) // 3, 2 * len(frames) // 3, len(frames) - 1})
            sheet = Image.new('RGB', (frames[0].width, frames[0].height * len(indices)), 'white')
            for j, index in enumerate(indices):
                sheet.paste(frames[index], (0, j * frames[index].height))
            name = f"{window['name']}-{view}"
            sheet_path = AUDIT / (name + '-contact.png')
            sheet.save(sheet_path)
            sheets.append(dict(file=sheet_path.name, sha256=sha(sheet_path), window=window['name'], view=view,
                frames=[window['images'][i]['frame'] for i in indices]))
            videos.append(dict(window=window['name'], view=view, **encode_video(frames, AUDIT, AUDIT / (name + '.mp4'))))
    save_json(AUDIT / 'independent-audit.json', dict(status=f'all{len(verified_images)}_images_and{len(verified_transitions)}_transitions_audited_videos_decoded_visual_review_pending',
        source_protocol_sha256=sha(OUT / 'protocol.json'), script_sha256=sha(Path(__file__)),
        sources=sources, source_receipts={b:sha(OUT / b / 'completion.json') for b in BRANCHES},
        images=verified_images, transitions=verified_transitions, window_summaries=summaries,
        videos=videos, contact_sheets=sheets, quality_accepted=False,
        boundaries=protocol['boundaries'] + ['MP4 is a lossy viewing copy; metrics computed from untouched source PNGs.',
            'Images and differences are independently recomputed; SSIM is not independently reimplemented here.',
            'Window means weight frames equally; not pooled error or foreground-only metrics.']))
    print(json.dumps(dict(status='temporal_independent_audit_complete_visual_review_pending', images=len(verified_images), transitions=len(verified_transitions), videos=len(videos), summaries=summaries), indent=2))


if __name__ == '__main__':
    main()
