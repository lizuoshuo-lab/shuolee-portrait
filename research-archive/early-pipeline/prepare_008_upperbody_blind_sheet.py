"""Prepare native source crops for independent upper-body visibility review."""

import argparse
from hashlib import sha256
import io
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import review_008_occlusion_control as evidence
import review_capture_formal_holdouts as base


RAW_POINTS = (5, 6, 7, 8, 9, 10)
NAMES = ('left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
         'left_wrist', 'right_wrist')
COLORS = ('#00e1ff', '#ffb52b')


def mark(draw, x, y, index):
    x, y = round(x), round(y)
    if index < 2:
        draw.ellipse((x-7, y-7, x+7, y+7), outline='black', width=4)
        draw.ellipse((x-7, y-7, x+7, y+7), outline=COLORS[index % 2], width=2)
    else:
        if index < 4:
            points = [(x-7, y-7), (x+7, y-7), (x+7, y+7), (x-7, y+7), (x-7, y-7)]
        else:
            points = [(x, y-9), (x+9, y), (x, y+9), (x-9, y), (x, y-9)]
        draw.line(points, fill='black', width=4)
        draw.line(points, fill=COLORS[index % 2], width=2)


def crop_region(row, points):
    x0, y0, x1, y1 = row['initial_box_xyxy']
    margin = (y1-y0)*.07
    lo = np.minimum([x0, y0], points.min(0)) - margin
    hi = np.maximum([x1, y0+(y1-y0)*.63], points.max(0)) + margin
    bounds = np.r_[np.maximum(np.floor(lo), 0), np.minimum(np.ceil(hi), row['wh'])].astype(int)
    base.need(np.all(bounds[2:] > bounds[:2]), f'Invalid review crop: {row["frame"]}')
    return bounds.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'project', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    run, project, out = (path.resolve() for path in (args.run, args.project, args.output))
    base.need(not out.is_relative_to(run.parent) and not out.is_relative_to(project)
              and not out.is_relative_to(Path('/mnt/d/00').resolve()) and not out.exists(),
              'Choose a new, separate diagnostic directory outside the record tree.')
    launch, cfg, rows, protected, _, _ = base.cohort(run, project)
    camera = run.parent/'camera.json'
    manifest_path = run/'data/inputs.json'
    obs_path = run/'data/observations.npz'
    decoded_path = run/'decoded.json'
    manifest, decoded = base.read(manifest_path), base.read(decoded_path)
    source_reference = base.read(evidence.EVIDENCE)
    pins = {path: base.sha(path) for path in (camera, manifest_path, decoded_path, run/'config.json',
        run.parent/'preregistration.json', Path(__file__).resolve())}
    pins.update({evidence.EVIDENCE: evidence.EVIDENCE_SHA,
                 evidence.ARCHIVED_INPUTS: source_reference['observation_manifest_sha256'],
                 obs_path: source_reference['observations_sha256']})
    base.check_pins(pins)
    layout = cfg['capture_layout']
    base.need(layout == base.read(camera)['capture_layout'] and len(rows) == launch['expected_native_samples']
              and source_reference['source_video_sha256'] == decoded['video_sha256']
              == manifest['video_sha256'] == launch['source_sha256'], 'Native input domain changed.')
    prior = base.read(evidence.ARCHIVED_INPUTS)['records']
    base.need(len(prior) == len(rows) == len(decoded['records']) and all(
        all(row[k] == old[k] for k in base.SOURCE_FIELDS)
        and all(row[k] == raw[k] for k in ('frame', 'source_sha256', 'source_frame_index', 'time_seconds', 'wh', 'image'))
        for row, old, raw in zip(rows, prior, decoded['records'], strict=True)), 'Native row identity changed.')

    sys.path.insert(0, str(project/'src'))
    from shuolee_portrait import official_smplifyx, stable_selection

    slots = official_smplifyx.COCO[list(RAW_POINTS)]
    base.need(slots.tolist() == [5, 2, 6, 3, 7, 4], 'Bilateral RTMW to SMPLify-X mapping changed.')
    scope = stable_selection.selection_scope(dict(rows=rows, source_manifest=manifest),
        arc=layout['arc'], capture_layout=layout)
    bins = {frame: i for i, group in enumerate(scope['source_view_bins'][:scope['view_targets']['front']])
            for frame in group}
    front = [row for row in rows if row['frame'] in bins and row['frame'] not in protected]
    front.sort(key=lambda row: row['source_frame_index'])
    counts = launch['expected_fit_candidates']
    base.need(len(front) == counts['front'] == 34 and scope['retained_holdouts'] == list(protected)
              and len(scope['fit_frame_ids']) == sum(counts.values()), 'Frozen front roster changed.')
    with np.load(obs_path, allow_pickle=False) as obs:
        ids = obs['frame_ids'].astype(str).tolist()
        points = obs['keypoints'][:, RAW_POINTS]
    base.need(len(set(ids)) == len(ids) == len(rows) and set(ids) == {row['frame'] for row in rows}
              and points.shape == (len(rows), 6, 2) and np.isfinite(points).all(), 'Native observations changed.')

    records, tiles = [], []
    for row in front:
        frame = row['frame']
        native = points[ids.index(frame)]
        path = Path(row['image'])
        raw = path.read_bytes()
        base.need(sha256(raw).hexdigest() == row['source_sha256'], f'Changed native image: {frame}')
        bounds = crop_region(row, native)
        with Image.open(io.BytesIO(raw)) as image:
            base.need(list(image.size) == row['wh'], f'Changed native dimensions: {frame}')
            tile = image.convert('RGB').crop(bounds)
        draw = ImageDraw.Draw(tile)
        for index, (x, y) in enumerate(native):
            mark(draw, x-bounds[0], y-bounds[1], index)
        tiles.append(tile)
        records.append(dict(frame=frame, source_frame_index=row['source_frame_index'],
            time_seconds=row['time_seconds'], source_bin=bins[frame], source_sha256=row['source_sha256'],
            native_crop_xyxy=bounds, detected_xy_native=native.tolist()))

    cols, gutter, label, header = 4, 10, 32, 60
    width = max(tile.width for tile in tiles)+gutter
    height = max(tile.height for tile in tiles)+gutter+label
    sheet = Image.new('RGB', (cols*width+gutter, ((len(tiles)+cols-1)//cols)*height+header), '#171717')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 19)
    draw.text((gutter, 5), '008 / Front source candidates / 34 native crops / chronological source order', font=font, fill='white')
    draw.text((gutter, 29), 'Detected locations only: cyan = subject left, amber = subject right. '
              'Circle = shoulder, square = elbow, diamond = wrist.', font=font, fill='white')
    for i, (tile, record) in enumerate(zip(tiles, records, strict=True)):
        x = i % cols * width + gutter
        y = i // cols * height + header
        draw.text((x, y), f'{record["frame"]}  /  source {record["source_frame_index"]}  /  {record["time_seconds"]:.3f}s',
                  font=font, fill='white')
        sheet.paste(tile, (x, y+label))

    base.check_pins(pins)
    out.mkdir()
    path = out/'source-blind-contact.jpg'
    sheet.save(path, quality=95, subsampling=0, optimize=True)
    with Image.open(path) as saved:
        saved.verify()
    result = dict(state='source_review_sheet_ready', review_completed=False, fit_calls=0, gpu_calls=0,
        source_video_sha256=launch['source_sha256'], input_sha256={str(p): digest for p, digest in pins.items()},
        protected_frames=protected, point_names=list(NAMES), raw_rtmw_slots=list(RAW_POINTS),
        smplifyx_slots=slots.tolist(), native_pixel_convention='integer_centers', columns=cols,
        sheet=path.name, sheet_sha256=base.sha(path), sheet_wh=list(sheet.size), records=records)
    (out/'source-blind-contact.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(state=result['state'], sheet=str(path), sha256=result['sheet_sha256'],
                          wh=result['sheet_wh'], frames=len(records))))


if __name__ == '__main__':
    main()
