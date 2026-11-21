"""Read-only saved-mesh comparison in bound native image coordinates.

No torch, model construction, detector, fitter, GPU or Gaussian renderer is
imported. Only this standalone script and its output files are written.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import ast
import hashlib
import json
import os

import cv2
import numpy as np
from PIL import Image

DEFAULT_FRAMES = ('frame_000001', 'frame_000016', 'frame_000032', 'frame_000048',
                  'frame_000063', 'frame_000079', 'frame_000094')
REVIEW_FRAMES = ('frame_000001', 'frame_000032', 'frame_000063')
H = Path(__file__).resolve().parent


def relocate(value):
    value = str(value).replace('\\', '/')
    if os.name == 'nt':
        if value.startswith('/opt/'):
            value = '//wsl.localhost/Ubuntu-20.04' + value
        elif value.startswith('/mnt/c/'):
            value = 'C:/' + value[7:]
        elif value.startswith('/mnt/d/'):
            value = 'D:/' + value[7:]
    else:
        if len(value) > 2 and value[1:3] == ':/':
            value = '/mnt/' + value[0].lower() + '/' + value[3:]
    return Path(value)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bound_file(item):
    path = relocate(item['path'])
    if not path.is_file() or sha(path) != item['sha256']:
        raise ValueError('Bound source changed: ' + str(path))
    return path


def mesh_renderer():
    source = H / 'review_smplify_corrected123.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'mesh')
    scope = dict(np=np, cv2=cv2, Image=Image)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), scope)
    return scope['mesh'], source


def load_snapshot(path, manifest, rows):
    with np.load(path, allow_pickle=False) as source:
        data = {key: source[key].copy() for key in source.files}
    ids = tuple(map(str, data['frame_ids'].tolist()))
    if len(set(ids)) != len(ids) or set(ids) != set(rows):
        raise ValueError('Snapshot is not the complete unique bound frame roster.')
    if str(data['reference_frame_id'].item()) != manifest['reference_frame_id']:
        raise ValueError('Snapshot reference frame differs from bound reference.')
    contract = json.loads(str(data['camera_length_contract_json'].item()))
    if contract != manifest['camera_length_contract']:
        raise ValueError('Snapshot camera length provenance mismatch.')
    slot = {name: i for i, name in enumerate(ids)}
    parity = 0.
    for name, i in slot.items():
        row = rows[name]
        if (int(data['source_frames'][i]) != row['source_frame_index'] or
                str(data['image_sha256'][i]) != row['source_sha256'] or
                str(data['mask_sha256'][i]) != row['mask_sha256']):
            raise ValueError('Snapshot frame/source pixels mismatch: ' + name)
        if not np.array_equal(data['K_integer_centers'][i], row['K_integer_centers']):
            raise ValueError('Snapshot full K changed: ' + name)
        if not np.array_equal(data['source_w2c'][i], row['w2c']):
            raise ValueError('Snapshot source E changed: ' + name)
        expected_E = np.array(row['w2c'], dtype=np.float64)
        expected_E[:3, 3] *= contract['meters_per_camera_unit']
        if not np.array_equal(data['w2c'][i], expected_E):
            raise ValueError('Snapshot normalized E differs from declared single conversion.')
        joints = data['joints_camera'][i]
        if not np.isfinite(joints).all() or (joints[:, 2] <= .05).any():
            raise ValueError('Invalid camera-domain joints in saved snapshot.')
        q = joints @ data['K_integer_centers'][i].T
        parity = max(parity, float(np.max(np.abs(q[:, :2] / q[:, 2:] - data['projected'][i]))))
    if parity > 1e-8:
        raise ValueError('Saved projection disagrees with complete-K camera joints.')
    vertices, faces = data['vertices_camera'], data['faces']
    if (vertices.ndim != 3 or vertices.shape[0] != len(ids) or vertices.shape[-1] != 3 or
            not np.isfinite(vertices).all() or faces.ndim != 2 or faces.shape[1] != 3 or
            faces.dtype.kind not in 'iu' or faces.min() < 0 or faces.max() >= vertices.shape[1]):
        raise ValueError('Invalid saved camera mesh.')
    return data, slot, parity


def crop_box(values, wh, padding=0):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError('Invalid fixed source crop.')
    b = np.r_[np.floor(values[:2] - padding), np.ceil(values[2:] + padding)].astype(int)
    b[:2] = np.maximum(b[:2], 0)
    b[2:] = np.minimum(b[2:], wh)
    if np.any(b[2:] <= b[:2]):
        raise ValueError('Empty fixed source crop.')
    return b.tolist()


def joined(panels):
    if len({im.size for im in panels}) != 1:
        raise ValueError('Different panel dimensions; stretching is not permitted.')
    result = Image.new('RGB', (sum(im.width for im in panels), panels[0].height))
    for i, image in enumerate(panels):
        result.paste(image, (i * image.width, 0))
    return result


def main(args):
    inputs = relocate(args.inputs)
    initial_path = relocate(args.initial)
    output = relocate(args.output)
    receipt = relocate(args.receipt)
    initial_only = bool(args.initial_only)
    if initial_only == bool(args.final):
        raise ValueError('Choose either --initial-only or --final, never both/neither.')
    # The initial smoke check stays outside the WPS record tree. It must not
    # look like a final fit. Only the parent's later final call writes 005.
    record_root = relocate('D:/00/记录')
    if initial_only and record_root.resolve() in (output.resolve(), *output.resolve().parents):
        raise ValueError('Initial-only check must stay in the experiment directory.')
    if receipt.resolve() == output.resolve() or output.resolve() in receipt.resolve().parents:
        raise ValueError('Write the JSON receipt outside the images-only output tree.')
    if receipt.exists():
        raise FileExistsError('Review existing receipt; do not regenerate/overwrite silently.')
    rules = record_root / '规则.md'
    if not rules.is_file() or '只要图片不要文本' not in rules.read_text(encoding='utf-8-sig'):
        raise ValueError('Expected current records/images-only rule.')
    manifest = read(inputs)
    frames_path = bound_file(manifest['frames'])
    rows_list = read(frames_path)['records']
    rows = {row['frame']: row for row in rows_list}
    if len(rows) != len(rows_list) or len(rows) != 94:
        raise ValueError('Expected exactly the bound94 unique frame records.')
    selected = tuple(args.frames or (REVIEW_FRAMES if initial_only else DEFAULT_FRAMES))
    if len(set(selected)) != len(selected) or any(name not in rows for name in selected):
        raise ValueError('Choose unique explicit bound frame IDs.')
    initial, initial_slot, initial_parity = load_snapshot(initial_path, manifest, rows)
    final_path = relocate(args.final) if args.final else None
    final_result_path = None
    if final_path:
        final_result_path = final_path.parent / 'result.json'
        final_result = read(final_result_path)
        if (final_result.get('frames') != 94 or final_result.get('all_named_frames_exported') is not True
                or final_result.get('output_fit_sha256') != sha(final_path)):
            raise ValueError('Final saved NPZ requires its completed export receipt and matching SHA; do not read an active write.')
        final, final_slot, final_parity = load_snapshot(final_path, manifest, rows)
        if not np.array_equal(initial['faces'], final['faces']):
            raise ValueError('Initial/final mesh topology differs.')
        for name in rows:
            for key in ('targets', 'confidence', 'fitting_confidence'):
                if not np.array_equal(initial[key][initial_slot[name]], final[key][final_slot[name]]):
                    raise ValueError('Initial/final observations changed: ' + key)
    else:
        final, final_slot, final_parity = None, None, None
    # Existing face crop source is identified through source-video frame ID,
    # not an assumption that array sorting or snapshot row order agrees.
    crop_source = relocate(args.crops)
    face_crops = {int(r['source_frame']): r['regions']['face']['crop_integer'] for r in read(crop_source)}
    mesh, renderer_source = mesh_renderer()
    snapshots = [('initial', initial, initial_slot)] + ([('final', final, final_slot)] if final else [])
    hashes = {str(p): sha(p) for p in [inputs, frames_path, initial_path, crop_source, renderer_source, rules] +
              ([final_path, final_result_path] if final_path else [])}
    files = []
    for name in selected:
        row = rows[name]
        image_path, mask_path = relocate(row['image']), relocate(row['mask'])
        if sha(image_path) != row['source_sha256'] or sha(mask_path) != row['mask_sha256']:
            raise ValueError('Selected source image/mask version changed: ' + name)
        with Image.open(image_path) as im:
            im.load()
            raw = im.convert('RGB')
        if raw.size != tuple(row['wh']):
            raise ValueError('Native image dimensions changed.')
        overlays = []
        for _, data, lookup in snapshots:
            i = lookup[name]
            data_frame = dict(vertices_camera=data['vertices_camera'][i:i+1], faces=data['faces'],
                              K_integer_centers=data['K_integer_centers'][i])
            overlays.append(mesh(raw, np.array([0, 0, raw.width, raw.height]), data_frame))
        boxes = [
            ('smpl-x/完整视野', '完整视野', [0, 0, raw.width, raw.height], 'entire native image'),
            ('smpl-x/全身', '全身', crop_box(row['initial_box_xyxy'], raw.size, padding=64),
             'original frozen initialization box plus64 native pixels; no final-mesh-dependent crop')]
        if name in REVIEW_FRAMES:
            source_frame = int(row['source_frame_index'])
            if source_frame not in face_crops:
                raise ValueError('Missing existing face crop for reviewed source frame.')
            boxes.append(('分析/面部', '面部', crop_box(face_crops[source_frame], raw.size),
                          'exact original three-frame native face crop; detector region is not physical truth'))
        for folder, kind, box, crop_evidence in boxes:
            panels = [raw.crop(box)] + [im.crop(box) for im in overlays]
            combined = joined(panels)
            label = '原图_初始SMPLX' if initial_only else '原图_初始SMPLX_拟合后SMPLX'
            prefix = '初始检查' if initial_only else '全94帧共享体型'
            filename = f'20261029_{prefix}_{name}_视频帧{row["source_frame_index"]:06d}_{kind}_从左到右_{label}.png'
            target = output / folder / filename
            if target.exists():
                raise FileExistsError('Do not overwrite previous image: ' + str(target))
            target.parent.mkdir(parents=True, exist_ok=True)
            combined.save(target)
            with Image.open(target) as saved:
                saved.load()
                if saved.size != combined.size or not np.array_equal(
                        np.asarray(saved)[:, :panels[0].width], np.asarray(panels[0])):
                    raise ValueError('Saved original panel changed pixels.')
            files.append(dict(frame_id=name, source_frame_index=row['source_frame_index'],
                snapshot_slots={tag: lookup[name] for tag, _, lookup in snapshots},
                source_image=str(image_path), source_image_sha256=row['source_sha256'],
                crop_integer_xyxy=box, crop_evidence=crop_evidence, native_panel_wh=list(panels[0].size),
                path=str(target), sha256=sha(target), wh=list(combined.size),
                panels=['raw'] + [tag + '_saved_camera_mesh_overlay' for tag, _, _ in snapshots]))
        print(json.dumps(dict(frame=name, complete=True, images=len(boxes))), flush=True)
    for p, expected in hashes.items():
        if sha(p) != expected:
            raise ValueError('Read-only source changed during render: ' + p)
    report = dict(updated=datetime.now(timezone.utc).isoformat(), state='initial_CPU_render_check' if initial_only else 'saved_mesh_images_generated_visual_review_pending',
        selected_frame_ids=list(selected), all94_views_rendered=False,
        review_face_frames=[f for f in selected if f in REVIEW_FRAMES],
        reference_frame_id=manifest['reference_frame_id'], initial_only=initial_only,
        output=str(output), image_count=len(files), files=files, source_sha256=hashes,
        script_sha256=sha(__file__), original_panels_pixel_identical=True,
        image_text_drawn=False, resampling=False, image_warp=False, new_model_forward=False,
        camera_or_fit_changed=False, GPU_calls=0, Gaussian_steps=0,
        projection_readback_max_px=dict(initial=initial_parity, final=final_parity),
        renderer='Reused depth-sorted CPU shaded triangle overlay with 0.6 mesh / 0.4 original. Not z-buffer-exact native rasterization, not 3DGS.',
        limits=['Only seven fixed sampled views planned, not exhaustive94 visual acceptance.',
                'Face crops only the three existing reviewed frames; remaining91 face observations stay unknown.',
                'Original image/crops use unchanged native pixels. Transparent mesh is bare SMPL-X, not clothing/hair/footwear reconstruction.',
                'Projection readback agreement verifies saved geometry/K arithmetic, not physical camera or fit accuracy.',
                'No new-view geometry, occluded feet, or unseen back-side acceptance is established.',
                'The initial-only check never represents a final fit or writes the 005 record tree.'],
        actual_visual_review_pending=True, real_quality_accepted=False,
        WPS_cloud_sync_verified=False, official_viewer_verified=False)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(receipt=str(receipt), images=len(files)), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--initial', required=True)
    parser.add_argument('--final')
    parser.add_argument('--initial-only', action='store_true')
    parser.add_argument('--crops', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--receipt', required=True)
    parser.add_argument('--frames', nargs='+')
    main(parser.parse_args())
