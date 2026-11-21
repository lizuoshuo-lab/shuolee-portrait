"""AV2 scene 000: reuse pinned SDK and unmodified DriveStudio export methods.

Adds capture-time camera poses and an audit; does not label box masks as segmentation.
Original JPEG bytes and official-format exports are preserved separately from diagnostics.
"""
from pathlib import Path
from collections import defaultdict
from types import SimpleNamespace
import ast
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

LAB = Path(__file__).resolve().parents[2]
SDK = LAB / 'repos/drivestudio-av2-api'
REPO = LAB / 'repos/drivestudio'
OUT = LAB / 'outputs/dynamic-human-street-005'
LOG = '00a6ffc1-6ce9-3bc3-a060-6006e9893a1a'
RAW = LAB / 'data/dynamic-human/argoverse2/sensor/train' / LOG
DEST = LAB / 'data/dynamic-human/argoverse2/processed/training/000'
CAMS = ['ring_front_center', 'ring_front_left', 'ring_front_right',
        'ring_side_left', 'ring_side_right', 'ring_rear_left', 'ring_rear_right']
TARGET_TRACK = 'ec986b20-004d-49f2-b9b0-d0d017bdeb1e'
sys.path.insert(0, str(SDK / 'src'))
from av2.datasets.sensor.sensor_dataloader import SensorDataloader, SynchronizedSensorData
from av2.datasets.sensor.constants import RingCameras
from av2.geometry.camera.pinhole_camera import PinholeCamera
from av2.structures.cuboid import CuboidList
from av2.structures.sweep import Sweep
from av2.structures.timestamped_image import TimestampedImage
from av2.utils.io import read_city_SE3_ego, read_img


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def official_exporter():
    """Load only CPU export methods verbatim, without importing GPU visualization tools."""
    source = REPO / 'datasets/argoverse/argoverse_preprocess.py'
    tree = ast.parse(source.read_text(encoding='utf8'))
    names = {'save_calib', 'save_lidar', 'save_pose', 'save_dynamic_mask'}
    nodes = [n for n in tree.body if isinstance(n, ast.Assign)
             and isinstance(n.targets[0], ast.Name) and n.targets[0].id.startswith('AV2_')]
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ArgoVerseProcessor')
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(methods) == len(names)
    nodes += methods
    namespace = dict(np=np, os=os, Image=Image, SynchronizedSensorData=SynchronizedSensorData)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    owner = SimpleNamespace(save_dir=str(DEST.parent), cam_list=CAMS)
    return owner, namespace, dict(path=str(source), sha256=sha(source),
                                  verbatim_export_methods=sorted(names))


def review_panels(item, filename):
    """Observed camera pixels plus projected annotation edges: a data-alignment diagnostic."""
    original = Image.open(item['source']).convert('RGB')
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 23)
    panels = []
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
             (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    for key, title, color in [('none', 'Source image', None),
                              ('legacy_uv', 'LiDAR-time ego pose', '#ff833e'),
                              ('compensated_uv', 'Camera-time ego pose', '#2bc7d5')]:
        im = original.copy()
        if key != 'none':
            points = item[key]
            draw = ImageDraw.Draw(im)
            for a, b in edges:
                draw.line([tuple(points[a]), tuple(points[b])], fill=color, width=5)
        im.thumbnail((760, 640), Image.Resampling.LANCZOS)
        panel = Image.new('RGB', (780, 715), 'white')
        panel.paste(im, ((780-im.width)//2, 54))
        d = ImageDraw.Draw(panel)
        d.text((18, 12), title, font=font, fill='#182c36')
        d.text((18, 675), f"frame {item['frame_idx']:03d}, camera {item['camera_id']}, dt {item['dt_ms']:+.2f} ms", font=font, fill='#182c36')
        panels.append(panel)
    canvas = Image.new('RGB', (2340, 715), 'white')
    for i, panel in enumerate(panels):
        canvas.paste(panel, (780*i, 0))
    canvas.save(filename)
    # Native-resolution crop: no resampling, enhancement, or generative restoration.
    uv = np.array(item['compensated_uv'])
    lo = np.maximum(np.floor(uv.min(0))-60, 0).astype(int)
    hi = np.minimum(np.ceil(uv.max(0))+60, original.size).astype(int)
    original.crop((*lo, *hi)).save(filename.with_name(filename.stem+'-person-source.png'))


def main():
    started = time.time()
    OUT.mkdir(exist_ok=True, parents=True)
    assert (REPO / 'data/argoverse_train_list.txt').read_text().splitlines()[0] == LOG
    assert shutil.disk_usage(LAB).free > 5_000_000_000
    for subdir in ['images', 'intrinsics', 'extrinsics', 'ego_pose', 'lidar', 'instances',
                   'camera_pose_capture', 'sky_masks', 'humanpose']:
        (DEST / subdir).mkdir(parents=True, exist_ok=True)
    loader = SensorDataloader(dataset_dir=RAW.parent.parent, with_cache=False,
                              cam_names=tuple(RingCameras(x) for x in CAMS))
    pose = read_city_SE3_ego(RAW)
    models = {name: PinholeCamera.from_feather(RAW, name) for name in CAMS}
    all_boxes = CuboidList.from_feather(RAW / 'annotations.feather')
    boxes_by_ts = defaultdict(list)
    for box in all_boxes.cuboids:
        boxes_by_ts[int(box.timestamp_ns)].append(box)
    exporter, funcs, source = official_exporter()
    camera_files = {name: sorted((RAW / 'sensors/cameras' / name).glob('*.jpg')) for name in CAMS}
    camera_times = {name: np.array([int(p.stem) for p in files], dtype=np.int64)
                    for name, files in camera_files.items()}
    instances, frame_instances, records, skipped, image_receipts, projections = {}, {}, [], [], [], []
    target_reviews = []
    for lidar_file in sorted((RAW / 'sensors/lidar').glob('*.feather')):
        ts = int(lidar_file.stem)
        paths = {}
        for name in CAMS:
            sdk_path = loader.find_closest_target_fpath(split='train', log_id=LOG,
                src_sensor_name='lidar', src_timestamp_ns=ts, target_sensor_name=name)
            delta = np.abs(camera_times[name]-ts)
            nearest = camera_files[name][int(delta.argmin())]
            # Independent integer comparison validates the SDK's timestamp pairing.
            if delta.min() <= 25_000_000:
                assert sdk_path == nearest
            else:
                assert sdk_path is None
            if sdk_path is not None and int(sdk_path.stem) in pose:
                paths[name] = sdk_path
        if len(paths) != 7 or ts not in pose:
            skipped.append(str(ts))
            continue
        frame = len(records)
        imagery = {name: TimestampedImage(read_img(path, channel_order='BGR'), models[name], int(path.stem))
                   for name, path in paths.items()}
        boxes = boxes_by_ts[ts]
        datum = SynchronizedSensorData(sweep=Sweep.from_feather(lidar_file),
            timestamp_city_SE3_ego_dict=pose, log_id=LOG, sweep_number=frame,
            num_sweeps_in_log=len(loader), timestamp_ns=ts,
            annotations=CuboidList(boxes), synchronized_imagery=imagery)
        if frame == 0:
            funcs['save_calib'](exporter, datum, 0, frame)
        funcs['save_lidar'](exporter, datum, 0, frame)
        funcs['save_pose'](exporter, datum, 0, frame)
        for category in ['all', 'human', 'vehicle']:
            funcs['save_dynamic_mask'](exporter, datum, 0, frame, category)
        frame_instances[frame] = []
        for box in boxes:
            if box.category not in funcs['AV2_DYNAMIC_CLASSES']:
                continue
            obj_id = box.track_uuid
            if obj_id not in instances:
                instances[obj_id] = dict(id=obj_id, class_name=box.category,
                    frame_annotations=dict(frame_idx=[], obj_to_world=[], box_size=[]))
            ann = instances[obj_id]['frame_annotations']
            ann['frame_idx'].append(frame)
            ann['obj_to_world'].append((pose[ts].transform_matrix @ box.dst_SE3_object.transform_matrix).tolist())
            ann['box_size'].append([box.length_m, box.width_m, box.height_m])
            frame_instances[frame].append(obj_id)
        record = dict(frame_idx=frame, lidar_timestamp_ns=str(ts), cameras=[])
        for cam_id, name in enumerate(CAMS):
            path, cam = paths[name], imagery[name]
            img_dest = DEST / 'images' / f'{frame:03d}_{cam_id}.jpg'
            shutil.copyfile(path, img_dest)
            source_hash = sha(path)
            assert sha(img_dest) == source_hash
            image_receipts.append(dict(path=str(img_dest.relative_to(DEST)), source=str(path.relative_to(RAW)), sha256=source_hash))
            assert (cam.img.shape[1], cam.img.shape[0]) == (cam.camera_model.width_px, cam.camera_model.height_px)
            capture_c2w = pose[cam.timestamp_ns].transform_matrix @ cam.camera_model.ego_SE3_cam.transform_matrix
            legacy_c2w = pose[ts].transform_matrix @ cam.camera_model.ego_SE3_cam.transform_matrix
            np.savetxt(DEST / 'camera_pose_capture' / f'{frame:03d}_{cam_id}.txt', capture_c2w)
            assert np.allclose(capture_c2w[:3, :3].T @ capture_c2w[:3, :3], np.eye(3), atol=1e-8)
            dt_ms = (cam.timestamp_ns-ts)*1e-6
            record['cameras'].append(dict(camera_id=cam_id, name=name, timestamp_ns=str(cam.timestamp_ns),
                dt_from_lidar_ms=dt_ms, camera_translation_change_m=float(np.linalg.norm(capture_c2w[:3, 3]-legacy_c2w[:3, 3])),
                capture_pose=f'camera_pose_capture/{frame:03d}_{cam_id}.txt'))
            for box in boxes:
                if box.category != 'PEDESTRIAN':
                    continue
                uv0, _, ok0 = cam.camera_model.project_ego_to_img(box.vertices_m)
                uv1, _, ok1 = cam.camera_model.project_ego_to_img_motion_compensated(box.vertices_m, pose[cam.timestamp_ns], pose[ts])
                # Independent homogeneous transform checks the SDK's motion compensation.
                world = pose[ts].transform_point_cloud(box.vertices_m)
                xc = (world-capture_c2w[:3, 3]) @ capture_c2w[:3, :3]
                uv_check = xc @ cam.camera_model.intrinsics.K.T
                uv_check = uv_check[:, :2]/uv_check[:, 2:]
                np.testing.assert_allclose(uv_check, uv1, atol=1e-6, rtol=1e-8)
                if not (all(ok0) and all(ok1)):
                    continue
                item = dict(frame_idx=frame, camera_id=cam_id, track_uuid=box.track_uuid,
                    source=str(path), dt_ms=dt_ms, legacy_uv=uv0.tolist(), compensated_uv=uv1.tolist(),
                    mean_corner_shift_px=float(np.linalg.norm(uv1-uv0, axis=1).mean()),
                    ego_range_m=float(np.linalg.norm(box.dst_SE3_object.translation)))
                projections.append(item)
                if box.track_uuid == TARGET_TRACK:
                    target_reviews.append(item)
        records.append(record)
        if frame % 10 == 0:
            print('PREPROCESSED', frame+1, '/', len(loader), flush=True)
    id_map = {track: i for i, track in enumerate(instances)}
    write_json(DEST/'instances/instances_info.json', {id_map[k]: v for k, v in instances.items()})
    write_json(DEST/'instances/frame_instances.json', {k: [id_map[i] for i in v] for k, v in frame_instances.items()})
    write_json(DEST/'capture_timestamps.json', dict(timestamp_encoding='decimal strings; exact nanoseconds', frames=records))
    write_json(OUT/'image-integrity.json', image_receipts)
    write_json(OUT/'pedestrian-projection-audit.json', projections)
    assert target_reviews, 'Target pedestrian must have visible projections'
    chosen = min(target_reviews, key=lambda item: item['ego_range_m'])
    (OUT/'assets').mkdir(exist_ok=True)
    review_panels(chosen, OUT/'assets/street-camera-time-review.png')
    cams = [c for record in records for c in record['cameras']]
    delta = np.array([c['dt_from_lidar_ms'] for c in cams])
    shift = np.array([p['mean_corner_shift_px'] for p in projections])
    write_json(OUT/'preprocessing-receipt.json', dict(
        status='camera_lidar_instances_and_coarse_masks_exported_pending_semantics_and_human_pose',
        scene=LOG, destination=str(DEST), frames=len(records), source_lidar_frames=len(loader),
        skipped_lidar_timestamps_ns=skipped, images=len(image_receipts), cameras=CAMS,
        instances=len(instances), pedestrian_instances=sum(x['class_name']=='PEDESTRIAN' for x in instances.values()),
        source_sdk=json.loads((OUT/'av2-api-source-receipt.json').read_text()),
        official_export=source, script_sha256=sha(Path(__file__)),
        image_handling='Original JPEG bytes copied; unlike official re-encoding, no additional JPEG quantization.',
        camera_pose_contract='Official ego_pose is at LiDAR time; camera_pose_capture is camera-to-city at actual image timestamp. Original loader does not read the extra poses yet.',
        coarse_mask_contract='Unmodified official projected 3D bounding rectangles, strict all-eight-corners visibility; human includes all official non-rigid categories. Not person segmentation or GT masks.',
        lidar_contract='Official Nx4 float32: x,y,z,intensity. AV2 points already in ego frame at sweep reference time.',
        annotation_contract='Object-to-city transforms at LiDAR time; timestamp correction compensates ego motion only, not object motion or rolling shutter.',
        timing=dict(max_abs_dt_ms=float(abs(delta).max()), median_abs_dt_ms=float(np.median(abs(delta))),
                    max_camera_translation_change_m=max(c['camera_translation_change_m'] for c in cams)),
        projection_check=dict(fully_visible_pedestrian_camera_instances=len(projections),
            mean_corner_shift_px=float(shift.mean()), p95_corner_shift_px=float(np.percentile(shift,95)),
            max_corner_shift_px=float(shift.max()), independent_projection_tolerance_px=1e-6),
        review_sample=chosen, pending=['semantic sky/person masks', 'preprocessed human pose archive',
          'licensed SMPL template', 'capture-time loader integration', 'street model training and held-out evaluation'],
        elapsed_seconds=time.time()-started))
    print('STREET PREPROCESS COMPLETE', len(records), len(image_receipts), len(instances), flush=True)


if __name__ == '__main__':
    main()
