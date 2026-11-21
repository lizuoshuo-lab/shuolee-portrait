"""Training-view-only COCO keypoint observations for body-alignment diagnosis."""
from pathlib import Path
import hashlib
import itertools
import json
import time
import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
from torchvision.transforms.functional import to_tensor

LAB = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
SCENE = LAB / 'data/dynamic-human/argoverse2/processed/training/000'
OUT = LAB / 'outputs/dynamic-human-training-alignment-013'
VIEWS = [(129, 4), (131, 4), (149, 6), (151, 6)]


def iou(a, b):
    lower, upper = np.maximum(a[:2], b[:2]), np.minimum(a[2:], b[2:])
    overlap = float(np.prod(np.maximum(upper - lower, 0)))
    return overlap / max(float(np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2])) - overlap, 1e-8)


def main():
    assert not (OUT / 'keypoint-observations.json').exists(), 'Preserve previous observations'
    OUT.mkdir(parents=True, exist_ok=True)
    assert all(f % 10 != 0 for f, _ in VIEWS)
    weights = KeypointRCNN_ResNet50_FPN_Weights.COCO_V1
    protocol = dict(status='frozen_before_training_view_inference', views=VIEWS, target=79,
        model='torchvision KeypointRCNN ResNet50 FPN COCO_V1', weights_url=weights.url,
        device='cpu', torchvision_version=torchvision.__version__,
        purpose='Independent2Dbody observations to diagnose initial alignment; no training changes yet.',
        association='List all person detections above0.8confidence and their IoU with the projected79box; take maximum only as a visual-review candidate.',
        permitted_joints_for_future_fit='Only COCO body joints5..16. Facial landmarks are excluded; source faces stay anonymized.',
        limitations=['Predicted keypoints are not ground truth.', 'Candidate-person association must be visually reviewed.',
            'These four training views cannot establish generalization or full-sequence alignment.'])
    (OUT / 'keypoint-protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    torch.set_num_threads(4)
    torch.manual_seed(73)
    started = time.perf_counter()
    model = keypointrcnn_resnet50_fpn(weights=weights, progress=False).cpu().eval()
    print('Official pretrained keypoint model loaded on CPU', flush=True)
    annotations = json.loads((SCENE / 'instances/instances_info.json').read_text())['79']['frame_annotations']
    stamps = json.loads((SCENE / 'capture_timestamps.json').read_text())['frames']
    signs = np.array(list(itertools.product((-1., 1.), repeat=3)))
    results = []
    with torch.inference_mode():
        for frame, camera in VIEWS:
            path = SCENE / 'images' / f'{frame:03d}_{camera}.jpg'
            original = Image.open(path).convert('RGB')
            ann_idx = annotations['frame_idx'].index(frame)
            camera_entry = next(c for c in stamps[frame]['cameras'] if int(c['camera_id']) == camera)
            transform = np.linalg.inv(np.loadtxt(SCENE / camera_entry['capture_pose'])) @ np.asarray(annotations['obj_to_world'][ann_idx])
            corners = signs * np.asarray(annotations['box_size'][ann_idx])[None] / 2
            xyz = corners @ transform[:3, :3].T + transform[:3, 3]
            assert xyz[:, 2].min() > .1
            fx, fy, cx, cy = np.loadtxt(SCENE / 'intrinsics' / f'{camera}.txt')[:4]
            uv = xyz[:, :2] / xyz[:, 2:] * [fx, fy] + [cx, cy]
            box = np.concatenate([np.maximum(uv.min(0), 0), np.minimum(uv.max(0), original.size)])
            pred = model([to_tensor(original)])[0]
            selected = torch.where(pred['scores'] >= .8)[0].tolist()
            candidates = []
            for index in selected:
                detection_box = pred['boxes'][index].numpy()
                candidates.append(dict(index=index, detection_score=float(pred['scores'][index]),
                    box=detection_box.tolist(), projected_box_iou=iou(box, detection_box),
                    keypoints=pred['keypoints'][index].tolist(), keypoint_scores=pred['keypoints_scores'][index].tolist()))
            assert candidates, (frame, camera)
            best = max(candidates, key=lambda item: item['projected_box_iou'])
            overlay = original.copy()
            draw = ImageDraw.Draw(overlay)
            # Show only body joints; no facial landmark overlay or face restoration.
            for index, (x, y, _) in enumerate(best['keypoints'][5:], 5):
                draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill='#ffcc22', outline='#202030', width=2)
                draw.text((x + 8, y - 7), str(index), fill='#ffcc22', stroke_fill='#202030', stroke_width=1)
            lo = np.maximum(np.floor(np.minimum(box[:2], best['box'][:2]) - 40), 0).astype(int)
            hi = np.minimum(np.ceil(np.maximum(box[2:], best['box'][2:]) + 40), original.size).astype(int)
            crop = tuple(map(int, [*lo, *hi]))
            a, b = original.crop(crop), overlay.crop(crop)
            pair = Image.new('RGB', (a.width * 2, a.height))
            pair.paste(a, (0, 0))
            pair.paste(b, (a.width, 0))
            name = f'frame{frame:03d}-camera{camera}-keypoints.png'
            pair.save(OUT / name)
            row = dict(frame=frame, camera=camera, training=True, target_projected_box=box.tolist(),
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                candidates=candidates, candidate_index=best['index'], figure=name)
            results.append(row)
            print(json.dumps(dict(frame=frame, camera=camera, candidates=len(candidates), selected_iou=best['projected_box_iou'])), flush=True)
    weight_path = Path(torch.hub.get_dir()) / 'checkpoints' / weights.url.rsplit('/', 1)[1]
    report = dict(status='four_training_views_inferred_candidate_association_and_joint_quality_require_visual_review',
        protocol=protocol, weights_sha256=hashlib.sha256(weight_path.read_bytes()).hexdigest(),
        views=results, seconds=time.perf_counter()-started, optimized_steps=0, used_heldout_RGB=False)
    (OUT / 'keypoint-observations.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=report['status'], seconds=report['seconds'])), flush=True)


if __name__ == '__main__':
    main()
