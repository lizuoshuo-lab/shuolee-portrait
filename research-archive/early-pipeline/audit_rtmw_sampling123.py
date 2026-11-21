"""Read-only sampling audit of the exact RTMW input geometry; no inference."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import time

import cv2
import numpy as np
from PIL import Image

T = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
F = T/'shuolee/current/local-clothed-surface-123/smplifyx-reference-control/official-first-cycle/full94-shared-shape'
OUT = F/'reviews/rtmw-input-sampling'
ENV = Path('/opt/gs-slam-lab/shuolee-portrait-clean-001/env/lib/python3.10/site-packages/rtmlib/tools')
PROJECT = Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118')
RECORDS = Path('/mnt/d/00/记录/005/分析')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    sources = [ENV/'pose_estimation/pre_processings.py', ENV/'pose_estimation/rtmpose.py',
               ENV/'pose_estimation/post_processings.py', ENV/'base.py',
               PROJECT/'src/shuolee_portrait/observations.py']
    provenance = json.loads((F/'input-provenance.json').read_text())
    inputs = next(Path(p) for p in provenance['input_files'] if p.endswith('/data/inputs.json'))
    observations = next(Path(p) for p in provenance['input_files'] if p.endswith('/data/observations.npz'))
    for p in (inputs, observations):
        assert sha(p) == provenance['input_files'][str(p)]
    sources += [inputs, observations]
    before = {str(p): sha(p) for p in sources}
    save(OUT/'plan.json', dict(created=datetime.now(timezone.utc).isoformat(),
        question='How much native facial detail reaches the existing whole-body detector?',
        method='Existing complete94 boxes and51 inner-face outputs; actual installed crop geometry; single-branch SimCC argmax split_ratio2.',
        budget='One CPU read-only pass; no detector inference, fitting, training or source changes.',
        selected_frames=['frame_000032', 'frame_000079', 'frame_000094'], source_hashes=before,
        limits=['Decode-bin spacing is not measured error or a hard accuracy floor.',
                'Final stored coordinates average original and flipped predictions.',
                'Detected landmark envelopes are not physical ground truth or visibility.']))
    spec = importlib.util.spec_from_file_location('rtmw_sampling_preprocess', sources[0])
    pre = importlib.util.module_from_spec(spec); spec.loader.exec_module(pre)
    rows = json.loads(inputs.read_text())['records']
    with np.load(observations, allow_pickle=False) as z:
        ids = z['frame_ids'].astype(str).tolist(); boxes = z['boxes'].copy(); points = z['keypoints'].copy()
    assert ids == [r['frame'] for r in rows]
    assert len(ids) == 94
    start = time.monotonic(); results = []; images = []
    selected = {'frame_000032', 'frame_000079', 'frame_000094'}
    for i, row in enumerate(rows):
        center, original_scale = pre.bbox_xyxy2cs(boxes[i], padding=1.25)
        _, scale = pre.top_down_affine((288,384), original_scale, center, np.zeros((1,1,3),np.uint8))
        spacing = scale / np.array([288,384]) / 2
        span = np.ptp(points[i,40:91], axis=0)
        reduced = span / scale * [288,384]
        results.append(dict(frame=ids[i], bbox_native=boxes[i].tolist(), padded_scale_native=scale.tolist(),
            inner_face_landmark_envelope_native_px=span.tolist(),
            inner_face_landmark_envelope_detector_px=reduced.tolist(),
            single_branch_decode_bin_native_px=spacing.tolist()))
        if ids[i] not in selected:
            continue
        assert sha(row['image']) == row['source_sha256']
        rgb = np.asarray(Image.open(row['image']).convert('RGB'))
        detector, actual_scale = pre.top_down_affine((288,384), original_scale, center, rgb)
        assert np.array_equal(scale, actual_scale)
        lo = np.maximum(np.floor(points[i,40:91].min(0)-20).astype(int), 0)
        hi = np.minimum(np.ceil(points[i,40:91].max(0)+21).astype(int), [rgb.shape[1],rgb.shape[0]])
        matrix = pre.get_warp_matrix(center, scale, 0, output_size=(288,384))
        transformed = np.c_[np.array([lo,hi]),np.ones(2)] @ matrix.T
        nl = np.maximum(np.floor(transformed[0]).astype(int),0)
        nh = np.minimum(np.ceil(transformed[1]).astype(int),[288,384])
        raw = rgb[lo[1]:hi[1],lo[0]:hi[0]]; small=detector[nl[1]:nh[1],nl[0]:nh[0]]
        assert raw.size and small.size
        canvas=np.full((max(len(raw),len(small)),raw.shape[1]+small.shape[1]+12,3),245,np.uint8)
        canvas[:len(raw),:raw.shape[1]]=raw
        canvas[:len(small),raw.shape[1]+12:]=small
        name=f'20261030_二维检测输入_{ids[i]}_原生脸部_实际网络脸部_各自像素.png'
        destination=RECORDS/name; assert not destination.exists()
        Image.fromarray(canvas).save(destination)
        local=OUT/(ids[i]+'-native-vs-detector.png');Image.fromarray(canvas).save(local)
        images.append(dict(frame=ids[i],record_path=str(destination),review_path=str(local),sha256=sha(local),
            raw_crop_xyxy=np.r_[lo,hi].tolist(),detector_crop_xyxy=np.r_[nl,nh].tolist(),
            raw_wh=[raw.shape[1],raw.shape[0]],detector_wh=[small.shape[1],small.shape[0]],
            no_labels_drawn=True,no_preview_upscale=True))
    assert all(sha(Path(p))==v for p,v in before.items())
    save(OUT/'result.json', dict(state='read_only_sampling_complete', seconds=time.monotonic()-start,
        actual_frames=94, model_input_wh=[288,384], simcc_split_ratio=2,
        records=results, images=images, all_sources_unchanged=True, source_hashes=before,
        model_inference_calls=0, gpu_used=False, optimizer_steps=0, quality_accepted=False,
        limits=json.loads((OUT/'plan.json').read_text())['limits'],
        color_contract='PIL RGB passes through installed affine/normalization/CHW transpose without a channel reversal; no evidence for an RGB/BGR interface fix.'))
    print(json.dumps(dict(state='read_only_sampling_complete', selected=[r for r in results if r['frame'] in selected], images=images), ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
