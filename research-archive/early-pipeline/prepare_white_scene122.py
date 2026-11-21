"""N12a: one native-frame/person-exclusion intake for the new white-T capture."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
from pathlib import Path
from datetime import datetime, timezone
import sys, json, hashlib, subprocess, time, traceback
import numpy as np
from PIL import Image

R = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
I = R / 'video-update-20261027'
P = I / 'clip03-scene-support-plan.json'
O = I / 'clip03-scene-support'
PROJECT = Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118')
ASSETS = Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets')
FRAMES = [184, 200, 216, 232, 248, 264, 280]

def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p, v): p.write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def register():
    assert not P.exists() and not O.exists(), 'Do not repeat this intake.'
    source = next(v for v in read(I / 'content-review.json')['videos'] if v['id'] == 'clip03')
    assert source['sha256'] == '0a4d31cfcf31036ef2244b2709daaf29b425c1921c5516183302670728d030de'
    assert (source['width'], source['height'], source['frames']) == (1520, 2704, 464)
    files = [Path(source['path']), I / 'content-review.json', I / 'output-intrinsics-audit/result.json',
             I / 'clip03/sample_02.png', ASSETS / 'sam2/sam2.1_hiera_tiny.pt', ASSETS / 'detection/yolov8n.pt',
             PROJECT / 'configs/server-assets.json', PROJECT / 'src/shuolee_portrait/observations.py',
             PROJECT / 'src/shuolee_portrait/runtime.py']
    hashes = {str(f): sha(f) for f in files}
    assert hashes[source['path']] == source['sha256']
    save(P, dict(stage='N12a', registered=now(), source_sha256=sha(Path(__file__)), source=source,
        question='Does the new white-T capture expose visually identifiable rigid scene materials at multiple apparent distances, unlike the prior near-planar/water-side capture?',
        causal_scope='Independent capture/pipeline validation, not a one-variable proof about the hat video. No diagnosis transfer or old fitted camera/geometry initialization.',
        frames=FRAMES, frame_selection='Seven fixed encoded-frame indices centered at the pre-existing midpoint232 with step16, chosen before new decoding or feature scores.',
        K_integer_centers=read(I / 'output-intrinsics-audit/result.json')['variants'][0]['K'],
        K_contract='First YAML K on4000x3000, height resize2704, ideal center crop1520, explicit half-pixel centers. Same conditionalK as prior diagnostic; not an independently measured exportK.',
        operation='FFmpeg exact-frame native PNG decode with no autorotation/resizing/undistortion. Existing YOLOv8n class0 confidence0.45 plus SAM2.1tiny CPU masks for every detected person; expand each prompt4percent, positive boxcenter, union silhouettes, dilate32 native pixels.',
        review='Actually inspect all7 full native frames and overlays. Identify building/step/fence/ground supports, repeated patterns, reflections, foliage and occlusions. Background status is only a candidate; no automatic static certification.',
        stop='Source/hash/raster mismatch, no detected people, more than8 detections, SAMscore<0.5, union area outside0.5to95percent, decode120s or mask180s -> preserve failure without threshold changes. Bad masks stop feature use.',
        budget=dict(attempts=1, decoded_frames=7, decode_seconds=120, mask_CPU_seconds=180, GPU=0, new_features=0, new_cameras=0, training_steps=0),
        downstream='Feature/camera work requires a separately recorded plan after native scene/mask review. These7frames cannot stand in for complete video registration or subpixel face acceptance.',
        input_hashes=hashes))
    print('N12A_PLAN_REGISTERED', flush=True)

def run():
    plan = read(P)
    assert not O.exists() and plan['source_sha256'] == sha(Path(__file__))
    for f, h in plan['input_hashes'].items(): assert sha(Path(f)) == h, f
    O.mkdir()
    start = time.monotonic()
    save(O / 'progress.json', dict(state='decoding_native', pid=os.getpid(), updated=now()))
    expr = "select='" + '+'.join(f'eq(n,{f})' for f in FRAMES) + "'"
    command = ['/usr/bin/ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-noautorotate',
               '-i', plan['source']['path'], '-vf', expr, '-vsync', '0', '-frames:v', '7', str(O / 'decode_%02d.png')]
    cp = subprocess.run(command, capture_output=True, text=True, timeout=120)
    (O / 'decode.log').write_text(cp.stderr, encoding='utf-8')
    save(O / 'decode-command.json', dict(argv=command, returncode=cp.returncode))
    assert cp.returncode == 0
    for index, f in enumerate(FRAMES, 1):
        src = O / f'decode_{index:02d}.png'
        assert Image.open(src).size == (1520, 2704)
        src.rename(O / f'frame_{f:06d}.png')
    assert sha(O / 'frame_000232.png') == plan['input_hashes'][str(I / 'clip03/sample_02.png')], 'Midframe must reproduce the previous native FFmpeg decode byte-for-byte.'
    sys.path.insert(0, str(PROJECT / 'src'))
    from shuolee_portrait import runtime as rt
    rt.configure(ASSETS)
    from shuolee_portrait.observations import load_detector
    import torch, cv2
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    manifest = read(PROJECT / 'configs/server-assets.json')['files']
    for name in ['sam2/sam2.1_hiera_tiny.pt', 'detection/yolov8n.pt']:
        assert sha(ASSETS / name) == next(r['sha256'] for r in manifest if r['path'] == name)
    detector = load_detector().to('cpu')
    network = build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml', str(ASSETS / 'sam2/sam2.1_hiera_tiny.pt'), device='cpu', apply_postprocessing=False)
    segmenter = SAM2ImagePredictor(network, max_hole_area=0, max_sprinkle_area=0)
    mask_start = time.monotonic()
    overview = Image.new('RGB', (304 * 7, 541))
    rows = []
    with torch.inference_mode():
        for i, f in enumerate(FRAMES):
            assert time.monotonic() - mask_start < 180
            rgb = np.array(Image.open(O / f'frame_{f:06d}.png').convert('RGB'))
            height, width = rgb.shape[:2]
            boxes = detector.predict(rgb[:, :, ::-1].copy(), classes=[0], conf=.45, verbose=False, device='cpu')[0].boxes.xyxy.cpu().numpy()
            assert 1 <= len(boxes) <= 8
            segmenter.set_image(rgb)
            union = np.zeros((height, width), dtype=np.uint8)
            people = []
            for box in boxes:
                pad = .04 * max(box[2:] - box[:2])
                prompt = np.clip(box + [-pad, -pad, pad, pad], [0, 0, 0, 0], [width, height, width, height])
                masks, scores, _ = segmenter.predict(box=prompt, point_coords=((prompt[:2] + prompt[2:]) / 2)[None], point_labels=np.ones(1, dtype=np.int32), multimask_output=False)
                assert scores[0] >= .5
                union |= masks[0].astype('uint8')
                people.append(dict(detected_box=box.tolist(), prompted_box=prompt.tolist(), score=float(scores[0]), pixels=int(masks[0].sum())))
            assert .005 < float(union.mean()) < .95
            exclusion = cv2.dilate(union, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (65, 65)), iterations=1).astype(bool)
            Image.fromarray(union * 255).save(O / f'person_{f:06d}.png')
            Image.fromarray(exclusion.astype('uint8') * 255).save(O / f'exclusion_{f:06d}.png')
            overlay = rgb.astype(float)
            overlay[exclusion & (union == 0)] = .55 * overlay[exclusion & (union == 0)] + .45 * np.array([255, 190, 0])
            overlay[union > 0] = .72 * overlay[union > 0] + .28 * np.array([235, 65, 85])
            image = Image.fromarray(overlay.astype('uint8'))
            image.save(O / f'overlay_{f:06d}.png')
            overview.paste(image.resize((304, 541), Image.Resampling.LANCZOS), (304 * i, 0))
            rows.append(dict(frame=f, people=people, person_pixels=int(union.sum()), exclusion_pixels=int(exclusion.sum()), source_sha256=sha(O / f'frame_{f:06d}.png'), exclusion_sha256=sha(O / f'exclusion_{f:06d}.png')))
            save(O / 'progress.json', dict(state='person_exclusion_cpu', pid=os.getpid(), updated=now(), done=i + 1, total=7))
            print('N12A_MASK', f, len(boxes), flush=True)
    overview.save(O / 'all7_overlays.png')
    for f, h in plan['input_hashes'].items(): assert sha(Path(f)) == h
    save(O / 'result.json', dict(stage='N12a', state='complete_pending_actual_review', updated=now(), elapsed_seconds=time.monotonic() - start, rows=rows, K_integer_centers=plan['K_integer_centers'], original_midframe_byte_equal=True, new_features=False, new_camera=False, training_started=False, physical_static_support_certified=False))
    save(O / 'progress.json', dict(state='complete_pending_actual_review', pid=None, updated=now()))
    print('N12A_NATIVE_MASKS_READY', flush=True)

if __name__ == '__main__':
    try:
        globals()[sys.argv[1]]()
    except BaseException as error:
        if O.exists():
            save(O / 'failure.json', dict(updated=now(), error=repr(error), traceback=traceback.format_exc()))
            save(O / 'progress.json', dict(state='failed', pid=None, updated=now()))
        raise
