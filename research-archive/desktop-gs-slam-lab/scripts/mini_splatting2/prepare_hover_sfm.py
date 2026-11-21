"""Non-destructive COLMAP preparation of computer00's short handheld capture.

Run in a new output directory. No original images, camera files or scripts are
overwritten. All extracted views participate in SfM; Mini-Splatting2's --eval
split is reserved for appearance training, not a held-out calibration test.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

LAB = Path(__file__).resolve().parents[2]


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fps', type=float, default=8)
    args = parser.parse_args()
    if not 0 < args.fps <= 24:
        raise ValueError('Invalid sampling rate')
    video = LAB / 'inputs/computer00-local-pipeline/HOVER_VERSA_0102.mp4'
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    images = out / 'raw_images'
    images.mkdir()
    sparse = out / 'raw_sparse'
    sparse.mkdir()
    ffmpeg = Path('C:/Users/Administrator/ffmpeg/bin/ffmpeg.exe')
    ffprobe = ffmpeg.with_name('ffprobe.exe')
    tool = Path('C:/Users/Administrator/colmap-x64-windows-cuda')
    colmap = tool / 'bin/colmap.exe'
    env = os.environ.copy()
    env['PATH'] = str(tool / 'bin') + os.pathsep + env.get('PATH', '')
    env['QT_PLUGIN_PATH'] = str(tool / 'plugins')
    state = dict(status='running', input=str(video), output=str(out), fps=args.fps,
                 input_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
                 resize='one half of each dimension: 760 x 1352',
                 camera_model='SIMPLE_RADIAL', use_gpu=False, threads=4,
                 holdout_scope='all views for SfM; every eighth registered image for subsequent --eval appearance test',
                 commands=[])
    write_json(out / 'state.json', state)
    start = time.time()

    def run(label, cmd):
        tick = time.time()
        state['current_stage'] = label
        state['commands'].append([str(x) for x in cmd])
        write_json(out / 'state.json', state)
        with (out / f'{label}.log').open('w', encoding='utf-8') as log:
            process = subprocess.run([str(x) for x in cmd], env=env, stdout=log, stderr=subprocess.STDOUT)
        state.setdefault('stage_seconds', {})[label] = time.time() - tick
        if process.returncode:
            raise RuntimeError(f'{label} failed with exit {process.returncode}; see {label}.log')

    try:
        metadata = subprocess.check_output([str(ffprobe), '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(video)], text=True, encoding='utf-8')
        write_json(out / 'input-video.json', json.loads(metadata))
        run('extract', [ffmpeg, '-hide_banner', '-loglevel', 'warning', '-n', '-i', video,
                        '-vf', f'fps={args.fps},scale=760:1352', '-q:v', '1', '-start_number', '0', images / '%06d.jpg'])
        state['extracted_images'] = len(list(images.glob('*.jpg')))
        if state['extracted_images'] < 16:
            raise ValueError('Too few images for this baseline')
        db = out / 'database.db'
        run('features', [colmap, 'feature_extractor', '--database_path', db, '--image_path', images,
                         '--ImageReader.camera_model', 'SIMPLE_RADIAL', '--ImageReader.single_camera', '1',
                         '--SiftExtraction.use_gpu', '0', '--SiftExtraction.num_threads', '4',
                         '--SiftExtraction.max_num_features', '8192'])
        run('matching', [colmap, 'sequential_matcher', '--database_path', db,
                         '--SiftMatching.use_gpu', '0', '--SiftMatching.num_threads', '4',
                         '--SiftMatching.guided_matching', '1', '--SequentialMatching.overlap', '10'])
        run('mapping', [colmap, 'mapper', '--database_path', db, '--image_path', images, '--output_path', sparse,
                        '--Mapper.num_threads', '4', '--Mapper.ba_use_gpu', '0',
                        '--Mapper.ba_refine_principal_point', '0'])
        sys.path.insert(0, str(LAB / 'repos/hierarchical-3d-gaussians/preprocess'))
        import read_write_model as rw
        candidates = []
        for model in sorted(sparse.iterdir()):
            if not (model / 'images.bin').exists():
                continue
            cameras, registered, points = rw.read_model(str(model), ext='.bin')
            candidates.append(dict(path=str(model), registered=len(registered), points=len(points)))
        if not candidates:
            raise ValueError('COLMAP produced no reconstruction')
        best = max(candidates, key=lambda x: x['registered'])
        state['models'] = candidates
        state['selected_model'] = best
        if best['registered'] < max(16, state['extracted_images'] * 2 / 3):
            raise ValueError('Largest model registers fewer than two thirds of extracted views')
        dataset = out / 'dataset'
        run('undistort', [colmap, 'image_undistorter', '--image_path', images, '--input_path', best['path'],
                          '--output_path', dataset, '--output_type', 'COLMAP', '--max_image_size', '1352'])
        # COLMAP exports sparse/*.bin; Mini-Splatting2 expects sparse/0/*.bin.
        normalized = dataset / 'sparse/0'
        normalized.mkdir(exist_ok=False)
        for name in ('cameras.bin', 'images.bin', 'points3D.bin'):
            shutil.copy2(dataset / 'sparse' / name, normalized / name)
        cameras, registered, points = rw.read_model(str(normalized), ext='.bin')
        names = sorted(image.name for image in registered.values())
        missing = [name for name in names if not (dataset / 'images' / name).is_file()]
        if missing or not all(c.model in ('PINHOLE', 'SIMPLE_PINHOLE') for c in cameras.values()):
            raise ValueError('Undistorted dataset invalid')
        state['appearance_split'] = dict(train=[n for i,n in enumerate(names) if i % 8],
                                         test=names[::8], rule='official LLFF hold=8 after lexicographic name sort')
        state['mean_sparse_reprojection_error_px'] = float(sum(p.error for p in points.values()) / len(points)) if points else None
        state['dataset'] = str(dataset)
        state['status'] = 'complete'
    except Exception as exc:
        state['status'] = 'failed'
        state['error'] = str(exc)
        raise
    finally:
        state['elapsed_seconds'] = time.time() - start
        write_json(out / 'state.json', state)
    print(json.dumps(state, ensure_ascii=False))


if __name__ == '__main__':
    main()
