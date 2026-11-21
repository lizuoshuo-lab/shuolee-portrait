"""Read-only video/calibration intake; no camera fitting or training admission."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone

SOURCE = Path(r'D:\00\工作\数据集')
OUT = Path(r'D:\00\工作日志\2026-10-26\新视频标定核验')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def matrix(text, name, expected):
    part = text.split(name + ': !!opencv-matrix', 1)[1]
    match = re.search(r'data:\s*\[([^]]+)\]', part)
    values = [float(x.strip()) for x in match.group(1).split(',')]
    if len(values) != expected:
        raise ValueError(name)
    return values


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cal_path = SOURCE / 'mono_main_calib.yaml'
    text = cal_path.read_text(encoding='utf-8-sig')
    k = matrix(text, 'camera_matrix', 9)
    fov_k = matrix(text, 'camera_matrix_fov', 9)
    d = matrix(text, 'distortion_coefficients', 8)
    calibration = dict(path=str(cal_path), sha256=digest(cal_path),
        image_wh=[int(re.search(r'image_width:\s*(\d+)', text)[1]),
                  int(re.search(r'image_height:\s*(\d+)', text)[1])],
        camera_model=re.search(r'camera_model:\s*(\S+)', text)[1],
        K=[k[i:i+3] for i in (0, 3, 6)],
        camera_matrix_fov=[fov_k[i:i+3] for i in (0, 3, 6)],
        distortion_coefficients=d,
        reported_board_reprojection_error=float(re.search(r'reprojection_error:\s*(\S+)', text)[1]),
        board_poses_are_video_extrinsics=False,
        fisheye8_formula_confirmed=False,
        reported_error_is_video_validation=False)
    report = dict(updated=datetime.now(timezone.utc).isoformat(), calibration=calibration,
        videos=[], training_started=False, physical_calibration_validated=False,
        pending=['Which video(s) share this calibration and lens settings?',
                 'Exact sensor-to-encoded crop/resize/rotation and stabilization transform.',
                 'FISHEYE_8 projection definition and whether exported video is already rectified.'])
    for video in sorted(SOURCE.glob('*.mp4')):
        before = (video.stat().st_size, video.stat().st_mtime_ns)
        folder = OUT / video.stem
        folder.mkdir(exist_ok=True)
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams',
            '-show_format', '-of', 'json', str(video)], encoding='utf-8'))
        (folder / 'ffprobe.json').write_text(json.dumps(probe, indent=2, ensure_ascii=False), encoding='utf-8')
        stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        duration = float(stream['duration'])
        with (folder / '完整解码检查.log').open('w', encoding='utf-8') as log:
            completed = subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-v', 'error',
                '-xerror', '-err_detect', 'explode', '-noautorotate', '-i', str(video),
                '-map', '0:v:0', '-an', '-f', 'null', '-'], stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f'Video decode failed: {video}; inspect preserved log')
        frames = []
        for index, fraction in enumerate((0.05, 0.25, 0.50, 0.75, 0.95), 1):
            seconds = duration * fraction
            filename = f'{index:02d}_原始画面_{seconds:06.2f}秒.jpg'
            image = folder / filename
            if not image.exists():
                subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-v', 'error', '-n',
                    '-ss', str(seconds), '-noautorotate', '-i', str(video), '-map', '0:v:0',
                    '-frames:v', '1', '-vf', "scale=w='min(iw,960)':h='min(ih,960)':force_original_aspect_ratio=decrease",
                    '-q:v', '2', str(image)], check=True)
            frames.append(dict(path=str(image), requested_time_seconds=seconds,
                purpose='Aspect-preserved preview only; not training pixels or calibrated projection.'))
        row = dict(path=str(video), size=before[0], sha256=digest(video),
            encoded_wh=[stream['width'], stream['height']], duration_seconds=duration,
            frames=int(stream['nb_frames']), pixel_format=stream['pix_fmt'],
            color_space=stream.get('color_space'), color_transfer=stream.get('color_transfer'),
            rotation_tag=stream.get('tags', {}).get('rotate'),
            side_data=stream.get('side_data_list', []),
            full_decode_passed=True, calibration_mapping_validated=False, previews=frames,
            hover_metadata_present=any(s.get('codec_tag_string') == 'hovr' for s in probe['streams']))
        if before != (video.stat().st_size, video.stat().st_mtime_ns):
            raise RuntimeError(f'Source changed during intake: {video}; sync is not stable')
        report['videos'].append(row)
        (OUT / 'intake.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        print(json.dumps({key:row[key] for key in ('path', 'encoded_wh', 'frames', 'full_decode_passed')}, ensure_ascii=False), flush=True)
    print(OUT / 'intake.json', flush=True)


if __name__ == '__main__':
    main()
