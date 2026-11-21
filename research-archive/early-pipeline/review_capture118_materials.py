"""Make native-pixel diagnostic contact sheets; never change observations or cameras."""
from pathlib import Path
import hashlib, json, math
from PIL import Image

ROOT = Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/shuolee-new-capture-118/data/alignment-diagnostics')
OUT = Path('D:/00/工作日志/2026-10-26/新视频118/对应点核验')
OUT.mkdir(parents=True, exist_ok=True)
manifest = json.loads((ROOT/'material-review/manifest.json').read_text(encoding='utf-8'))
measured = json.loads((ROOT/'measurements.json').read_text(encoding='utf-8'))
frames = {f['frame']: f for f in manifest['frames']}
tracks = {t['track_id']: t for t in manifest['tracks']}
face = [t for t in measured['tracks'] if t['region'] == 'face']
bad = sorted([t for t in measured['tracks'] if t['status']=='measured' and t['region']=='static_scene_candidate'],
             key=lambda t: max(t['held_error_px']), reverse=True)[:5]
rows = []
for group, selected in [('face_all',face), ('background_outlier_diagnosis',bad)]:
    for n, t in enumerate(selected,1):
        obs = tracks[t['track_id']]['observations']
        tile, gap, columns = 128, 8, min(8,len(obs))
        sheet = Image.new('RGB',(columns*(tile+gap)-gap, math.ceil(len(obs)/columns)*(tile+gap)-gap),(220,220,220))
        for i,o in enumerate(obs):
            path = ROOT/'material-review'/frames[o['frame']]['asset']
            uv = o['uv']; x,y = [int(math.floor(v+0.5))-tile//2 for v in uv]
            with Image.open(path) as im:
                patch = im.convert('RGB').crop((x,y,x+tile,y+tile))
            sheet.paste(patch,((i%columns)*(tile+gap),(i//columns)*(tile+gap)))
        name = f'{group}_{n:02d}_{t["track_id"][7:15]}.png'
        sheet.save(OUT/name)
        rows.append(dict(file=name,track_id=t['track_id'],region=t['region'],frames=[o['frame'] for o in obs],
            original_uv=[o['uv'] for o in obs],columns=columns,tile_native_pixels=tile,point_at_tile_xy=[64,64],
            max_held_error_px=max(t['held_error_px']),source_only=True,image_resized=False,text_burned_in=False,
            selection='All face candidates' if group=='face_all' else 'Five largest background residuals for failure diagnosis only; no observation or metric removed'))
report=dict(source_manifest_sha256=hashlib.sha256((ROOT/'material-review/manifest.json').read_bytes()).hexdigest(),
    source_measurements_sha256=hashlib.sha256((ROOT/'measurements.json').read_bytes()).hexdigest(),sheets=rows,
    observation_membership_changed=False,cameras_changed=False,training_admitted=False)
(OUT/'index.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(sheets=len(rows),destination=str(OUT)),ensure_ascii=False))
