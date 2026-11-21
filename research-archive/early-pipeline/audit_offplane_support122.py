"""N9: inspect existing off-plane material support, never refit or select by residual."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,time,sys
import numpy as np
from PIL import Image,ImageDraw
T=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab\outputs\mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';B=R/'video-update-20261027/background-geometry'
P=B/'offplane-support-plan.json';O=B/'offplane-support'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def register():
    assert not P.exists() and not O.exists()
    support=read(B/'residual-structure/fit-point-support.json')
    cut=float(np.quantile([x['plane_distance'] for x in support],.9))
    candidates=[x for x in support if x['plane_distance']>=cut]
    selected=[]
    for family in sorted(set(x['family'] for x in candidates)):
        selected.append(min((x for x in candidates if x['family']==family),key=lambda x:x['track']))
    inputs=[B/'observations.json',B/'residual-structure/fit-point-support.json',B/'shared-path/solver-repair/result.json',Path(__file__).parent/'background_geometry122.py']
    inputs += sorted(B.glob('frame_*.png'))
    save(P,dict(stage='N9',registered=now(),source_sha256=sha(Path(__file__)),
        hypothesis='Small ground residuals may coexist with poor off-plane pose constraints. Inspect whether the saved off-plane support represents static materials at useful depth, and whether the broad central exclusion discards nearby background.',
        actions=['Inspect all53 existing FIT families represented in top10percent absolutePCAplane distance; choose lowesttrackID per family, first/middle/last actual observations.',
            'Summarize depth/parallax andduplicatefamily counts using saved results; no residual-based point deletion.',
            'Show originalframe278 with registered excluded rectangle outline as evidence of detector coverage. The outline is presentation only.'],
        selected_tracks=[x['track'] for x in selected],selection_threshold_baseline_units=cut,
        sampled_track_count=len(candidates),sampled_families=len(selected),
        limitations='Off-plane classification is conditional onN5cK/poses, notphysicalgroundtruth. Texture appearance alone doesnot guarantee staticness. The sample doesnot certifyall935FITpoints.',
        prohibited=['No new feature extraction','No K/pose/point optimization','No validation deletion orsplit reassignment','No training'],
        budget=dict(runs=1,CPU_seconds=120,GPU=0),
        input_hashes={str(p):sha(p) for p in inputs}))
    print('N9_REGISTERED')
def run():
    assert not O.exists();p=read(P);assert p['source_sha256']==sha(Path(__file__))
    for f,h in p['input_hashes'].items():assert sha(Path(f))==h
    start=time.monotonic();O.mkdir();save(O/'progress.json',dict(state='running_readonly',pid=None,updated=now()))
    tracks=read(B/'observations.json')['tracks'];lookup={t['track']:t for t in tracks};support={t['track']:t for t in read(B/'residual-structure/fit-point-support.json')}
    frames={f:Image.open(B/f'frame_{f:06d}.png').convert('RGB') for f in [230,246,262,278,294,310,326]}
    pages=[];selected=p['selected_tracks']
    for page in range((len(selected)+5)//6):
        ids=selected[page*6:(page+1)*6];sheet=Image.new('RGB',(480,160*len(ids)),(235,235,235));indices=[]
        for row,tid in enumerate(ids):
            tr=lookup[tid];obs=sorted(tr['observations'],key=lambda o:o['frame']);chosen=[obs[0],obs[len(obs)//2],obs[-1]]
            for col,ob in enumerate(chosen):
                u,v=ob['uv'];x,y=round(u)-80,round(v)-80;crop=frames[ob['frame']].crop((x,y,x+160,y+160));d=ImageDraw.Draw(crop);d.ellipse((u-x-3,v-y-3,u-x+3,v-y+3),outline=(0,255,80));sheet.paste(crop,(col*160,row*160))
            indices.append(dict(row=row,track=tid,family=tr['family'],geometry=support[tid],observations=chosen))
        file=f'offplane_families_{page:02d}.png';sheet.save(O/file);pages.append(dict(file=file,rows=indices))
    im=frames[278].copy();d=ImageDraw.Draw(im);d.rectangle((350,900,1100,2400),outline=(255,180,0),width=3);im.save(O/'frame278_detector_exclusion.png')
    def stats(v):return dict(min=float(min(v)),median=float(np.median(v)),max=float(max(v)))
    save(O/'result.json',dict(stage='N9',updated=now(),state='prepared_pending_material_review',
        original_FIT_tracks=sum(t['split']=='fit' for t in tracks),original_FIT_families=len(set(t['family'] for t in tracks if t['split']=='fit')),
        selected_families=len(selected),selected_depth=stats([support[i]['median_camera_depth'] for i in selected]),
        selected_max_parallax_degrees=stats([support[i]['max_parallax_deg'] for i in selected]),
        source_excluded_rectangle=[350,900,1100,2400],cameras_changed=False,training=False,
        input_hashes=p['input_hashes'],elapsed_seconds=time.monotonic()-start))
    save(O/'patch-index.json',pages)
    assert time.monotonic()-start<120
    for f,h in p['input_hashes'].items():assert sha(Path(f))==h
    save(O/'progress.json',dict(state='completed_pending_review',pid=None,updated=now()))
    print(json.dumps(read(O/'result.json')))
if __name__=='__main__':globals()[sys.argv[1]]()
