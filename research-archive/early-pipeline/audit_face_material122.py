"""C1a: audit frozen face-ROI material support; no fitting or residual selection."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, io, json, math, time
import numpy as np
from PIL import Image

R = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
OLD = R.parent / 'raw-domain-121'
ROOT = Path('/opt/gs-slam-lab/shuolee-raw-domain-121')
OUT = R / 'face-material-support'
SUBJECTS = ['grass', 'sofa', 'portrait', 'indoor']
PLAN = R / 'face-material-support-plan.json'

def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n')
    t.replace(p)

def inside(uv, b):
    return b is not None and b[0]-12 <= uv[0] <= b[2]+12 and b[1]-12 <= uv[1] <= b[3]+12

def inputs(s):
    p = ROOT/s/'roi-supplement/observations.json'
    a = OLD/f'supplement-{s}-track-family-audit.json'
    d, family = read(p), read(a)
    ids = set(family['retained_track_ids'])
    rois = {r['frame']:r['face_box'] for r in d['rois']}
    tracks = [t for t in d['tracks'] if t['track_id'] in ids and t['split']=='heldout'
              and any(inside(o['uv'], rois[o['frame']]) for o in t['observations'])]
    tracks.sort(key=lambda t:(min(o['frame'] for o in t['observations']), t['track_id']))
    return d, tracks, dict(observations_file=str(p), observations_sha256=sha(p),
                          family_audit=str(a), family_audit_sha256=sha(a))

def register():
    if PLAN.exists(): raise RuntimeError('Plan already registered; do not overwrite')
    subjects = []
    for s in SUBJECTS:
        d, tracks, files = inputs(s)
        subjects.append(dict(subject=s, **files, selected_track_ids=[t['track_id'] for t in tracks],
                             tracks=len(tracks), observations=sum(len(t['observations']) for t in tracks)))
    save(PLAN, dict(stage='C1a', created=now(), source_sha256=sha(Path(__file__)),
        question='Does a face ROI observation identify a stable facial material point, or hair/glasses/silhouette/background, and what anatomical/view coverage is actually present?',
        new_causal_evidence='Independent material-location review of existing frozen heldout face-ROI tracks, previously not anatomically audited in full. No new K/pose candidate.',
        selection='Every retained heldout family having at least one observation in the existing face_box+12px. Include every observation of each selected track, even outside ROI. Frozen IDs/splits unchanged. No residuals or fitted cameras are read.',
        subjects=subjects, patch_size=128, columns=6, max_page_height=1100,
        budget=dict(attempts=1, cpu_wall_seconds=120, GPU=0, feature_extraction=0, fitting=0, training=0),
        classification=['skin texture candidate', 'eye or eyelid feature', 'nose feature', 'mouth feature',
                        'glasses edge or reflection', 'hair/beard', 'hat/accessory', 'silhouette/occlusion',
                        'off-face/background', 'ambiguous material identity'],
        interpretation='Visual material localization can reject an unsupported semantic claim. It cannot establish subpixel 3D accuracy, static identity or measurement uncertainty by itself. Do not delete or reassign any observation from its split based on this review.',
        stop_conditions=['Input hash mismatch', 'More than120s preparation', 'Any geometry/heldout fitting or source mutation would be required'],
        output='Unscaled original-pixel patches, no text/dots/arrows/warps drawn on images; labels, UV and fractional crop center offsets only in JSON/HTML. All unreviewed items explicitly remain unreviewed.'))
    print(json.dumps({'registered':str(PLAN),'tracks':sum(s['tracks'] for s in subjects),'observations':sum(s['observations'] for s in subjects)}))

def run(repair=False):
    plan = read(PLAN)
    expected=read(R/'face-material-support-presentation-repair.json')['repaired_source_sha256'] if repair else plan['source_sha256']
    assert expected==sha(Path(__file__))
    if OUT.exists() and not repair: raise RuntimeError('One registered extraction only; output already exists')
    OUT.mkdir(exist_ok=repair)
    start=time.monotonic(); summaries=[]
    for spec in plan['subjects']:
        s=spec['subject']; d, tracks, files=inputs(s)
        assert all(files[k]==spec[k] for k in files)
        assert [t['track_id'] for t in tracks]==spec['selected_track_ids']
        completed=OUT/f'{s}-inventory.json'
        if repair and completed.exists():
            done=read(completed)
            assert done['source_files']==files
            assert [t['track_id'] for t in done['tracks_detail']]==spec['selected_track_ids']
            for page in done['pages']: assert sha(OUT/page['file'])==page['sha256']
            summaries.append({k:v for k,v in done.items() if k!='tracks_detail'})
            continue
        rows={r['frame']:r for r in d['records']}; rois={r['frame']:r['face_box'] for r in d['rois']}
        frames=sorted({o['frame'] for t in tracks for o in t['observations']})
        images={}; hashes={}
        for f in frames:
            p=Path(rows[f]['image']); assert sha(p)==rows[f]['source_sha256']
            images[f]=Image.open(p).convert('RGB'); hashes[f]=sha(p)
        receipts=[]; pages=[]; page_tracks=[]; page_rows=[]; total_height=0
        def flush():
            nonlocal page_tracks,page_rows,total_height
            if not page_rows:return
            name=f'{s}_face_material_page_{len(pages)+1:02d}.png'
            canvas=Image.new('RGB',(128*6,total_height),(242,242,242))
            y=0
            for a in page_rows: canvas.paste(a,(0,y)); y+=a.height+12
            buf=io.BytesIO();canvas.save(buf,format='PNG');data=buf.getvalue()
            if (OUT/name).exists(): assert (OUT/name).read_bytes()==data
            else: (OUT/name).write_bytes(data)
            pages.append(dict(file=name, tracks=page_tracks, height=total_height,sha256=sha(OUT/name)))
            page_tracks=[];page_rows=[];total_height=0
        for idx,t in enumerate(tracks,1):
            obs=sorted(t['observations'],key=lambda o:o['frame']); nrow=math.ceil(len(obs)/6)
            block=Image.new('RGB',(128*6,nrow*128),(242,242,242)); rec=[]
            for j,o in enumerate(obs):
                f=o['frame']; im=images[f]; u,v=o['uv']; x,y=int(round(u)),int(round(v))
                assert o['image_sha256']==hashes[f]
                box=(x-64,y-64,x+64,y+64)
                valid=(max(0,box[0]),max(0,box[1]),min(im.width,box[2]),min(im.height,box[3]))
                assert valid[0]<valid[2] and valid[1]<valid[3]
                patch=Image.new('RGB',(128,128),(242,242,242))
                patch.paste(im.crop(valid),(valid[0]-box[0],valid[1]-box[1]))
                block.paste(patch,((j%6)*128,(j//6)*128))
                rec.append(dict(frame=f,uv=o['uv'],crop_xyxy=list(box),local_uv=[u-box[0],v-box[1]],
                                valid_source_xyxy=list(valid),outside_image_padding_not_observed=(valid!=box),
                                inside_existing_face_roi=inside(o['uv'],rois[f]),source_sha256=hashes[f],
                                block_row=j//6,block_column=j%6))
            if total_height+block.height+12>1100:flush()
            page_tracks.append(dict(index=idx,track_id=t['track_id'],y0=total_height,rows=nrow,observations=len(obs)))
            page_rows.append(block);total_height+=block.height+12
            receipts.append(dict(index=idx,track_id=t['track_id'],split=t['split'],observations=rec,
                                 material_review='pending',physical_identity_verified=False))
        flush()
        assert all(files[k]==inputs(s)[2][k] for k in files)
        perframe={f:sum(o['frame']==f and inside(o['uv'],rois[f]) for t in tracks for o in t['observations']) for f in rows}
        summary=dict(subject=s, tracks=len(tracks),observations=sum(len(t['observations']) for t in tracks),
                     face_roi_observations=sum(perframe.values()),frames_with_face_roi_support=sum(n>0 for n in perframe.values()),
                     total_diagnostic_frames=len(rows),per_frame_face_roi_observations=perframe,pages=pages,source_files=files)
        save(OUT/f'{s}-inventory.json',dict(**summary,tracks_detail=receipts))
        summaries.append(summary)
        if time.monotonic()-start>120:raise RuntimeError('Preparation budget exceeded')
    result=dict(stage='C1a',state='prepared_for_visual_review',updated=now(),subjects=summaries,
                elapsed_seconds=time.monotonic()-start,physical_identity_verified=False,training_started=False,
                face_accuracy_passed=False,residual_selection=False,source_images_changed=False,
                fit_or_holdout_reassigned=False,images_resampled=False)
    save(OUT/'result.json',result)
    print(json.dumps({k:result[k] for k in ['stage','state','elapsed_seconds']}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--register',action='store_true');ap.add_argument('--resume-presentation-repair',action='store_true');a=ap.parse_args()
    register() if a.register else run(a.resume_presentation_repair)
