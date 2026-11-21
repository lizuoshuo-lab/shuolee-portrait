"""Read-only review of actual 074 reconstruction tracks, not tentative edges."""
from pathlib import Path
import collections
import datetime as dt
import hashlib
import json
import struct
import numpy as np
from PIL import Image, ImageDraw, ImageFont

B = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
S = B/'current/full-scene-localization-074'
O = B/'current/portrait-alignment-open-source-116/pixsfm'
LABELS = ['background', 'face_lower', 'face_upper', 'torso', 'legs_shoes', 'other_foreground']

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()

def load_model():
    images = {}
    with (S/'refined_sparse/images.bin').open('rb') as f:
        for _ in range(struct.unpack('<Q', f.read(8))[0]):
            record = struct.unpack('<i7di', f.read(64)); iid = record[0]
            name = bytearray()
            while True:
                ch = f.read(1)
                if ch == b'\0': break
                assert ch, 'Truncated image name'
                name.extend(ch)
            n = struct.unpack('<Q', f.read(8))[0]
            p = np.frombuffer(f.read(n*24), dtype=[('xy','<f8',(2,)),('pid','<i8')]).copy()
            images[iid] = dict(name=name.decode(), points=p, raw_frame=int(Path(name.decode()).stem.split('_')[-1]))
    points = {}
    with (S/'refined_sparse/points3D.bin').open('rb') as f:
        for _ in range(struct.unpack('<Q', f.read(8))[0]):
            rec = struct.unpack('<Q3d3Bd', f.read(43)); n = struct.unpack('<Q', f.read(8))[0]
            tr = np.frombuffer(f.read(n*8), dtype='<i4').reshape(n,2).copy()
            points[rec[0]] = dict(xyz=np.array(rec[1:4]), error=rec[-1], track=tr)
    return images, points

def main():
    target = O/'retained-track-review.json'
    assert not target.exists() or json.loads(target.read_text(encoding='utf8'))['visual_review']=='pending', 'Do not overwrite reviewed evidence.'
    hashes = {p.name:sha(p) for p in (S/'refined_sparse').glob('*.bin')}
    images, points = load_model()
    with np.load(O/'database-regions.npz') as z: regions = {int(k):z[k] for k in z.files}
    pool = collections.defaultdict(list); mixed = []
    for pid,p in points.items():
        tr=p['track']; group,n=collections.Counter(int(regions[i][j]) for i,j in tr).most_common(1)[0]
        fs=[images[i]['raw_frame'] for i,_ in tr]
        p['raw_span']=max(fs)-min(fs);p['region']=group;p['region_fraction']=n/len(tr)
        if len(tr)>=5 and n/len(tr)>=.8: pool[group].append(pid)
        if len(tr)>=5 and any(regions[i][j] in (1,2) for i,j in tr) and n/len(tr)<.8: mixed.append(pid)
    # Choose coverage extremes, not the largest/smallest residual. Keep full observations in JSON.
    selections = []
    for group in [1,2,3,4,0]:
        ordered=sorted(pool[group], key=lambda pid:(-points[pid]['raw_span'],-len(points[pid]['track']),pid))
        for pid in ordered[:2]: selections.append((LABELS[group],pid,'two largest acquisition spans per region; no residual ranking'))
    for pid in sorted(mixed,key=lambda pid:(-points[pid]['raw_span'],pid))[:2]:
        selections.append(('mixed_regions',pid,'two largest-span actual reconstructed face-cross-region tracks'))
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
    small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
    results=[]; sheets=[]
    for category,pid,rule in selections:
        p=points[pid];tr=sorted(p['track'].tolist(),key=lambda a:images[a[0]]['raw_frame'])
        obs=[]
        for iid,j in tr:
            assert images[iid]['points']['pid'][j]==pid
            obs.append(dict(image_id=iid,raw_frame=images[iid]['raw_frame'],keypoint_index=j,
                            xy_colmap=images[iid]['points']['xy'][j].tolist(),region=LABELS[int(regions[iid][j])]))
        sample=np.unique(np.round(np.linspace(0,len(obs)-1,6)).astype(int))
        board=Image.new('RGB',(1440,330),'white');draw=ImageDraw.Draw(board)
        views=len(set(ob['raw_frame'] for ob in obs))
        draw.text((8,5),f'074 point {pid} | {category} | {len(obs)} observations / {views} views | span {p["raw_span"]}',font=font,fill='black')
        for col,idx in enumerate(sample):
            ob=obs[idx];raw=np.array(ob['xy_colmap'])-.5
            box=np.round(np.r_[raw-70,raw+70]).astype(int)
            with Image.open(S/'images'/images[ob['image_id']]['name']) as im:
                patch=im.convert('RGB').crop(tuple(box)).resize((224,224),Image.Resampling.LANCZOS)
            board.paste(patch,(col*240+8,55))
            x,y=(raw-box[:2])*1.6+[col*240+8,55]
            draw.ellipse((x-6,y-6,x+6,y+6),outline='#00aeff',width=2)
            draw.line((x-10,y,x+10,y),fill='#00aeff',width=1)
            draw.text((col*240+8,32),f'raw {ob["raw_frame"]:06d}',font=small,fill='black')
            draw.text((col*240+8,285),ob['region'],font=small,fill='black')
        draw.text((8,310),'140 x 140 original-pixel crops; identical 1.6x scale; no image warp. Circle is the retained SIFT observation.',font=small,fill='#444444')
        sheets.append(board)
        results.append(dict(point3D_id=pid,category=category,selection=rule,unique_views=views,original_reprojection_error=p['error'],observations=obs,display_observation_indices=sample.tolist()))
    files=[]
    for start in range(0,len(sheets),4):
        subset=sheets[start:start+4];board=Image.new('RGB',(1440,330*len(subset)),'white')
        for j,im in enumerate(subset):board.paste(im,(0,j*330))
        name=f'retained-tracks-{start//4+1}.jpg';board.save(O/name,quality=95);files.append(name)
    assert hashes=={p.name:sha(p) for p in (S/'refined_sparse').glob('*.bin')}
    duplicate_groups=[]
    for pid,p in points.items():
        by_image=collections.defaultdict(list)
        for iid,j in p['track']:by_image[int(iid)].append(int(j))
        for iid,indices in by_image.items():
            if len(indices)<2:continue
            xy=images[iid]['points']['xy'][indices]
            duplicate_groups.append(dict(point3D_id=int(pid),image_id=iid,count=len(indices),keypoint_indices=indices,
              span_bbox_diagonal_raw_px=float(np.linalg.norm(xy.max(0)-xy.min(0)))))
    spans=np.array([r['span_bbox_diagonal_raw_px'] for r in duplicate_groups])
    duplicate_summary=dict(points_with_repeated_image_id=len(set(r['point3D_id'] for r in duplicate_groups)),groups=len(duplicate_groups),
      median_span_raw_px=float(np.median(spans)),p90_span_raw_px=float(np.percentile(spans,90)),max_span_raw_px=float(spans.max()),
      groups_span_over={str(v):int((spans>v).sum()) for v in [1,2,5,10]},
      worst=sorted(duplicate_groups,key=lambda r:-r['span_bbox_diagonal_raw_px'])[:10],
      interpretation='Multiple distinct SIFT observations of one 3D point can occur in one image after track merging. Count unique views separately. This measurement does not by itself establish a software mapping bug or the ghosting cause.')
    data=dict(created=dt.datetime.now(dt.timezone.utc).isoformat(),source_hashes=hashes,same_image_duplicates=duplicate_summary,
      source_model_unchanged=True,total_points=len(points),total_frames=len(images),selected_tracks=results,figures=files,
      visual_review='pending',scope='Actual 074 reconstructed point identity. Region labels are approximate image regions, not physical material labels.',
      limitations=['A region-consistent or low-residual SfM track can still follow an occluding contour, reflection, or a moving material point.',
                   'A deterministic sample does not prove all retained tracks correct. No optimization, rejection, camera or Gaussian change is performed.'])
    target.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(selected=[(a,b) for a,b,_ in selections],figures=files,source_unchanged=True)))

if __name__=='__main__':main()
