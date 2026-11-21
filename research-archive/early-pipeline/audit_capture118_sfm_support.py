"""Read-only audit of actual 118 camera-fit support, separate from held metrics."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sqlite3
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import pycolmap

ROOT=Path('/opt/gs-slam-lab/shuolee-new-capture-118')
REC=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/new-capture-118')
OUT=ROOT/'alignment/camera-fit-support-audit'
OUT.mkdir(parents=True,exist_ok=False)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def summary(a):
    a=np.asarray(a,float)
    return dict(count=len(a),median=float(np.median(a)),p90=float(np.quantile(a,.9)),maximum=float(a.max())) if a.size else dict(count=0)
save(OUT/'plan.json',dict(created=datetime.now(timezone.utc).isoformat(),
    scope='Read actual COLMAP training observations; do not recompute SfM or filter frozen validation.',
    question='Did the confirmed door-to-light validation mismatch or its excluded feature rows enter camera fitting?',
    visual_sample='Up to six longest tracks per face/body region, stable point ID tie-break; no residual-based selection.',
    cameras_changed=False,held_tracks_changed=False,training_started=False))
split=read(ROOT/'sfm/holdout/split.json')
bad=next(t for t in split['tracks'] if t['track_id'].startswith('presfm_094e99a4'))
frames={r['frame']:r for r in split['frames']}
db=sqlite3.connect((ROOT/'sfm/database.db').resolve().as_uri()+'?mode=ro',uri=True)
bad_audit=[]
for o in bad['observations']:
    f=frames[o['frame']]
    with np.load(ROOT/'sfm/holdout'/f['row_map'],allow_pickle=False) as z:
        excluded=bool(o['feature_index'] in z['removed'])
        assert excluded and o['feature_index'] not in z['kept']
    n,c,blob=db.execute('SELECT rows,cols,data FROM keypoints WHERE image_id=?',(f['image_id'],)).fetchone()
    xy=np.frombuffer(blob,dtype='<f4').reshape(n,c)[:,:2]
    distance=float(cKDTree(xy).query(o['raw_colmap_xy'])[0])
    bad_audit.append(dict(frame=o['frame'],original_feature_index=o['feature_index'],excluded=excluded,
                          nearest_retained_keypoint_px=distance))
db.close()
rows=read(ROOT/'data/training-inputs.json')['records']
byname={r['name']:r for r in rows}
recon=pycolmap.Reconstruction(str(ROOT/'rectified/sparse'))
region_obs={'face':[],'body':[],'background':[],'unknown':[]}
tracks={}
per_frame=[]
for iid,im in recon.images.items():
    row=byname[im.name];E=np.asarray(row['w2c']);K=np.asarray(row['K'])
    mask=np.asarray(Image.open(row['mask']).convert('L'))>127
    face=row.get('face_box_xyxy');counts={r:0 for r in region_obs}
    for idx,p in enumerate(im.points2D):
        if not p.has_point3D():continue
        xyz=recon.points3D[p.point3D_id].xyz
        cam=E[:3,:3]@xyz+E[:3,3];q=K@cam
        observed=np.asarray(p.xy)-.5;predicted=q[:2]/q[2]-.5
        error=float(np.linalg.norm(observed-predicted))
        x,y=np.rint(observed).astype(int)
        region='unknown'
        if 0<=x<mask.shape[1] and 0<=y<mask.shape[0]:
            if not mask[y,x]:region='background'
            elif face is not None and face[0]+4<=x<=face[2]-4 and face[1]+4<=y<=face[3]-4:region='face'
            else:region='body'
        counts[region]+=1;region_obs[region].append(error)
        tracks.setdefault(int(p.point3D_id),[]).append(dict(frame=row['frame'],image=row['image'],xy=observed.tolist(),
            index=idx,region=region,residual_px=error,time_seconds=row['time_seconds']))
    per_frame.append(dict(frame=row['frame'],counts=counts))

selected=[]
for region in ['face','body']:
    pool=[(pid,sorted(obs,key=lambda v:v['time_seconds'])) for pid,obs in tracks.items()
          if len(obs)>=3 and all(v['region']==region for v in obs)]
    pool.sort(key=lambda p:(-len(p[1]),p[0]))
    for rank,(pid,obs) in enumerate(pool[:6],1):
        indices=np.unique(np.linspace(0,len(obs)-1,min(8,len(obs))).astype(int))
        tiles=[];chosen=[]
        for i in indices:
            o=obs[i];im=Image.open(o['image']).convert('RGB');x,y=np.rint(o['xy']).astype(int)
            tiles.append(im.crop((x-40,y-40,x+41,y+41)));chosen.append(o)
        sheet=Image.new('RGB',(81*len(tiles),81),(235,235,235))
        for i,tile in enumerate(tiles):sheet.paste(tile,(81*i,0))
        name=f'{region}_camera_fit_track_{rank:02d}_point{pid}_native_pixels.png'
        sheet.save(OUT/name)
        selected.append(dict(point_id=pid,region=region,observations=len(obs),
            span_seconds=obs[-1]['time_seconds']-obs[0]['time_seconds'],path=name,displayed=chosen))
result=dict(updated=datetime.now(timezone.utc).isoformat(),frames=len(rows),
    camera_manifest_sha256=sha(ROOT/'data/training-inputs.json'),
    source_split_sha256=sha(ROOT/'sfm/holdout/split.json'),
    identified_false_holdout_track=dict(track_id=bad['track_id'],observations=bad_audit,
        all_original_observations_excluded=all(x['excluded'] for x in bad_audit),
        same_material_alias_elsewhere_not_proven_absent=True),
    actual_camera_fit_residuals={k:summary(v) for k,v in region_obs.items()},
    per_frame=sorted(per_frame,key=lambda x:x['frame']),visual_samples=selected,
    camera_fit_points=len(tracks),cameras_changed=False,held_observations_removed=False,
    interpretation='In-sample fit statistics cannot establish physical identity or camera accuracy. Exact excluded validation observations do not directly cause the solved camera poses.',
    quality_accepted=False,training_admitted=False)
save(OUT/'result.json',result);save(REC/'camera-fit-support-audit.json',result)
print(json.dumps({k:v for k,v in result.items() if k in ['frames','actual_camera_fit_residuals','camera_fit_points','interpretation']}))
print(json.dumps(dict(false_holdout_observations_excluded=len(bad_audit),native_pixel_sheets=len(selected),output=str(OUT))))
