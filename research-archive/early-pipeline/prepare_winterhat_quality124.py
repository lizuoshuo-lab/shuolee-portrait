"""Apply the frozen quality-first rule to the existing complete winterhat fit."""
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from PIL import Image
import audit_gaussianavatars_params123 as a
from select_quality_frames124 import POLICY,select,self_test
from prepare_quality_eight124 import local

ROOT=a.REC/'cross-subject-sh0-124'
RUN=ROOT/'winterhat-temporal'
OUT=ROOT/'quality-first-eight124/winterhat'

def main():
    assert not a.lp(OUT/'shortlist.json').exists(), 'Read the existing selection, never repeat silently.'
    fitpath=RUN/'official-fit/fit/fitted.npz';digest=a.sha(fitpath)
    receipt=a.read(RUN/'official-fit/fit/official-fit-result.json');assert digest==receipt['fit_sha256']
    source=a.read(ROOT/'winterhat-input/selection.json')
    held=source['review_only_frames']
    a.save(OUT/'plan.json',dict(updated=datetime.now(timezone.utc).isoformat(),policy=POLICY,
        source_fit=str(fitpath),source_fit_sha256=digest,review_only_frames=held,
        selection_source=str(ROOT/'winterhat-input/selection.json'),source_selection_sha256=a.sha(ROOT/'winterhat-input/selection.json'),
        new_fit_calls=0,gaussian_started=False,parameters_changed=False,controls=self_test()))
    rows=a.read(RUN/'official-fit/data/training-inputs.json')['records'];byid={r['frame']:r for r in rows}
    cached={r['frame']:r for r in source['candidates']}
    ev=a.read(RUN/'final-review/result.json');evby={r['frame']:r for r in ev['records']}
    with np.load(a.lp(fitpath),allow_pickle=False) as z:
        ids=z['frame_ids'].astype(str).tolist();obs=z['observed_native_keypoints'];proj=z['joints_projected_native']
        rotations=z['w2c'][:,:3,:3]@z['root_rotation'];imgsha=z['image_sha256'].astype(str)
    chronological=np.argsort([byid[f]['time_seconds'] for f in ids],kind='stable')
    y0=np.arctan2(rotations[:,0,2],rotations[:,2,2]);yaw=np.empty(len(ids));yaw[chronological]=np.unwrap(y0[chronological])
    flags={f:[] for f in ids}
    for t in ev['transitions']:
        dt=byid[t['current']]['time_seconds']-byid[t['previous']]['time_seconds'];assert dt>0
        if t['max_local_body_angle_deg']/dt>POLICY['joint_speed_deg_per_second']:
            for f in (t['previous'],t['current']):flags[f].append('body_pose_jump_over_existing135deg_s')
    records=[];body=np.array([0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11])
    for i,f in enumerate(ids):
        row=byid[f];assert imgsha[i]==row['source_sha256']
        h=row['initial_box_xyxy'][3]-row['initial_box_xyxy'][1];assert h>0
        e=np.linalg.norm(proj[i]-obs[i,:,:2],axis=-1);m={};ff=list(flags[f])
        for name,slots in [('body',body),('face',np.arange(67,118)),('feet',np.arange(19,25))]:
            w=obs[i,slots,2]**2;assert w.sum()>0
            m[name+'_weighted_rms_px']=float(np.sqrt(np.sum(w*e[slots]**2)/w.sum()))
            maximum=float(e[slots][w>0].max());m[name+'_supported_max_over_height']=maximum/h
            if name in ('body','feet') and maximum/h>POLICY[name+'_max_over_height']:
                ff.append(name+'_reprojection_over_existing3percent_height')
        if f in held:ff.append('preregistered_Gaussian_review_only_never_train')
        records.append(dict(frame=f,time_seconds=row['time_seconds'],source_fit_sha256=digest,
            source_image_sha256=imgsha[i],yaw_proxy_deg=float(np.degrees(yaw[i])),
            face_sharpness=cached[f]['sharpness'],body_confidence=float(obs[i,body,2].mean()),screen_flags=ff,**m))
    result=select(records,admitted_only=False)
    result.update(updated=datetime.now(timezone.utc).isoformat(),source_fit=str(fitpath),source_fit_sha256=digest,
        actual_3d_pose_review=False,review_only_frames=held,policy_sha256=a.sha(OUT/'plan.json'))
    a.save(OUT/'shortlist.json',result)
    render_only()
    print(dict(frames=len(ids),shortlist=result['selected_frames'],missing_bins=result['missing_bins']))

def render_only():
    result=a.read(OUT/'shortlist.json')
    ev=a.read(RUN/'final-review/result.json');evby={r['frame']:r for r in ev['records']}
    # Reuse saved raw/mesh evidence. No mesh rasterizer or fitted parameters run.
    pages=[];tiles=[]
    for f in sorted(r['frame'] for r in result['records'] if r['eligible']):
        item=evby[f];im=Image.open(local(item['images']['body'])).convert('RGB')
        cols=3 if item['has_old_same_frame'] else 2;w=im.width//cols
        pair=Image.new('RGB',(2*w,im.height));pair.paste(im.crop((0,0,w,im.height)),(0,0))
        pair.paste(im.crop(((cols-1)*w,0,cols*w,im.height)),(w,0));pair.thumbnail((620,480),Image.Resampling.LANCZOS)
        chronological_index=int(f[-6:])-1
        saved=Image.open(local(ev['pages'][chronological_index//4])).convert('RGB')
        y=(chronological_index%4)*480
        tile=Image.new('RGB',(1300,480));tile.paste(pair,(0,0))
        # Older pages have no separate solid-front column; the two sides are
        # always the final680 pixels. Validate the actual saved layout.
        assert saved.width in (1130,1470) and y+480<=saved.height
        left=saved.width-680
        tile.paste(saved.crop((left,y,left+340,y+480)),(620,0));tile.paste(saved.crop((left+340,y,saved.width,y+480)),(960,0))
        tiles.append((f,tile))
    for start in range(0,len(tiles),4):
        chunk=tiles[start:start+4];page=Image.new('RGB',(1300,480*len(chunk)))
        for j,(_,im) in enumerate(chunk):page.paste(im,(0,j*480))
        p=Path('D:/00/记录/001')/f'分析_质量候选124_原图_同帧拟合_双侧面完整_页{start//4+1:02d}.png'
        assert not a.lp(p).exists();page.save(a.lp(p))
        pages.append(dict(path=str(p),sha256=a.sha(p),frames=[f for f,_ in chunk],overview_only=True))
    if a.lp(OUT/'review-pages.json').exists():
        a.save(OUT/'review-pages-incomplete.json',a.read(OUT/'review-pages.json'))
    a.save(OUT/'review-pages.json',pages)
    a.save(OUT/'layout-recovery.json',dict(original_layout_width=1130,initial_assumption=1470,
        original_fit_unchanged=True,shortlist_unchanged=True,new_fit_calls=0,
        initial_preview_invalid=True,original_preview_retained=True,corrected=True))
    print(dict(eligible=len(tiles),pages=len(pages)))

if __name__=='__main__':
    import sys
    render_only() if '--render-only' in sys.argv else main()
