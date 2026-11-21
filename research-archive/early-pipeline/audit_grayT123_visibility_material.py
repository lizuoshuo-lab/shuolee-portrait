"""One read-only native-pixel visibility and quantization audit, no inference."""
from pathlib import Path
from datetime import datetime,timezone
import time,json
import numpy as np
from scipy.ndimage import distance_transform_edt,map_coordinates
from PIL import Image,ImageDraw
from audit_grayT123_camera_support import REC,ROOT,read,save,sha,native,stats

OUT=REC/'grayT-visibility-material-review'
PROJECT=ROOT.parent/'runtime'
PYLIB=ROOT.parent.parent/'shuolee-portrait-clean-001/env/lib/python3.10/site-packages/rtmlib'
def main():
    started=time.monotonic()
    if (OUT/'result.json').exists():raise RuntimeError('Audit already completed. No repeat.')
    OUT.mkdir(exist_ok=True)
    files=[REC/'grayT-proxy-visibility-material-plan.json',ROOT/'data/observations.npz',ROOT/'fit/observation-weights.npz',
        ROOT/'data/training-inputs.json',PROJECT/'src/shuolee_portrait/observations.py',PROJECT/'src/shuolee_portrait/fitting.py',
        PROJECT/'src/shuolee_portrait/observation_support.py',PYLIB/'tools/pose_estimation/rtmpose.py']
    hashes=[sha(p) for p in files];p=read(files[0]);assert p['source_frames']==list(range(40,81))
    # Supplementary read-only question was registered before the full41frame results.
    add=dict(created=datetime.now(timezone.utc).isoformat(),question='Does actual argmax/flip-mean decoding quantize the saved native nose observations?',
        cause='Existing RTMPose argmax /2, whole-body affine scale, then original+flipped coordinates averaged.',
        formula='Each individual prediction has step=scale/(2*model_input_size). The mean lattice has step=scale/(4*model_input_size), x origin=center_x, y origin=center_y-scale_y/2.',
        checks='Compare every one of the369 original nose observations to the analytic per-frame lattice, without fitting an origin or deleting residuals.',
        no_new_model_inference=True,no_geometry_or_observation_changes=True,
        limits='Lattice spacing is a discretization property, not a measured accuracy bound or explanation for all large reconstruction errors.')
    if not (OUT/'decoding-audit-addendum.json').exists():save(OUT/'decoding-audit-addendum.json',add)
    o=np.load(files[1]);w=np.load(files[2]);rows=read(files[3])['records'];skin=read(REC/'grayT-skin-localization-review/result.json')['observations']
    skin={x['frame_number']:x for x in skin};records=[];frame_stats=[];images=[];headtiles=[];nosetiles=[];inputs=[]
    for frame in p['source_frames']:
        i=frame-1;row=rows[i];assert row['frame']==str(o['frame_ids'][i]);assert row['official_prediction_slot']==i
        assert o['joint_names'][50:59].tolist()==[f'face-{j}' for j in range(27,36)]
        uv=o['keypoints'][i,50:59];box=o['boxes'][i];assert np.max(np.abs(box-row['initial_box_xyxy']))==0
        center=(box[:2]+box[2:])*.5;sc=(box[2:]-box[:2])*1.25;aspect=.75
        scale=np.array([sc[0],sc[0]/aspect]) if sc[0]>sc[1]*aspect else np.array([sc[1]*aspect,sc[1]])
        step=scale/(4*np.array([288,384]));origin=np.array([center[0],center[1]-scale[1]/2]);binid=(uv-origin)/step
        lattice_residual=(binid-np.rint(binid))*step
        image=native(row['image']);mask=native(row['mask']);assert sha(image)==row['source_sha256'] and sha(mask)==row['mask_sha256']
        inputs.append(dict(frame=frame,image=str(image),image_sha256=sha(image),mask=str(mask),mask_sha256=sha(mask)))
        with Image.open(image) as im:rgb=im.convert('RGB')
        with Image.open(mask) as im:binary=np.array(im.convert('L'))>127
        outside=distance_transform_edt(~binary);dist=map_coordinates(outside,[uv[:,1],uv[:,0]],order=1,mode='constant',cval=1e6)
        near=binary[np.rint(uv[:,1]).astype(int),np.rint(uv[:,0]).astype(int)]
        for j in range(9):
            records.append(dict(frame=row['frame'],frame_number=frame,IBUG=j+27,RTMW_slot=j+50,xy=uv[j].tolist(),
                detector_score=float(o['scores'][i,j+50]),flip_disagreement_px=float(o['flip_disagreement'][i,j+50]),
                original_fit_weight=float(w['face'][i,j+10]),saved_surface_visibility=float(w['face_surface_visibility'][i,j+10]),
                person_mask_nearest_foreground=bool(near[j]),outside_person_mask_distance_px=float(dist[j]),
                legacy4px_support=bool(dist[j]<=4),decode_average_step_xy=step.tolist(),
                distance_to_analytic_decode_lattice_xy=lattice_residual[j].tolist(),
                image_visibility='pending_actual_native_review',physical_identity_certified=False))
        facebox=np.array(row['face_box_xyxy']);face_model_pixels=(facebox[2:]-facebox[:2])/scale*np.array([288,384])
        frame_stats.append(dict(frame_number=frame,body_crop_scale_xy=scale.tolist(),individual_decode_step_xy=(step*2).tolist(),
            flip_mean_lattice_step_xy=step.tolist(),face_ROI_on_model_xy=face_model_pixels.tolist(),
            analytic_lattice_max_abs_px=float(np.abs(lattice_residual).max()),positive_fit_nose_count=int((w['face'][i,10:19]>0).sum())))
        # These are native crops, not image warps. The point overlay is separate from the raw view.
        cx,cy=map(int,np.rint(uv.mean(0)));crop=[cx-80,cy-80,cx+80,cy+80]
        raw=rgb.crop(crop);overlay=raw.copy();draw=ImageDraw.Draw(overlay)
        for j,xy in enumerate(uv):
            x,y=xy-crop[:2];color=(0,240,255) if near[j] else (255,160,0)
            draw.ellipse((x-2,y-2,x+2,y+2),outline=color,width=1)
        tile=Image.new('RGB',(320,160));tile.paste(raw,(0,0));tile.paste(overlay,(160,0));nosetiles.append(tile)
        name=f'frame_{frame:03d}_nose_native_mask_support.png';tile.save(OUT/name)
        images.append(dict(frame=row['frame'],region='原生鼻部：原图／原点；青为mask内，橙为mask外；mask不等于解剖可见性',file=name,
            columns=['native160px','原9点与原mask位置'],crop=crop))
        sx,sy=skin[frame]['xy'];hc=[int(round(sx))-192,int(round(sy+40))-192,int(round(sx))+192,int(round(sy+40))+192]
        ht=rgb.crop(hc);headtiles.append(ht);ht.save(OUT/f'frame_{frame:03d}_head_native.png')
    sheets=[]
    for start in range(0,41,6):
        nc=Image.new('RGB',(960,320),(245,245,245));hc=Image.new('RGB',(1152,768),(245,245,245))
        for j in range(min(6,41-start)):
            nc.paste(nosetiles[start+j],((j%3)*320,(j//3)*160));hc.paste(headtiles[start+j],((j%3)*384,(j//3)*384))
        a=f'nose_sheet_{start//6+1:02d}.png';b=f'head_sheet_{start//6+1:02d}.png';nc.save(OUT/a);hc.save(OUT/b)
        sheets.append(dict(frames=p['source_frames'][start:start+6],nose=a,head=b))
    assert all(sha(f)==h for f,h in zip(files,hashes))
    out=[q for q in records if not q['person_mask_nearest_foreground']]
    result=dict(created=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-started,observations=records,frames=frame_stats,
        summary=dict(total=369,positive_weight=369,mask_outside_count=len(out),outside_with_positive_weight=sum(q['original_fit_weight']>0 for q in out),
            maximum_distance_from_mask_px=max(q['outside_person_mask_distance_px'] for q in records),
            mean_grid_spacing_px=stats([f['flip_mean_lattice_step_xy'][0] for f in frame_stats]),
            analytic_lattice_max_abs_px=max(f['analytic_lattice_max_abs_px'] for f in frame_stats),
            face_model_height_px=stats([f['face_ROI_on_model_xy'][1] for f in frame_stats])),
        sources=[dict(path=str(f),sha256=h) for f,h in zip(files,hashes)],inputs=inputs,sheets=sheets,
        no_new_inference_or_pose_fit=True,no_observation_reweighted=True,no_camera_change=True,quality_accepted=False)
    save(OUT/'index.json',dict(images=images,image_labels_burned_in=False));save(OUT/'result.json',result)
    print(json.dumps(result['summary'],ensure_ascii=False))
    print(json.dumps([dict(frame=q['frame_number'],IBUG=q['IBUG'],outside_px=q['outside_person_mask_distance_px'],weight=q['original_fit_weight']) for q in out],ensure_ascii=False))

if __name__=='__main__':main()
