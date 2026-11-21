"""Bounded existing-FAN observation adapter control; original94 inputs stay frozen."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import fcntl
import json
import os
import sys
import time
import traceback
import numpy as np

T=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REF=T/'shuolee/current/local-clothed-surface-123/smplifyx-reference-control'
F=REF/'official-first-cycle/full94-shared-shape'
O=REF/'official-first-cycle/local-face-input-control'
DEPS=Path('/opt/gs-slam-lab/smpl-x-alignment-20261029/face-diagnostic-deps')
MODEL=Path('/opt/gs-slam-lab/smpl-x-alignment-20261029/face-diagnostic-cache/hub/checkpoints/3DFAN4-4a694010b9.zip')
MODEL_SHA='4a694010b93140609b64a6ea734afc0ee80a4798394740281b0c5bea0ebef1d9'
FRAMES=['frame_000032','frame_000063','frame_000094']
POINTS=[30,33,36,39,42,45,48,54]

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):
    import hashlib
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4194304),b''):h.update(chunk)
    return h.hexdigest()
def now():return datetime.now(timezone.utc).isoformat()

def register():
    assert not O.exists();O.mkdir()
    assert sha(MODEL)==MODEL_SHA
    sources=[F/'data/training-inputs.json',F/'fit/fitted.npz',MODEL,DEPS/'face_alignment/utils.py',DEPS/'face_alignment/api.py',Path(__file__)]
    plan=dict(created=now(),state='registered',previous_goal_turn='progress: native1520 trial delivered; official shared-shape update rejected by front protection, actual94 foot vertex/projection binding verified.',
        human_scope='SMPLify-X拟合要能看的过去；要增大像素，其他你决策。Minimal input adapter control; no new sequence solver.',
        cause='Wholebody network face sampling is small despite native video pixels. Existing local face network changes model/context, not camera or SMPLify-X objective.',
        frames=FRAMES,negative_control='frame_000094 is hair-occluded; high heatmap responses do not establish visibility.',
        input='Frozen original native RGB and face ROI; fixed official FAN crop and heatmap decoder. Continuous inverse avoids the known native int32 truncation. No warp or ROI sweep.',
        model=dict(path=str(MODEL),sha256=MODEL_SHA,source='https://github.com/1adrianb/face-alignment',new_downloads=0),
        fitting_variable='Only native face51 coordinates may later change. Body/feet/hands/confidence/K/shared beta/official five stages/priors/optimizer stay frozen. No automatic fit before actual observation review.',
        inference_budget=dict(images=3,network_forwards=6,max_seconds=120,GPU_tasks=1),
        conditional_fit_budget=dict(frames=['frame_000032','frame_000063'],maximum_total_seconds=180,fresh_fits=2,gaussian_steps=0),
        acceptance='Actual source overlays first; visible eyes/nose/mouth must be plausible, with no invented occluded observations treated as verified. Compare fits against original native images and unchanged non-face observations. No residual-selected dropping.',
        stop='One observation and at most one fit per preregistered visible frame. Stop on source mismatch, implausible observations, or runtime budget; no threshold/weight/crop sweep.',
        source_hashes={str(p):sha(p) for p in sources})
    save(O/'plan.json',plan);print(json.dumps(dict(plan=str(O/'plan.json'))))

def observe():
    plan=read(O/'plan.json');assert not (O/'observation-progress.json').exists()
    for p,h in plan['source_hashes'].items():assert sha(p)==h,p
    start=time.monotonic();save(O/'observation-progress.json',dict(state='loading',pid=os.getpid()))
    try:
        sys.path.insert(0,str(DEPS))
        import torch
        from PIL import Image,ImageDraw
        from face_alignment.utils import crop,flip,get_preds_fromhm
        from face_alignment.detection.folder import FaceDetector
        assert FaceDetector(device='cuda',verbose=False).reference_scale==195
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        with np.load(F/'fit/fitted.npz',allow_pickle=False) as z:
            ids=list(z['frame_ids'].astype(str));targets=z['observed_native_keypoints'].copy()
        rows=read(F/'data/training-inputs.json')['records']
        images=[];records=[];heatmaps=[];fan_all=[];previous=[]
        with open('/tmp/shuolee-native122-gpu.lock','a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            model=torch.jit.load(str(MODEL),map_location='cuda').eval()
            with torch.inference_mode():
                for frame in FRAMES:
                    assert time.monotonic()-start<120
                    slot=ids.index(frame);r=rows[slot];assert r['frame']==frame and sha(r['image'])==r['source_sha256']
                    raw=Image.open(r['image']).convert('RGB');rgb=np.asarray(raw);box=np.array(r['face_box_xyxy'],dtype=np.float64)
                    center=torch.tensor([(box[0]+box[2])/2,(box[1]+box[3])/2],dtype=torch.float64)
                    center[1]-=(box[3]-box[1])*.12;scale=float((box[2]-box[0]+box[3]-box[1])/195.)
                    inp=crop(rgb,center,scale);assert inp.shape==(256,256,3)
                    tensor=torch.from_numpy(inp.transpose(2,0,1).copy()).to(device='cuda',dtype=torch.float32)[None]/255.
                    begin=time.monotonic();hm=model(tensor).detach();hm+=flip(model(flip(tensor)).detach(),is_label=True)
                    hm=hm.cpu().numpy();seconds=time.monotonic()-begin
                    points,integer,scores=get_preds_fromhm(hm.copy(),center.numpy(),scale)
                    continuous=(points[0].astype(np.float64)/64.-.5)*(200.*scale)+center.numpy()[None]
                    assert np.isfinite(continuous).all() and np.array_equal(np.trunc(continuous).astype(np.int32),integer[0])
                    old=targets[slot,67:118,:2];disp=np.linalg.norm(continuous[17:]-old,axis=1)
                    mid=(box[:2]+box[2:])/2;xy=np.floor(mid-96).astype(int);cb=[*xy,*list(xy+192)]
                    panels=[]
                    for variant,pts in [('raw',None),('rtmw',old),('fan',continuous[17:])]:
                        panel=raw.crop(cb)
                        if pts is not None:
                            draw=ImageDraw.Draw(panel)
                            for ibug in POINTS:
                                x,y=pts[ibug-17]-xy
                                draw.ellipse((x-1,y-1,x+1,y+1),outline='#00bb4a' if variant=='rtmw' else '#ad31d6',width=1)
                        panels.append(panel)
                    sheet=Image.new('RGB',(576,192));[sheet.paste(p,(192*j,0)) for j,p in enumerate(panels)]
                    name=f'20261030_局部人脸观测对照_{frame}_原生像素_左原图_中RTMW绿_右FAN紫.png'
                    rec=Path('/mnt/d/00/记录/005/特征点')/name;assert not rec.exists();sheet.save(rec);sheet.save(O/f'{frame}-face.png')
                    point_sheet=Image.new('RGB',(144*4,48*2))
                    for i,ibug in enumerate(POINTS):
                        anchor=old[ibug-17];corner=np.floor(anchor-24).astype(int);bounds=[*corner,*list(corner+48)]
                        for j,pts in enumerate([None,old,continuous[17:]]):
                            panel=raw.crop(bounds)
                            if pts is not None:
                                x,y=pts[ibug-17]-corner;ImageDraw.Draw(panel).ellipse((x-1,y-1,x+1,y+1),outline='#00bb4a' if j==1 else '#ad31d6',width=1)
                            point_sheet.paste(panel,((i%4)*144+j*48,(i//4)*48))
                    point_sheet.save(O/f'{frame}-points.png')
                    point_rec=Path('/mnt/d/00/记录/005/分析')/name.replace('局部人脸观测对照','局部人脸八点对照');assert not point_rec.exists();point_sheet.save(point_rec)
                    images.append(dict(frame_id=frame,context=str(rec),eight_points=str(point_rec),context_sha256=sha(rec),points_sha256=sha(point_rec),crop_xyxy=cb,no_resize=True,no_text=True))
                    records.append(dict(frame_id=frame,source_frame=r['source_frame_index'],source_sha256=r['source_sha256'],face_box=box.tolist(),
                        center=center.numpy().tolist(),scale=scale,network_input_wh=[256,256],input_crop_extent_native=200.*scale,
                        native_inverse_integer_parity=True,face51_displacement_px=dict(median=float(np.median(disp)),maximum=float(disp.max())),
                        scores_min=float(scores.min()),scores_max=float(scores.max()),inference_seconds=seconds,physical_visibility='not_established',quality_accepted=False))
                    heatmaps.append(hm);fan_all.append(continuous);previous.append(old)
                    save(O/'observation-progress.json',dict(state='observing',pid=os.getpid(),done=len(records),total=3))
        np.savez_compressed(O/'observations.npz',frame_ids=np.array(FRAMES),fan68_native=fan_all,rtmw51_native=previous,heatmaps=heatmaps)
        assert all(sha(p)==h for p,h in plan['source_hashes'].items())
        save(O/'observation-result.json',dict(updated=now(),state='complete_actual_review_pending',records=records,images=images,
             actual_visual_review=False,observations_sha256=sha(O/'observations.npz'),plan_sha256=sha(O/'plan.json'),new_downloads=0,
             native_images_unchanged=True,official_solver_unchanged=True,fit_updates=0,gaussian_steps=0,seconds=time.monotonic()-start))
        save(O/'observation-progress.json',dict(state='complete',pid=None));print(json.dumps(dict(output=str(O),rows=records),ensure_ascii=False))
    except BaseException as e:
        save(O/'observation-failure.json',dict(error=repr(e),traceback=traceback.format_exc()))
        save(O/'observation-progress.json',dict(state='failed',pid=None));raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['register','observe']);a=parser.parse_args()
    register() if a.action=='register' else observe()
