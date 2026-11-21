"""One native five-view semantic observation audit, no cameras or real fitting.

Reuse the installed FAN model only. Keep the official crop and heatmap decoder;
record the continuous inverse alongside its existing integer-valued output.
The continuous output is not automatically a more accurate physical landmark.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,time,os,sys
import numpy as np

H=Path(__file__).resolve().parent
REC=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
PRE=REC/'field-preflight';OUT=PRE/'semantic-face-binding'
ROOT=Path('/opt/gs-slam-lab/shuolee-local-surface-123/preflight/field')
DEPS=Path('/opt/gs-slam-lab/smpl-x-alignment-20261029/face-diagnostic-deps')
MODEL=Path('/opt/gs-slam-lab/smpl-x-alignment-20261029/face-diagnostic-cache/hub/checkpoints/3DFAN4-4a694010b9.zip')
MODEL_SHA='4a694010b93140609b64a6ea734afc0ee80a4798394740281b0c5bea0ebef1d9'
SLOTS=[0,44,88,132,176]
POINTS=[30,33,36,39,42,45,48,54]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stamp():return datetime.now(timezone.utc).isoformat()

def register():
    assert not OUT.exists();OUT.mkdir()
    rows=read(PRE/'coverage/observations.json')['rows'];assert len(rows)==177
    assert sha(MODEL)==MODEL_SHA
    files=[MODEL,DEPS/'face_alignment/api.py',DEPS/'face_alignment/utils.py',
           PRE/'coverage/observations.json',PRE/'coverage/observations.npz',
           REC/'direct-smplx-bindings/bindings.json',Path(__file__)]
    files += [Path(rows[i]['image']) for i in SLOTS]
    save(OUT/'plan.json',dict(updated=stamp(),previous_goal_turn='progress: direct 2D-to-SMPL-X objective implemented and known-geometry optimization tested; real-person result remains unaccepted',
        objective='Verify native 2D anatomical locations and explicit SMPL-X bindings in real images before fitting.',
        samples=[dict(slot=i,source_frame=rows[i]['source_frame_index'],frame=rows[i]['frame']) for i in SLOTS],
        anatomical_indices=POINTS,all_original_pixels_and_observations_preserved=True,
        hypotheses=['A whole-body detector can leave insufficient head-region sampling for precise facial fitting.',
                    'The installed FAN native inverse casts coordinates to int32; the same heatmaps allow measuring that avoidable coordinate loss.',
                    'Semantic predictions on occluded facial surfaces are not observed landmarks.'],
        experiment=dict(fresh_inference_images=5,forwards_per_image=2,network='existing 3DFAN4 with original+flipped input',
            crop='official implementation and original frozen RTMW ROI; no crop/decoder sweep',
            variants=['Existing whole-body RTMW, unchanged','FAN official integer native output','Same FAN heatmaps and official quarter-offset decoder, continuous native inverse'],
            causal_scope='Integer vs continuous inverse is the isolated variable. RTMW vs FAN changes model/context and is only a descriptive comparison.',
            dtype='CPU float32 network, float64 algebra for continuous inverse',
            new_downloads=0,max_seconds=300,CPU_threads=4,GPU_tasks=0),
        review='5 native heads and 8 fixed landmarks per view, 40 native point panels. Native crops, no geometric warp or burned text. Visibility and semantic region only; measurement uncertainty unknown unless independently established.',
        stops=['Source SHA changes','Unexpected checkpoint download or network call','Nonfinite or unsupported crop','Official result differs from truncated continuous inverse','Budget exhausted'],
        no_camera_estimation=True,no_SMPLX_optimization=True,no_gaussian_training=True,
        no_independent_physical_truth=True,quality_accepted=False,
        sources={str(p):sha(p) for p in files}))
    print(json.dumps(dict(plan=str(OUT/'plan.json'),model_download=False)))

def run(repair_preinference=False, repair_folder_import=False):
    started=time.monotonic();plan=read(OUT/'plan.json')
    assert not (OUT/'result.json').exists()
    repair_mode=repair_preinference or repair_folder_import
    if repair_folder_import:
        repair=read(OUT/'folder-import-repair.json')
        assert repair['new_source_sha256']==sha(Path(__file__)) and repair['network_forwards_before_failure']==0
        assert (OUT/'repair-started.json').exists() and not (OUT/'folder-import-started.json').exists()
    elif repair_preinference:
        repair=read(OUT/'preinference-repair.json')
        assert repair['new_source_sha256']==sha(Path(__file__)) and repair['network_forwards_before_failure']==0
        assert (OUT/'started.json').exists() and not (OUT/'repair-started.json').exists()
    else:
        assert not (OUT/'started.json').exists()
    for p,h in plan['sources'].items():
        expected=repair['new_source_sha256'] if repair_mode and Path(p)==Path(__file__) else h
        assert sha(Path(p))==expected,p
    marker='folder-import-started.json' if repair_folder_import else ('repair-started.json' if repair_preinference else 'started.json')
    save(OUT/marker,dict(updated=stamp(),pid=os.getpid(),command=sys.argv))
    os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONNOUSERSITE='1')
    sys.path.insert(0,str(DEPS))
    import torch
    from PIL import Image,ImageDraw
    from face_alignment.utils import crop,flip,get_preds_fromhm
    from face_alignment.detection.folder import FaceDetector
    assert FaceDetector(device='cpu',verbose=False).reference_scale==195
    torch.set_num_threads(4)
    # Load the known local file directly; never call the network/download helper.
    net=torch.jit.load(str(MODEL),map_location='cpu').eval()
    rows=read(PRE/'coverage/observations.json')['rows'];obs=dict(np.load(PRE/'coverage/observations.npz'))
    bindings={b['ibug68_index']:b for b in read(REC/'direct-smplx-bindings/bindings.json')['anchors']}
    result_rows=[];hm_all=[];float_all=[];official_all=[];old_all=[];hm_points=[]
    heads=[];point_panels=[]
    with torch.inference_mode():
        for slot in SLOTS:
            assert time.monotonic()-started<(repair['remaining_seconds'] if repair_mode else 300)
            r=rows[slot];raw=Image.open(r['image']).convert('RGB');assert raw.size==tuple(r['wh'])
            image=np.asarray(raw);box=np.asarray(r['face_box'])
            center=torch.tensor([(box[0]+box[2])/2,(box[1]+box[3])/2],dtype=torch.float64)
            center[1]-=(box[3]-box[1])*.12
            scale=float((box[2]-box[0]+box[3]-box[1])/195.) # installed folder detector reference_scale
            inp=crop(image,center,scale)
            t=torch.from_numpy(inp.transpose(2,0,1).copy()).float()[None]/255.
            hm=net(t).detach();hm+=flip(net(flip(t)).detach(),is_label=True)
            hm=hm.numpy();points,official,scores=get_preds_fromhm(hm.copy(),center.numpy(),scale)
            length=200.*scale
            continuous=(points[0].astype(np.float64)/64.-.5)*length+center.numpy()[None]
            truncated=np.trunc(continuous).astype(np.int32)
            assert np.array_equal(truncated,official[0]),'Official continuous inverse mismatch'
            old=obs['keypoints'][slot,23:91];displacement=np.linalg.norm(continuous-official[0],axis=-1)
            # The figure is raw native pixels. Only a crosshair marks the queried
            # coordinate; no words, scaling, warping, or error-based selection.
            mid=np.mean(box.reshape(2,2),axis=0);x0,y0=np.floor(mid-96).astype(int)
            cb=[int(x0),int(y0),int(x0+192),int(y0+192)]
            for variant,pts in [('raw',None),('rtmw',old),('fan',continuous)]:
                panel=raw.crop(cb)
                if pts is not None:
                    draw=ImageDraw.Draw(panel)
                    for point in POINTS:
                        x,y=pts[point]-[x0,y0]
                        draw.ellipse((x-2,y-2,x+2,y+2),outline='#0a89ff' if variant=='fan' else '#d82344',width=1)
                heads.append(panel)
            for point in POINTS:
                p=continuous[point];px,py=np.floor(p-24).astype(int);cropbox=[int(px),int(py),int(px+48),int(py+48)]
                panels=[]
                for variant,pts in [('raw',None),('rtmw',old),('fan',continuous)]:
                    patch=raw.crop(cropbox)
                    if pts is not None:
                        draw=ImageDraw.Draw(patch);x,y=pts[point]-[px,py]
                        draw.line((x-3,y,x+3,y),fill='#0a89ff' if variant=='fan' else '#d82344',width=1)
                        draw.line((x,y-3,x,y+3),fill='#0a89ff' if variant=='fan' else '#d82344',width=1)
                    panels.append(patch)
                pic=Image.new('RGB',(144,48),'white')
                for col,im in enumerate(panels):pic.paste(im,(48*col,0))
                filename=f'frame_{r["source_frame_index"]:06d}_ibug_{point:02d}_native.png';pic.save(OUT/filename)
                point_panels.append(dict(slot=slot,source_frame=r['source_frame_index'],ibug=point,
                    rtmw133=point+23,anchor=bindings[point],old_uv=old[point].tolist(),
                    fan_uv_continuous=p.tolist(),fan_uv_official=official[0,point].tolist(),
                    native_crop=cropbox,image=filename,source_sha256=r['source_sha256'],
                    anatomical_visibility='pending_review',physical_accuracy_verified=False))
            result_rows.append(dict(slot=slot,source_frame=r['source_frame_index'],frame=r['frame'],roi=box.tolist(),
                crop_center=center.numpy().tolist(),crop_scale=scale,head_figure_crop=cb,
                max_integer_loss_px=float(displacement.max()),median_integer_loss_px=float(np.median(displacement)),
                count_integer_loss_ge1=int((displacement>=1).sum()),
                FAN_RTMw_median_distance_px=float(np.median(np.linalg.norm(continuous[17:]-old[17:],axis=-1))),
                native_pixel_precision_is_not_physical_accuracy=True))
            hm_all.append(hm[0]);hm_points.append(points[0]);float_all.append(continuous);official_all.append(official[0]);old_all.append(old)
            save(OUT/'progress.json',dict(updated=stamp(),done=len(result_rows),total=5,pid=os.getpid(),state='CPU_face_observations'))
    np.savez_compressed(OUT/'observations.npz',slots=SLOTS,source_indices=[rows[i]['source_frame_index'] for i in SLOTS],
        old_rtmw_native=old_all,fan_native_continuous=float_all,fan_native_official=official_all,heatmap_points=hm_points,heatmaps=hm_all)
    image=Image.new('RGB',(576,960),'white')
    for i,p in enumerate(heads):image.paste(p,((i%3)*192,(i//3)*192))
    image.save(OUT/'five-native-heads.png')
    # Each row contains all 8 fixed semantic locations for one source frame.
    for i,slot in enumerate(SLOTS):
        grid=Image.new('RGB',(576,96),'white')
        for j,row in enumerate(point_panels[i*8:(i+1)*8]):grid.paste(Image.open(OUT/row['image']),((j%4)*144,(j//4)*48))
        grid.save(OUT/f'points_slot_{slot:03d}.png')
    save(OUT/'point-review-index.json',dict(order=POINTS,columns=['raw','RTMW','continuous_FAN'],panels=point_panels))
    result=dict(updated=stamp(),state='complete_real_semantic_review_pending',rows=result_rows,
        images=5,network_forwards=10,output_points=340,native_points_continuous=True,
        original_inputs_unchanged=all(sha(Path(p))==h for p,h in plan['sources'].items() if Path(p)!=Path(__file__)),
        implementation_repair=repair_mode,executed_source_sha256=sha(Path(__file__)),
        inference_crop_resized_by_official_network=True,reported_figures_native_unscaled=True,
        new_camera=False,new_SMPLX_fit=False,new_training=False,physical_truth=False,quality_accepted=False,
        elapsed_seconds=time.monotonic()-started)
    save(OUT/'result.json',result);save(OUT/'progress.json',dict(updated=stamp(),state='complete_review_pending',pid=None))
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--register',action='store_true');p.add_argument('--repair-preinference',action='store_true');p.add_argument('--repair-folder-import',action='store_true');args=p.parse_args()
    if args.register:register()
    else:run(args.repair_preinference,args.repair_folder_import)
