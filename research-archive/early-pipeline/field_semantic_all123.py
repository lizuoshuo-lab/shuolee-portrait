"""One full-sequence native face observation pass, reusing the five-view audit.

No camera estimation, 3D initialization, training or visibility inference.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,sys,time,traceback
import numpy as np

H=Path(__file__).parent
REC=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
PRE=REC/'field-preflight';OLD=PRE/'semantic-face-binding';OUT=PRE/'semantic-face-all'
DEPS=Path('/opt/gs-slam-lab/smpl-x-alignment-20261029/face-diagnostic-deps')
MODEL=Path('/opt/gs-slam-lab/smpl-x-alignment-20261029/face-diagnostic-cache/hub/checkpoints/3DFAN4-4a694010b9.zip')
POINTS=[30,33,36,39,42,45,48,54]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()

def register():
    assert not OUT.exists();OUT.mkdir()
    rows=read(PRE/'coverage/observations.json')['rows'];assert len(rows)==177
    prior=read(OLD/'review-result.json');assert prior['direct_bindings_saved'] and not prior['quality_accepted']
    files=[PRE/'coverage/observations.json',PRE/'coverage/observations.npz',OLD/'observations.npz',OLD/'visual-review.json',
        OLD/'real-2d-smplx-bindings.json',REC/'direct-smplx-bindings/bindings.json',MODEL,
        DEPS/'face_alignment/api.py',DEPS/'face_alignment/utils.py',DEPS/'face_alignment/detection/folder/folder_detector.py',Path(__file__)]
    plan=dict(updated=now(),previous_goal_turn='progress: completed actual five-view face observations and 40-point review, explicit SMPL-X topology bindings, continuous inverse and viewer/PPT publication',
        purpose='Generate the complete registered sequence of native 2D anatomical observations for direct SMPL-X correspondence; independently review visibility.',
        hypothesis='Continuous native coordinates and explicit observed/occluded/unknown states can supply the actual SMPL-X objective without quantization or invisible face targets.',
        isolated_scope='Same existing model/crop/decoder as the completed five-view check; only expand to the remaining registered frames. No camera change or new detector comparison.',
        frames=len(rows),source_frames=882,reuse_slots=[0,44,88,132,176],new_images=172,new_network_forwards=344,
        body_observations='Existing RTMW native coordinates preserved, not re-extracted.',
        native_size=[2704,1520],points_to_review=POINTS,
        review='All177 heads in source order, native192px raw and detector panels, 12frames per sheet. Review semantic regions and visibility; unknown localization uncertainty is not subpixel truth.',
        image_policy='No burned labels, resizing or warping in review figures; frame identities stored beside the figures in JSON/HTML.',
        budget=dict(CPU_threads=4,max_seconds=600,GPU_tasks=0,new_downloads=0,new_models=0,training=0),
        stops=['Source identity or native dimensions change','Nonfinite crop/output','Official integer output differs from truncation of continuous inverse','Budget exhausted'],
        original_frame_hashes={r['frame']:r['source_sha256'] for r in rows},sources={str(p):sha(p) for p in files},
        training_admitted=False,physical_truth=False,quality_accepted=False)
    assert plan['sources'][str(MODEL)]=='4a694010b93140609b64a6ea734afc0ee80a4798394740281b0c5bea0ebef1d9'
    save(OUT/'plan.json',plan);print(json.dumps(dict(plan=str(OUT/'plan.json'),new_images=172,reuse=5)),flush=True)

def run():
    plan=read(OUT/'plan.json');assert not (OUT/'started.json').exists()
    for path,h in plan['sources'].items():assert sha(Path(path))==h,path
    save(OUT/'started.json',dict(updated=now(),pid=os.getpid(),command=sys.argv))
    started=time.monotonic();os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONNOUSERSITE='1')
    sys.path.insert(0,str(DEPS))
    import torch
    from PIL import Image,ImageDraw
    from face_alignment.utils import crop,flip,get_preds_fromhm
    from face_alignment.detection.folder import FaceDetector
    torch.set_num_threads(4);assert FaceDetector(device='cpu',verbose=False).reference_scale==195
    net=torch.jit.load(str(MODEL),map_location='cpu').eval()
    rows=read(PRE/'coverage/observations.json')['rows'];old=dict(np.load(OLD/'observations.npz'))
    reuse={int(slot):i for i,slot in enumerate(old['slots'])};assert set(reuse)==set(plan['reuse_slots'])
    xy=[];official_xy=[];hm_pts=[];records=[];sheets=[];forwards=0;sheet=None
    with torch.inference_mode():
        for slot,row in enumerate(rows):
            assert time.monotonic()-started<plan['budget']['max_seconds'],'CPU budget exhausted'
            raw_path=Path(row['image']);assert sha(raw_path)==plan['original_frame_hashes'][row['frame']]
            raw=Image.open(raw_path).convert('RGB');assert raw.size==(2704,1520)
            box=np.asarray(row['face_box'],dtype=np.float64);center=np.array([(box[0]+box[2])/2,(box[1]+box[3])/2])
            center[1]-=(box[3]-box[1])*.12;scale=float(((box[2]-box[0])+(box[3]-box[1]))/195.)
            assert np.isfinite(center).all() and np.isfinite(scale) and scale>0
            if slot in reuse:
                i=reuse[slot];points=old['heatmap_points'][i];native=old['fan_native_continuous'][i];official=old['fan_native_official'][i]
            else:
                inp=crop(np.asarray(raw),torch.from_numpy(center),scale)
                t=torch.from_numpy(inp.transpose(2,0,1).copy()).float()[None]/255.
                hm=net(t).detach();hm+=flip(net(flip(t)).detach(),is_label=True);forwards+=2
                points,official,_=get_preds_fromhm(hm.numpy().copy(),center,scale)
                points=points[0];official=official[0]
                native=(points.astype(np.float64)/64.-.5)*(200.*scale)+center[None]
            assert np.isfinite(native).all() and np.array_equal(np.trunc(native).astype(np.int32),official)
            xy.append(native);official_xy.append(official);hm_pts.append(points)
            mid=np.mean(box.reshape(2,2),axis=0);x0,y0=np.floor(mid-96).astype(int)
            cropbox=[int(x0),int(y0),int(x0+192),int(y0+192)]
            raw_panel=raw.crop(cropbox);marked=raw_panel.copy();d=ImageDraw.Draw(marked)
            for point in POINTS:
                x,y=native[point]-[x0,y0];d.ellipse((x-2,y-2,x+2,y+2),outline='#0a89ff',width=1)
            if slot%12==0:sheet=Image.new('RGB',(1152,768),'white')
            cell=slot%12;left=(cell%3)*384;top=(cell//3)*192
            sheet.paste(raw_panel,(left,top));sheet.paste(marked,(left+192,top))
            if slot%12==11 or slot==len(rows)-1:
                begin=(slot//12)*12;fn=f'native_slots_{begin:03d}_{slot:03d}.png';sheet.save(OUT/fn)
                sheets.append(dict(file=fn,sha256=sha(OUT/fn),first_slot=begin,last_slot=slot,columns=3,rows=4,
                    frames=[dict(slot=j,source_frame=rows[j]['source_frame_index'],frame=rows[j]['frame']) for j in range(begin,slot+1)]))
            records.append(dict(slot=slot,source_frame=row['source_frame_index'],frame=row['frame'],source_sha256=row['source_sha256'],
                source_image=row['image'],wh=row['wh'],roi=box.tolist(),crop_center=center.tolist(),crop_scale=scale,native_review_crop=cropbox,
                reused=slot in reuse,visibility='unreviewed',uncertainty_px=None,training_eligible=False))
            if (slot+1)%12==0 or slot==len(rows)-1:
                progress=dict(updated=now(),pid=os.getpid(),done=slot+1,total=len(rows),new_network_forwards=forwards,state='CPU_native_face_observations',elapsed_seconds=time.monotonic()-started)
                save(OUT/'progress.json',progress);print(json.dumps(progress),flush=True)
    assert forwards==344 and len(records)==177
    np.savez_compressed(OUT/'observations.npz',slots=np.arange(177),source_indices=[r['source_frame_index'] for r in rows],
        frame_ids=[r['frame'] for r in rows],fan_native_continuous=xy,fan_native_official=official_xy,heatmap_points=hm_pts)
    save(OUT/'observations.json',dict(coordinate_domain='native_integer_pixel_centres',rows=records))
    save(OUT/'review-sheets.json',dict(panels=['raw','continuous_FAN'],point_indices=POINTS,sheets=sheets))
    for slot,i in reuse.items():assert np.array_equal(np.asarray(xy)[slot],old['fan_native_continuous'][i])
    result=dict(updated=now(),state='completed_visibility_review_pending',frames=177,reused=5,new_images=172,new_network_forwards=forwards,
        original_source_hashes_verified=True,reused_predictions_byte_equal=True,review_sheets=len(sheets),observations_sha256=sha(OUT/'observations.npz'),
        no_camera_update=True,no_SMPLX_fit=True,no_gaussian_training=True,physical_truth=False,quality_accepted=False,elapsed_seconds=time.monotonic()-started)
    save(OUT/'result.json',result);save(OUT/'progress.json',dict(updated=now(),state='completed_review_pending',pid=None,done=177,total=177))
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--register',action='store_true');args=p.parse_args()
    if args.register:register()
    else:
        try:run()
        except BaseException as e:
            if OUT.exists():
                save(OUT/'failure.json',dict(updated=now(),error=repr(e),traceback=traceback.format_exc(),terminal=True))
                save(OUT/'progress.json',dict(updated=now(),state='failed',pid=None,error=repr(e)))
            raise
