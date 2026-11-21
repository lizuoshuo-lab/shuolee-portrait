"""Fresh CPU body observations on immutable full images; no geometry fitting."""
from pathlib import Path
import datetime as dt, hashlib, json, os, sys, time, zipfile
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current')
H=ROOT/'portrait-alignment-open-source-116/hsfm'
OUT=H/'input-pilot'
sys.path.insert(0,'/opt/gs-slam-lab/shuolee-hsfm116/python')
import onnxruntime as ort
from rtmlib import RTMPose
from hsfm116_adapter import dust3r_raster_transform,map_pixels,colmap_to_raster_K
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,data):
    path=OUT/name;tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)
assert not OUT.exists(),'Do not overwrite a completed or partial pilot'
OUT.mkdir()
man=ROOT/'avatar-repository-pair-114/smplx/data/training-inputs.json'
rows=json.loads(man.read_text())['records'];assert len(rows)==175
assert {r['source_frame_index'] for r in rows}==set(range(1,176))
ids=[100,13,25,38,50,63,75,88,113,125,138,150,163]
selected=[next(r for r in rows if r['source_frame_index']==i) for i in ids]
model=Path('/mnt/d/GS-SLAM-Data/models/rtmw/rtmw-l-384x288.onnx')
asset=json.loads((H/'rtmw-compatibility.json').read_text())
assert sha(model)==asset['sha256']
now=dt.datetime.now(dt.timezone.utc).isoformat()
save('protocol.json',dict(created=now,state='input_preparation_only',pid=os.getpid(),
    selection='13 fixed previously inspected source identities; frame100 first as visible front initialization',
    raw_frame_ids=ids,full_identity_count=175,source_manifest=str(man),source_manifest_sha256=sha(man),
    source_images='Full1514x2694 images rectified once by immutable074 estimates; verified115 raw-video chain. No person-only warp.',
    only_subject_data_loaded='Existing original-observation RGB/masks and estimated camera metadata; no fitted geometry or learned Gaussian.',
    camera_physical_accuracy_proven=False,model_sha256=asset['sha256'],
    dependency='rtmlib0.0.16 + CPU ONNX Runtime1.23.2',
    body_reid='One user target from existing foreground mask; other image people not assigned target ID',
    keypoints='Keep all133 native observations and raw scores; planned HSfM SMPL branch uses body23 names only',
    score_semantics='Raw SimCC peak responses, not probabilities. No threshold is calibrated by this script.',
    body_template='Supported HSfM SMPL branch avoids unverified hand/face-contour mapping; independent face texture validation still required',
    inference_device='cpu',max_minutes=10,subject_fit=False,gaussian_training=False))
with zipfile.ZipFile(model.with_suffix('.zip')) as z:pipeline=json.loads(z.read('pipeline.json'))
norm=next(t for t in pipeline['pipeline']['tasks'][0]['transforms'] if t['type']=='Normalize')
class Detector(RTMPose):
    def __init__(self):
        opt=ort.SessionOptions();opt.intra_op_num_threads=4;opt.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(model),sess_options=opt,providers=['CPUExecutionProvider'])
        self.model_input_size=(288,384);self.mean=np.array(norm['mean']);self.std=np.array(norm['std'])
        self.onnx_model=str(model);self.backend,self.device,self.to_openpose='onnxruntime','cpu',False
d=Detector();start=time.monotonic();records=[];keypoints=[];scores=[];boards=[]
for count,row in enumerate(selected):
    image=Path(row['image']);mask=Path(row['mask'])
    assert sha(image)==row['source_sha256'] and sha(mask)==row['mask_sha256']
    rgb=np.asarray(Image.open(image).convert('RGB'));m=np.asarray(Image.open(mask).convert('L'))>127
    assert list(rgb.shape[1::-1])==row['wh'] and m.shape==rgb.shape[:2]
    yy,xx=np.where(m);assert len(xx)>1000
    box=np.array([xx.min(),yy.min(),xx.max()+1,yy.max()+1],dtype=np.float32)
    kp,sc=d(rgb,bboxes=[box]);kp,sc=kp[0],sc[0]
    assert kp.shape==(133,2) and np.isfinite(kp).all() and np.isfinite(sc).all()
    tx=dust3r_raster_transform(tuple(row['wh']))
    processed=map_pixels(kp,tx.centers)
    error=float(abs(map_pixels(processed,np.linalg.inv(tx.centers))-kp).max());assert error<1e-8
    records.append(dict(raw_frame=row['source_frame_index'],image=str(image),mask=str(mask),
        image_sha256=row['source_sha256'],mask_sha256=row['mask_sha256'],wh=row['wh'],bbox_xyxy=box.tolist(),
        processed_wh=tx.output_wh,center_affine=tx.centers.tolist(),edge_affine=tx.edges.tolist(),
        K_from_existing_manifest=row['K'],K_origin='COLMAP corner convention; subtract0.5 before center-affine if used',
        K_processed_centers=tx.intrinsics(colmap_to_raster_K(row['K'])).tolist(),
        keypoint_coordinate_origin='zero-based integer raster centers',roundtrip_max_px=error))
    keypoints.append(kp);scores.append(sc)
    # Isotropic resize for review only. Inference used the original entire RGB.
    scale=360/rgb.shape[0];w=int(round(rgb.shape[1]*scale));im=Image.fromarray(rgb).resize((w,360))
    panel=Image.new('RGB',(w,388),'white');panel.paste(im,(0,28));draw=ImageDraw.Draw(panel)
    draw.text((4,5),f"raw{row['source_frame_index']:06d} | RTMW",fill='black')
    for k,(x,y) in enumerate(kp[:23]):
        xp,yp=x*scale,y*scale+28
        color='lime' if k<17 else 'cyan'
        draw.ellipse((xp-2,yp-2,xp+2,yp+2),fill=color)
    boards.append(panel)
    save('progress.json',dict(updated=dt.datetime.now(dt.timezone.utc).isoformat(),pid=os.getpid(),done=count+1,total=len(ids),seconds=time.monotonic()-start))
    assert time.monotonic()-start<600
np.savez_compressed(OUT/'observations.npz',raw_frame_ids=ids,keypoints=np.stack(keypoints),scores=np.stack(scores))
save('inputs.json',dict(records=records,all175_identities=[r['source_frame_index'] for r in rows]))
for bi in range(0,len(boards),5):
    subset=boards[bi:bi+5];canvas=Image.new('RGB',(sum(p.width for p in subset),388),'white');x=0
    for p in subset:canvas.paste(p,(x,0));x+=p.width
    canvas.save(OUT/f'body23-observations-{bi//5+1}.jpg',quality=95)
result=dict(updated=dt.datetime.now(dt.timezone.utc).isoformat(),state='fresh_input_observations_complete_visual_review_pending',
    frames=len(ids),raw_frame_ids=ids,seconds=time.monotonic()-start,previous_fit_loaded=False,gpu=False,
    subject_fit=False,observations_sha256=sha(OUT/'observations.npz'),
    views_are_simultaneous=False,template_visibility_or_rigidity_verified=False,
    physical_camera_accuracy_verified=False,point_cloud_created=False)
save('result.json',result);print(json.dumps(result),flush=True)
