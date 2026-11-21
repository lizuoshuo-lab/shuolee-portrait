"""Register a fresh official SMPLify-X initialization; no network body template."""
import argparse, copy, hashlib, os, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
from run_official_gaussians123 import read,save,sha,now,PROJECT,ASSETS
H=Path(__file__).resolve().parent
REC=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
ROOT=REC/'cross-subject-sh0-124/winterhat-input'
DATA=Path('/opt/gs-slam-lab/shuolee-local-surface-123/winterhat')
VIDEO=Path('/mnt/d/00/工作/数据集/video1.mp4')
EXPECTED='32be30636adabfc4c25a1c8387cf19302d3ca39b4f040072081646d36523317c'

def rank(v):
    v=np.array(v);return np.array([(sum(v<x)+.5*sum(v==x))/len(v) for x in v])

def prepare():
    assert not ROOT.exists();assert sha(VIDEO)==EXPECTED
    ROOT.mkdir();(ROOT/'data').mkdir()
    save(ROOT/'plan.json',dict(updated=now(),user='草地白衣＋冬帽人物',subject='winterhat',
        route='Official SMPLify-X zero/mean initialization, shared subject shape, frozen SH0 Gaussian configuration',
        prefit_rule='8 equal-population chronological bins; minimize .6 rank(-body confidence)+.2 rank(-face confidence)+.2 rank(-native face sharpness), tie by frame ID. No fit/Gaussian losses used.',
        reference_rule='Among8 selected, lowest normalized eye-nose asymmetry; tie by frame ID. Measured 2D heuristic, not physical visibility.',
        review_rule='4 additional evenly spaced chronological midpoint frames outside training list, fit only for review; never transfer to training.',
        comparison_limit='Grass uses cached94-frame fits and fitted-yaw selection. Winterhat has no official fit: this is same Gaussian configuration, not identical full-pipeline or speed comparison.',
        one_reference_fit=True,one_12frame_fit=True,gaussian_budget=18000,review_at=1000,
        no_network_3d_template=True,no_vhap=True,no_download=True,no_threshold_sweep=True,
        cold_start_benchmark=False,preprocessing_cached=True,quality_accepted=False))
    import cv2
    cv2.setNumThreads(1)
    from smplify_score_contract123 import relative_confidence
    rows=read(DATA/'data/inputs.json')['records'];n=len(rows)
    with np.load(DATA/'data/observations.npz',allow_pickle=False) as z:obs=dict(z)
    assert obs['frame_ids'].astype(str).tolist()==[r['frame'] for r in rows]
    confidence,_=relative_confidence(obs['scores'],obs['boxes'],obs['flip_disagreement'])
    candidates=[]
    for i,r in enumerate(rows):
        assert sha(r['image'])==r['source_sha256'] and sha(r['mask'])==r['mask_sha256']
        raw=Image.open(r['image']).convert('RGB');assert list(raw.size)==r['wh']
        face=obs['keypoints'][i,23:91];le=face[36:42].mean(0);re=face[42:48].mean(0)
        eye=re-le;asym=abs(np.dot(face[30]-(le+re)/2,eye))/(np.dot(eye,eye)+1e-6)
        b=np.r_[face.min(0)-16,face.max(0)+16].astype(int);b[:2]=np.maximum(b[:2],0);b[2:]=np.minimum(b[2:],raw.size)
        crop=np.array(raw.crop(b));sharp=float(cv2.Laplacian(cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY),cv2.CV_64F).var())
        candidates.append(dict(slot=i,frame=r['frame'],time_seconds=r['time_seconds'],body_conf=float(np.median(confidence[i,:17])),face_conf=float(np.median(confidence[i,40:91])),sharpness=sharp,asymmetry=float(asym),face_box=b.tolist()))
    scores=.6*rank([-v['body_conf'] for v in candidates])+.2*rank([-v['face_conf'] for v in candidates])+.2*rank([-v['sharpness'] for v in candidates])
    order=sorted(range(n),key=lambda i:(rows[i]['time_seconds'],rows[i]['frame']));chosen=[]
    for group in np.array_split(order,8):chosen.append(min(map(int,group),key=lambda i:(scores[i],rows[i]['frame'])))
    remaining=[i for i in order if i not in chosen]
    held=[int(g[len(g)//2]) for g in np.array_split(remaining,4)]
    ref=min(chosen,key=lambda i:(candidates[i]['asymmetry'],rows[i]['frame']))
    for i,v in enumerate(candidates):v.update(score=float(scores[i]),training=i in chosen,review_only=i in held)
    # Explicit model-relative gauge. Do not import the old network pose or scale.
    document=copy.deepcopy(read(DATA/'data/inputs.json'))
    for r in document['records']:
        r['source_camera_w2c_unused']=r['w2c'];r['w2c']=np.eye(4).tolist()
        r['camera_kind']='camera_relative_fixedK_body_motion';r['camera_registration']=dict(static_trajectory_claim=False,interpolation_used=False)
    save(ROOT/'data/inputs.json',document)
    (ROOT/'data/observations.npz').write_bytes((DATA/'data/observations.npz').read_bytes())
    contract=dict(translation_unit='meters',meters_per_camera_unit=1.,basis='model_conditioned',evidence='Camera-relative SMPL-X units; identity E and fitted per-frame root body poses. No static scene trajectory or physical metric claim.')
    save(ROOT/'length-contract.json',dict(frames=dict(path=str(ROOT/'data/inputs.json'),sha256=sha(ROOT/'data/inputs.json')),observations=dict(path=str(ROOT/'data/observations.npz'),sha256=sha(ROOT/'data/observations.npz')),camera_length_contract=contract))
    selection=dict(updated=now(),source_video_sha256=EXPECTED,source_roster=n,training_frames=[rows[i]['frame'] for i in chosen],review_only_frames=[rows[i]['frame'] for i in held],reference_frame=rows[ref]['frame'],fit_frames=[rows[i]['frame'] for i in order if i in chosen+held],candidates=candidates,quality_accepted=False)
    save(ROOT/'selection.json',selection)
    pics=[]
    for i in chosen:
        im=Image.open(rows[i]['image']).convert('RGB');mask=Image.open(rows[i]['mask']).convert('L');b=mask.getbbox();crop=im.crop(b);crop.thumbnail((250,430));pics.append(crop)
    w=max(p.width for p in pics);h=max(p.height for p in pics);sheet=Image.new('RGB',(4*w,2*h),'white')
    for j,pic in enumerate(pics):sheet.paste(pic,((j%4)*w,(j//4)*h))
    sheet.save(ROOT/'selected-eight.png');print(selection['training_frames'],selection['reference_frame'],flush=True)

def fit_reference():
    import fcntl,official_smplifyx_entry123 as entry
    from run_official_sequence123_topology import cache_static_topology
    lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sel=read(ROOT/'selection.json');original=entry.fit_one
    def wrapped(**kw):cache_static_topology(kw['body_model']);return original(**kw)
    entry.fit_one=wrapped
    sys.argv=['official_smplifyx_entry123.py','--official-root',str(REC/'smplifyx-reference-control/official'),
        '--data-root',str(ROOT),'--runtime-root',str(PROJECT),'--assets-root',str(ASSETS),
        '--frame-id',sel['reference_frame'],'--output-dir',str(ROOT/'reference-fit'),'--device','cuda']
    entry.main()

def fit_sequence():
    import run_official_sequence123_topology as original
    sel=read(ROOT/'selection.json')
    sys.argv=['run_official_sequence123_topology.py','--official-root',str(REC/'smplifyx-reference-control/official'),
        '--data-root',str(ROOT),'--runtime-root',str(PROJECT),'--assets-root',str(ASSETS),'--project-root',str(PROJECT),
        '--length-contract-json',str(ROOT/'length-contract.json'),'--reference-frame-id',sel['reference_frame'],
        '--shared-betas-npz',str(ROOT/'reference-fit/winner-camera-space.npz'),
        '--output-dir',str(ROOT/'official-fit'),'--device','cuda','--frame-ids',*sel['fit_frames']]
    original.main()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','reference','sequence']);args=ap.parse_args()
    {'prepare':prepare,'reference':fit_reference,'sequence':fit_sequence}[args.mode]()
