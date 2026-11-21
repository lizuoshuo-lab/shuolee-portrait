"""Train-only actor5_6 frame750 data for a direct anisotropic-3DGS alternative."""
from pathlib import Path
import json,hashlib,shutil,cv2,numpy as np
from audit_enerf_input import read_cameras,digest
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
OUT=LAB/'outputs/dynamic-human-enerf-representation-048';OUT.mkdir(exist_ok=True)
DATA=Path('/opt/gs-slam-lab/4k4d/actor5_6-mask039')
RAW=Path('/mnt/d/GS-SLAM-Data/enerf_outdoor/extracted/actor5_6')
assert not (OUT/'input-audit.json').exists()
cams=read_cameras(RAW/'optimized');records=[]
receipt={r['camera']:r for r in json.loads((LAB/'outputs/dynamic-human-4k4d-mask-control-039/input-preparation.json').read_text())['records']}
for i in range(18):
 if i in [0,6,12]:continue
 c=f'{i:02d}';cam=cams[c];K=np.array(cam['K']);D=np.array(cam['D']);R=np.array(cam['w2c'])
 ip=DATA/'images'/c/'000750.png';mp=DATA/'bgmtv2'/c/'000750.png'
 rgb=cv2.imread(str(ip))[...,::-1].copy();matte=cv2.imread(str(mp),0).astype(np.float32)/255
 mx,my=cv2.initUndistortRectifyMap(K,D,None,K,(1920,1080),cv2.CV_32FC1)
 valid=(mx>=0)&(mx<1919)&(my>=0)&(my<1079)
 # Exact039 hashes establish common inputs. Its official loader uses float32
 # calibration; independent float64 remapping has sparse subpixel rounding differences.
 expected=cv2.remap(cv2.imread(str(RAW/'images'/c/'000750.jpg')),*[mx,my],cv2.INTER_LINEAR)[...,::-1]
 assert digest(ip)==receipt[c]['derived_rgb_sha256'] and digest(mp)==receipt[c]['derived_mask_sha256']
 diff=np.abs(expected.astype(np.int16)-rgb.astype(np.int16))
 assert diff.mean()<.02 and np.count_nonzero(diff)/diff.size<.01
 np.savez_compressed(OUT/f'train-camera-{c}.npz',rgb=rgb,matte=matte,valid=valid,K=K,w2c=R)
 records.append(dict(camera=c,rgb_sha256=digest(ip),mask_sha256=digest(mp),same039_pixels_exact=True,
                     float64_remap_mean_u8_difference=float(diff.mean()),float64_remap_max_u8_difference=int(diff.max())))
pts=[]
for p in (LAB/'outputs/dynamic-human-enerf-foreground-geometry-045').glob('foreground-??-??.npz'):
 z=np.load(p);keep=(abs(z['signed_sampson'])<1)&z['positive']&(z['reprojection'].max(1)<1)
 pts.append(z['world_points'][keep])
points=np.concatenate(pts);assert len(points)>500 and np.isfinite(points).all()
save=lambda p,x:p.write_text(json.dumps(x,indent=2)+'\n')
save(OUT/'geometry-audit.json',dict(frame=750,point_quantiles=np.quantile(points,[.005,.5,.995],axis=0).tolist(),
 points_used_for_initial_bounds=len(points),origin='045train-only mutually matched foreground features, both positive depths, Sampson and each reprojection <1px. Used for coarse hull bounds only.'))
save(OUT/'input-audit.json',dict(status='15native_train_inputs_verified_same_RGB_and_candidate_masks_as039_044',records=records,heldout_appearance_opened=False))
save(OUT/'protocol.json',dict(status='fixed_before_training',sequence='actor5_6',frame=750,
 method='Direct anisotropic3DGS with SH degree2 and gsplat1.3DefaultStrategy adaptive splitting/pruning.',
 budget_updates=6000,seed=73,training_cameras=[r['camera'] for r in records],appearance_heldout=['00','06','12'],
 shared_with_044='Same native training RGB, masks, unchanged calibration, single frame and heldout views.',
 differences='Model family, surface-shell initialization, adaptive point count, objective, optimizer, iteration budget and runtime. A method alternative, NOT a single-factor causal ablation or official4K4D reproduction.',
 purpose='Check whether direct anisotropic scene optimization recovers details missing in the point-based IBR pipeline.',
 selection='Fixed final6000updates; same development case, not a new final test.',goal_complete=False))
snap=OUT/'source-snapshot';snap.mkdir(exist_ok=True)
for n in ['prepare_enerf_representation_control.py','train_enerf_first_frame.py']:shutil.copy2(LAB/'scripts/dynamic_human'/n,snap/n)
print('048 prepared:15native inputs; no heldout appearance opened;6000step direct3DGS')
