"""Prepare lossless, already rectified inputs for one fixed mask-pipeline intervention."""
from pathlib import Path
import json,cv2,numpy as np,yaml,shutil
from PIL import Image
from easyvolcap.utils.data_utils import load_resize_undist_im_bytes,load_image_from_bytes
from audit_enerf_input import read_cameras,digest
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
ROOT=Path('/mnt/d/GS-SLAM-Data/enerf_outdoor/extracted/actor5_6')
MASK=LAB/'outputs/dynamic-human-enerf-mask-quality-038/train15/hybrid-closed-core'
OUT=LAB/'outputs/dynamic-human-4k4d-mask-control-039';OUT.mkdir(exist_ok=True)
DATA=Path('/opt/gs-slam-lab/4k4d/actor5_6-mask039');DATA.mkdir(exist_ok=True)
assert not (OUT/'input-preparation.json').exists()
cameras=read_cameras(ROOT/'optimized');train=[i for i in range(18) if i not in [0,6,12]]
protocol=dict(status='fixed_before_input_preparation',frame=750,training_cameras=train,appearance_heldout=[0,6,12],
    intervention='Training-view masks: SAM2 manually prompted union plus released high-alpha pixels only within49px elliptical morphological closing. Recompute train-only visual hull, mask crop and source masking from these candidates.',
    unchanged='Same official4K4D point representation,262144points,15training cameras,8texture sources,2gradient sources,native resolution,K/extrinsics,1000updates,strict15/15visualhull,originalimageappearance.',
    distinction='This changes the whole input-mask pipeline including initialization; not isolated optimization-loss-only ablation. Source images pre-rectified once losslessly for mask-coordinate agreement; derivedD=0 prevents double distortion. Originalcamera files untouched.',
    mask_limit='Manually reviewed input candidates, not semanticGT; some near-body wall residue and hard boundaries remain.',
    evaluation='Same released target references and native full image evaluation as035; additionally inspect raw target photos to avoid accepting erased body regions as truth.',
    calibration_candidate_037='Not applied',goal_complete=False)
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
(DATA/'optimized').mkdir(exist_ok=True)
fs=cv2.FileStorage(str(DATA/'optimized/intri.yml'),cv2.FILE_STORAGE_WRITE)
fs.startWriteStruct('names',cv2.FileNode_SEQ)
for c in cameras:fs.write('',c)
fs.endWriteStruct()
for c,cam in cameras.items():
    fs.write('K_'+c,np.array(cam['K'],dtype=np.float64));fs.write('dist_'+c,np.zeros((1,len(cam['D'])),dtype=np.float64))
    fs.write('H_'+c,1080);fs.write('W_'+c,1920)
fs.release();shutil.copy2(ROOT/'optimized/extri.yml',DATA/'optimized/extri.yml')
derived=read_cameras(DATA/'optimized');records=[]
for c,cam in cameras.items():
    K=np.array(cam['K'],dtype=np.float32);D=np.array(cam['D'],dtype=np.float32)
    assert np.array_equal(derived[c]['K'],cam['K']) and np.array_equal(derived[c]['w2c'],cam['w2c'])
    assert not np.any(derived[c]['D'])
    (DATA/'images'/c).mkdir(parents=True,exist_ok=True);(DATA/'bgmtv2'/c).mkdir(parents=True,exist_ok=True)
    rgb_path=ROOT/'images'/c/'000750.jpg';mask_path=ROOT/'bgmtv2'/c/'000750.jpg'
    a,k,h,w=load_resize_undist_im_bytes(str(rgb_path),K.copy(),D.copy(),encode_ext='.png',dist_opt_K=False)
    # Official loader returns an encoded BGR PNG; write its bytes verbatim.
    target=DATA/'images'/c/'000750.png';target.write_bytes(a.tobytes())
    reread,kk,hh,ww=load_resize_undist_im_bytes(str(target),K.copy(),np.zeros_like(D),encode_ext='.png',dist_opt_K=False)
    assert np.array_equal(load_image_from_bytes(a),load_image_from_bytes(reread))
    assert np.array_equal(k,kk) and (h,w,hh,ww)==(1080,1920,1080,1920)
    original_mask,mk,_,_=load_resize_undist_im_bytes(str(mask_path),K.copy(),D.copy(),encode_ext='.png',dist_opt_K=False)
    dest=DATA/'bgmtv2'/c/'000750.png'
    if int(c) in train:
        shutil.copy2(MASK/f'camera{c}-mask.png',dest)
        # Segmentation uses native double-K cv2 maps; quantify exact RGB difference
        # from official float32-K decoding instead of claiming byte equivalence.
        segmentation_rgb=np.array(Image.open(MASK.parent/f'camera{c}-source.png'))
        official_rgb=cv2.cvtColor(load_image_from_bytes(a),cv2.COLOR_BGR2RGB)
        diff=np.abs(official_rgb.astype(np.int16)-segmentation_rgb.astype(np.int16))
        segmentation_difference=dict(max_abs_u8=int(diff.max()),changed_values=int(np.count_nonzero(diff)),mean_abs_u8=float(diff.mean()))
    else:
        dest.write_bytes(original_mask.tobytes());segmentation_difference=None
    records.append(dict(camera=c,source_rgb_sha256=digest(rgb_path),source_matte_sha256=digest(mask_path),derived_rgb_sha256=digest(target),derived_mask_sha256=digest(dest),official_decode_roundtrip_exact=True,segmentation_to_official_rgb_difference=segmentation_difference))
cfg=yaml.safe_load((LAB/'outputs/dynamic-human-4k4d-official-035/pilot.yaml').read_text())
cfg['exp_name']='enerf_actor5_6_train15_frame750_mask039'
for key in ['dataloader_cfg','val_dataloader_cfg']:
    dc=cfg[key]['dataset_cfg'];dc['data_root']=str(DATA);dc['ims_pattern']='000750.png'
    assert dc['frame_sample']==[750,751,1]
for key,suffix in [('trained_model','model')]:cfg['runner_cfg'][key]=str(OUT/suffix)
cfg['runner_cfg']['visualizer_cfg']['result_dir']=str(OUT/'official-render')
cfg['runner_cfg']['recorder_cfg']['record_dir']=str(OUT/'tensorboard')
(OUT/'pilot.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
(OUT/'input-preparation.json').write_text(json.dumps(dict(status='18native_input_files_and_calibration_prepared',records=records,
    data_root=str(DATA),frame_path_mapping='Single fixed filename000750.png with original frame_sample750:751; no synthetic frames.',
    config_sha256=digest(OUT/'pilot.yaml'),goal_complete=False),indent=2)+'\n')
print('Prepared039:18nativeRGB,15candidate+3unchangedreleasedmasks; officialRGBroundtripexact')
