"""Native training-only RGB/matte remapping for the independent actor2_3 scene."""
from pathlib import Path
import json,hashlib,shutil
import cv2,numpy as np
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-actor23-independent-106'
D=Path('D:/GS-SLAM-Data/enerf_outdoor/extracted/actor2_3');N=O/'native-input';N.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf8'))
save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (O/'training-input-audit.json').exists()
plan=read(O/'validation-reservation.json');cameras=read(O/'cameras.json');rows=[];cv2.setNumThreads(2)
for c in plan['train_cameras']:
    assert c not in plan['heldout_cameras']
    K=np.array(cameras[c]['K'],dtype=np.float64);dist=np.array(cameras[c]['D'],dtype=np.float64);W=np.array(cameras[c]['w2c'],dtype=np.float64)
    mx,my=cv2.initUndistortRectifyMap(K,dist,None,K,(1920,1080),cv2.CV_32FC1)
    support=cv2.remap(np.ones((1080,1920),np.float32),mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    valid=support>.999
    assert valid.mean()>.7
    for f in plan['train_frames']:
        p=D/'images'/c/f'{f:06d}.jpg';m=D/'bgmtv2'/c/f'{f:06d}.jpg'
        rgb=cv2.imread(str(p),cv2.IMREAD_COLOR);matte=cv2.imread(str(m),cv2.IMREAD_GRAYSCALE)
        assert rgb.shape==(1080,1920,3) and matte.shape==(1080,1920)
        rgb=cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB)
        rgb=cv2.remap(rgb,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        matte=cv2.remap(matte.astype(np.float32)/255,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        assert np.isfinite(matte).all() and float(matte.min())>=0 and float(matte.max())<=1
        yy,xx=np.where((matte>.05)&valid);assert len(xx)>1000
        bbox=np.array([max(0,xx.min()-24),max(0,yy.min()-24),min(1920,xx.max()+25),min(1080,yy.max()+25)],dtype=np.int64)
        out=N/f'camera{c}-frame{f:06d}.npz';assert not out.exists()
        np.savez_compressed(out,rgb=rgb,matte=matte,valid=valid,K=K,w2c=W,bbox=bbox)
        if c in ['04','13'] and f in [600,612,624]:
            z=np.load(out);assert np.array_equal(z['rgb'],rgb) and np.array_equal(z['matte'],matte) and np.array_equal(z['valid'],valid)
        rows.append(dict(camera=c,frame=f,split='train',source_rgb_sha256=sha(p),source_matte_sha256=sha(m),input_sha256=sha(out),input=str(out.relative_to(O)),bbox=bbox.tolist(),foreground_pixels=int(((matte>.5)&valid).sum()),valid_pixels=int(valid.sum())))
        save(O/'preprocessing-progress.json',dict(training_inputs=len(rows)))
    print('ACTOR23_NATIVE',c,len(rows),flush=True)
assert len(rows)==195
save(O/'training-input-audit.json',dict(status='195_native_training_inputs_prepared_geometry_timing_unqualified',records=rows,heldout_RGB_opened=False,images_changed_in_source=False,width=1920,height=1080,preprocessing='OpenCV pinhole undistortion with original K, RGB and grayscale matte use identical INTER_LINEAR maps. Full native canvas retained, zero border. Valid support uses remapped all-ones image >0.999. No sharpening, inpainting or learned image enhancement.',matte_semantics='Author released background-matting estimates, not human semantic ground truth. Toy and boundary uncertainty retained.',numeric_roundtrip_cases=6,calibration_matrices_sha256=sha(O/'cameras.json'),required_next=['training-only reprojection qualification','training-only timing qualification','native mask visual review','train-only initialization and frozen static/dynamic protocol'],training_started=False,goal_complete=False))
shutil.copy2(__file__,O/Path(__file__).name);print('ACTOR23_PREPROCESS_COMPLETE',flush=True)
