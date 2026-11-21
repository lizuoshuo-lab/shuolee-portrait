"""CPU source/pixel/metric verification separate from the rendering process."""
from pathlib import Path
import numpy as np,cv2,json,hashlib,os
from PIL import Image
from audit_enerf_input import read_cameras,digest
LAB=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab')
OUT=Path(os.environ.get('FOURKD_RUN_DIR',str(LAB/'outputs/dynamic-human-4k4d-official-035')))
EVAL=OUT/'native-evaluation';ROOT=Path(r'D:\GS-SLAM-Data\enerf_outdoor\extracted\actor5_6')
report=json.loads((EVAL/'metrics.json').read_text());cameras=read_cameras(ROOT/'optimized');checks=[]
assert len(report['records'])==6
def u8(a):return np.rint(np.clip(a,0,1)*255).astype(np.uint8)
for row in report['records']:
    name=row['camera'];c=int(name);prefix=f'camera{name}-frame000750'
    z=np.load(EVAL/f'{prefix}-raw.npz');cam=cameras[name];K=np.array(cam['K']);D=np.array(cam['D'])
    assert np.allclose(z['K'],K,atol=1e-10) and np.allclose(z['w2c'],cam['w2c'],atol=1e-10)
    assert not set(row['source_cameras'])&{0,6,12,c}
    src=ROOT/'images'/name/'000750.jpg';mask=ROOT/'bgmtv2'/name/'000750.jpg'
    assert digest(src)==row['source_rgb_sha256'] and digest(mask)==row['source_matte_sha256']
    mx,my=cv2.initUndistortRectifyMap(K,D,None,K,(1920,1080),cv2.CV_32FC1)
    rgb=cv2.remap(cv2.imread(str(src))[...,::-1],mx,my,cv2.INTER_LINEAR)
    matte=cv2.remap(cv2.imread(str(mask),cv2.IMREAD_GRAYSCALE).astype(np.float32)/255,mx,my,cv2.INTER_LINEAR)
    valid=(mx>=0)&(mx<1919)&(my>=0)&(my<1079)
    assert np.array_equal(rgb,z['source']) and np.array_equal(matte,z['matte']) and np.array_equal(valid,z['valid'])
    ref=rgb.astype(np.float32)/255*matte[...,None]+.5*(1-matte[...,None])
    pred=np.clip(z['rgb']+.5*(1-z['alpha'][...,None]),0,1)
    assert np.isfinite(pred).all() and np.isfinite(z['alpha']).all()
    for suffix,arr in [('source',rgb),('reference',u8(ref)),('trained',u8(pred)),('alpha',u8(z['alpha']))]:
        assert np.array_equal(np.array(Image.open(EVAL/f'{prefix}-{suffix}.png')),arr)
    ys,xs=np.where((matte>.5)&valid);bbox=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]
    assert row['bbox']==bbox
    x0,y0,x1,y1=bbox;ym=y0+round(.6*(y1-y0))
    masks={'full_valid':valid,'foreground_matte_gt0_5':valid&(matte>.5)}
    boxes=[('foreground_union_bbox',bbox),('upper_union',[x0,y0,x1,ym]),('lower_union',[x0,ym,x1,y1])]
    for label,(a,b,cx,d) in boxes:
        m=np.zeros_like(valid);m[b:d,a:cx]=True;masks[label]=m&valid
    for label,m in masks.items():
        diff=pred[m].astype(np.float64)-ref[m].astype(np.float64)
        psnr=-10*np.log10(np.mean(diff**2));mae=np.abs(diff).mean();expected=row['metrics'][label]
        assert int(m.sum())==expected['pixels'] and abs(psnr-expected['psnr_db'])<1e-10 and abs(mae-expected['mae'])<1e-12
    # Compare native content of composed panels as well as full PNGs.
    for label,(a,b,cx,d) in [('full-person',bbox),('upper',[x0,y0,x1,ym])]:
        pane=np.array(Image.open(EVAL/f'{prefix}-{label}-comparison.png'));w=cx-a
        assert pane.shape==(d-b+38,2*w,3)
        assert np.array_equal(pane[38:,:w],u8(ref[b:d,a:cx])) and np.array_equal(pane[38:,w:],u8(pred[b:d,a:cx]))
    checks.append(dict(camera=name,status='fourfullPNGs_twoNativePanels_sources_and_five_region_metrics_exact',raw_sha256=digest(EVAL/f'{prefix}-raw.npz')))
result=dict(status='sixnative_outputs_and_references_verified',checks=checks,full_pngs=24,native_comparison_panels=12,
            scope='Independent CPU pixel and double-precision metric recomputation; common OpenCV rectification convention, not an independent camera-calibration algorithm.',goal_complete=False)
(EVAL/'independent-image-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
