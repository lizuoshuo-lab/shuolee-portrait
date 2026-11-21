"""Bounded pose-conditioned DA3-Base diagnostic on nine training views.

No Gaussian training and no commercial-company deployment. The selected model
card is Apache-2.0. Sparse agreement remains development-only, not depth GT.
"""
import json
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import least_squares

from shuolee_geometry_diagnosis import LAB, BASE, ROOT, OUT, sha, write_json, sample, stats

CODE = ROOT/'shuolee/code/depth-anything-3'
WEIGHTS = ROOT/'shuolee/data/weights/da3-base'
sys.path[:0]=['/opt/shuolee/da3-extra',str(CODE/'src')]
from depth_anything_3.api import DepthAnything3


def align_inverse(predicted_z, target_z, fit):
    x=1/np.maximum(predicted_z,1e-6);y=1/target_z
    xm,xs=np.median(x[fit]),np.std(x[fit]);ym,ys=np.median(y[fit]),np.std(y[fit])
    xx=(x-xm)/max(xs,1e-8);yy=(y-ym)/max(ys,1e-8)
    initial=np.polyfit(xx[fit],yy[fit],1)
    result=least_squares(lambda ab:ab[0]*xx[fit]+ab[1]-yy[fit],initial,loss='huber',f_scale=.3)
    a=result.x[0]*ys/max(xs,1e-8);b=result.x[1]*ys+ym-a*xm
    inverse=a*x+b
    assert a>0 and np.all(inverse>0), 'Reject inverted/invalid calibrated prior'
    return 1/inverse,dict(a=float(a),b=float(b),fitting_points=int(fit.sum()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--all-training',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4);cv2.setNumThreads(2)
    prefix='da3-full' if args.all_training else 'da3'
    output=OUT/(prefix+'-pilot.json');assert not output.exists()
    frames=[i for i in range(59) if i%8] if args.all_training else [1,7,14,21,28,35,42,49,58]
    cameras=sorted(json.loads((BASE/'viewer/cameras.json').read_text()),key=lambda c:c['img_name'])
    cams=[c for c in cameras if int(c['img_name']) in frames]
    assert len(cams)==len(frames) and all(int(c['img_name'])%8!=0 for c in cams)
    inputs=[BASE/f'hover-sfm-001/dataset/images/{c["img_name"]}.jpg' for c in cams]
    poses=[];intrinsics=[]
    for c in cams:
        R=np.asarray(c['rotation']);C=np.asarray(c['position'])
        v=np.eye(4);v[:3,:3]=R.T;v[:3,3]=-R.T@C;poses.append(v)
        intrinsics.append([[c['fx'],0,c['width']/2],[0,c['fy'],c['height']/2],[0,0,1]])
    weights_meta=json.loads((WEIGHTS/'download.json').read_text())
    for f in weights_meta['files']:assert sha(WEIGHTS/f['name'])==f['sha256']
    protocol=dict(owner='shuolee',created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        code_revision='3d835ec1a5802d64a8b8b15f817a1ab54809bfe4',weights=weights_meta,
        frames=[c['img_name'] for c in cams],input_sha256={p.name:sha(p) for p in inputs},
        resolution=504,process_res_method='upper_bound_resize',pose_conditioned=True,align_to_input_ext_scale=True,
        gaussian_head=False,gaussian_training=False,source_sha256=sha(Path(__file__)),
        comparison='DA2 original depth vs DA3 camera-scale prediction and separate robust inverse-depth alignment using only nonreserved points.',
        domain='Same robust triangulated points and training cameras; compare both all source-valid points and originalDA2 confidence>0.85.',
        selection='All51training frames' if args.all_training else 'Existing nine head-annotation anchor frames, fixed before DA3 inference.',
        scope='Development geometry agreement, not independent geometry accuracy or final perceptual acceptance.')
    write_json(OUT/(prefix+'-protocol.json'),protocol)
    start=time.time();torch.manual_seed(0);torch.cuda.reset_peak_memory_stats()
    torch.cuda.set_per_process_memory_fraction(.82)
    model=DepthAnything3.from_pretrained(str(WEIGHTS),local_files_only=True).to('cuda').eval()
    with torch.inference_mode():
        prediction=model.inference([str(p) for p in inputs],extrinsics=np.array(poses,dtype='float32'),
            intrinsics=np.array(intrinsics,dtype='float32'),align_to_input_ext_scale=True,
            process_res=504,process_res_method='upper_bound_resize',infer_gs=False)
    torch.cuda.synchronize()
    elapsed=time.time()-start;peak=torch.cuda.max_memory_allocated()/1024**2
    print('DA3 inference',prediction.depth.shape,elapsed,'seconds',peak,'MiB',flush=True)
    assert np.isfinite(prediction.depth).all() and (prediction.depth>0).all()
    assert np.allclose(prediction.extrinsics,np.array(poses)[:,:3],atol=1e-5)
    cache_type='float32' if args.all_training else 'float16'
    np.savez_compressed(OUT/(prefix+'-predictions.npz'),depth=prediction.depth.astype(cache_type),
        confidence=prediction.conf.astype(cache_type),intrinsics=prediction.intrinsics,
        extrinsics=prediction.extrinsics,frames=np.array(protocol['frames']))
    obs=json.loads((OUT/'sparse-observations.json').read_text());rows=[];fitting=[]
    for i,c in enumerate(cams):
        selected=[r for r in obs if r['frame']==c['img_name']]
        xy=np.asarray([r['xy'] for r in selected]);k=np.array(intrinsics[i]);pk=prediction.intrinsics[i]
        pxy=np.stack([(xy[:,0]-k[0,2])/k[0,0]*pk[0,0]+pk[0,2],
                      (xy[:,1]-k[1,2])/k[1,1]*pk[1,1]+pk[1,2]],axis=-1)
        native=sample(prediction.depth[i],pxy)
        target=np.array([r['z'] for r in selected])
        fit=np.array([not r['prior_fit_reserved'] for r in selected])
        assert fit.sum()>=80 and (~fit).sum()>=20
        corrected,record=align_inverse(native,target,fit);record['frame']=c['img_name'];fitting.append(record)
        for j,r in enumerate(selected):
            if not r['prior_fit_reserved']:continue
            rows.append(dict(frame=c['img_name'],point_id=r['point_id'],head_crop=r['head_crop'],
                original_high_confidence=r['confidence']>.85,
                da2_error=r['relative_depth_error'],da3_camera_scale_error=float(abs(native[j]/target[j]-1)),
                da3_aligned_error=float(abs(corrected[j]/target[j]-1))))
    aggregate=[]
    for region in ['head_crop','outside_head_crop']:
        for common in [False,True]:
            rr=[r for r in rows if r['head_crop']==(region=='head_crop') and (not common or r['original_high_confidence'])]
            aggregate.append(dict(region=region,original_high_confidence_only=common,observations=len(rr),
                unique_points=len({r['point_id'] for r in rr}),
                **{key:stats([r[key] for r in rr]) for key in ['da2_error','da3_camera_scale_error','da3_aligned_error']}))
    result=dict(status='complete',protocol=protocol,inference_seconds=elapsed,peak_allocated_mib=peak,
        device=torch.cuda.get_device_name(0),prediction_resolution=list(prediction.depth.shape),
        aggregate=aggregate,alignment=fitting,rows=rows,trained_model_improved=False,
        next_gate='Dense boundary/occlusion and cross-view agreement before using as training supervision.')
    write_json(output,result);print(json.dumps(aggregate,indent=2),flush=True)


if __name__=='__main__':main()
