"""Bounded CPU tests of the new adapter, not a subject fit or training run."""
from pathlib import Path
import ast
import datetime as dt
import hashlib
import json
import time
import cv2
import numpy as np
from PIL import Image
import torch
from hsfm116_adapter import *

LAB = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
B = LAB/'outputs/mini-splatting2-mip-depth-001/shuolee/current'
H = B/'portrait-alignment-open-source-116/hsfm'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name, data):
    p=H/name; t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');t.replace(p)
def extract(path, names, ns):
    tree=ast.parse(path.read_text())
    funcs=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names]
    assert {x.name for x in funcs} == set(names)
    exec(compile(ast.Module(body=funcs,type_ignores=[]),str(path),'exec'),ns)
    return ns

def main():
    assert not (H/'adapter-check.json').exists(), 'Do not rerun unchanged fixtures'
    t0=time.monotonic(); now=dt.datetime.now(dt.timezone.utc)
    files=[Path(__file__),Path(__file__).with_name('hsfm116_adapter.py'),
           H/'forks/dust3r/utils/image.py',H/'forks/dust3r/utils/geometry.py',
           H/'forks/dust3r/cloud_opt/optimizer.py',
           B/'full-scene-localization-074/cameras.json',
           B/'avatar-repository-pair-114/smplx/data/training-inputs.json']
    before={str(p):sha(p) for p in files}
    save('adapter-check-protocol.json',dict(created=now.isoformat(),scope='CPU interface fixtures only',
        source_sha256=before,previous_subject_fit_loaded=False,gaussian_training=False,
        cases=['PIL resize/crop pixel ramps with actual portrait dimensions',
               'nonidentity 175-view OpenCV projection and resized per-view K',
               'fork native depth backprojection and pose convention',
               'perspective translation with noncoplanar joints, anisotropic K and non-pelvis first row',
               'bbox translation and scale invariance, invalid-input rejection'],
        stop_seconds=120,pass_threshold_raw_pixel_ramp=.02,pass_threshold_projection_px=1e-7))
    results={}
    tx=dust3r_raster_transform((1520,2704))
    resize_fn=extract(H/'forks/dust3r/utils/image.py',['_resize_pil_image'],dict(PIL=__import__('PIL')))['_resize_pil_image']
    import PIL.Image
    ramps=[]
    for axis in (0,1):
        arr=np.indices((2704,1520),dtype=np.float32)[1-axis]
        im=resize_fn(Image.fromarray(arr),512).crop(tx.crop_xyxy)
        actual=np.asarray(im); yy,xx=np.indices(actual.shape)
        expected=map_pixels(np.stack((xx,yy),-1),np.linalg.inv(tx.centers))[...,axis]
        error=float(abs(actual[5:-5,5:-5]-expected[5:-5,5:-5]).max())
        assert error<.02
        ramps.append(error)
    results['pil_ramps']=dict(original_wh=tx.original_wh,output_wh=tx.output_wh,
        max_raw_px_by_axis=ramps,raw_center_to_processed=tx.centers.tolist(),
        note='Actual pinned PIL resize helper, independent float intensity ramps away from padding')

    # Random nonidentity cameras come from the175 immutable observed-view records;
    # points are synthetic in front of each camera, not learned subject geometry.
    cameras=json.loads((B/'full-scene-localization-074/cameras.json').read_text())
    rows=cameras['records']; assert len(rows)==175
    f,cx,cy,k=cameras['intrinsics']; K=colmap_to_raster_K([[f,0,cx],[0,f,cy],[0,0,1]])
    rng=np.random.default_rng(116); errors=[]; reversals=[]
    for row in rows:
        W=np.asarray(row['w2c']); C=invert_pose(W)
        Xc=np.column_stack((rng.uniform(-.6,.6,24),rng.uniform(-1,1,24),rng.uniform(2,5,24)))
        X=Xc@C[:3,:3].T+C[:3,3]
        rvec=cv2.Rodrigues(W[:3,:3])[0]
        reference=cv2.projectPoints(X,rvec,W[:3,3],K,None)[0].reshape(-1,2)
        processed,_=project_pinhole(X,tx.intrinsics(K),C)
        back=map_pixels(processed,np.linalg.inv(tx.centers))
        errors.append(float(abs(back-reference).max()))
        reversals.append(float(abs(invert_pose(C)-W).max()))
    assert max(errors)<1e-7 and max(reversals)<1e-9
    results['actual_175_camera_contract']=dict(count=175,max_opencv_projection_difference_raw_px=max(errors),
        max_se3_inverse_error=max(reversals),physical_calibration_verified=False,
        distortion_note='This fixture tests pinhole conversion only. SIMPLE_RADIAL k must not be silently discarded for raw image use.')

    ns=extract(H/'forks/dust3r/utils/geometry.py',['xy_grid'],dict(np=np,torch=torch))
    ns=extract(H/'forks/dust3r/cloud_opt/optimizer.py',['_fast_depthmap_to_pts3d'],ns)
    grid=ns['xy_grid'](17,21,device='cpu').reshape(1,-1,2)
    depth=torch.ones((1,17*21),dtype=torch.float64)*3.2
    focal=torch.tensor([[411.]],dtype=torch.float64); pp=torch.tensor([[8.3,9.6]],dtype=torch.float64)
    xyz=ns['_fast_depthmap_to_pts3d'](depth,grid,focal,pp)
    uv=xyz[...,:2]/xyz[...,2:]*focal[:,:,None]+pp[:,None]
    err=float((uv-grid).abs().max());assert err<1e-10
    results['native_fork_grid']=dict(max_backprojection_reprojection_px=err,origin='integer raster centers',
        c2w=True,single_focal_only=True,fx_fy_of_resized074=tx.intrinsics(K).diagonal()[:2].tolist(),
        anisotropy_must_be_explicit=True)

    rel=rng.normal(size=(20,3))*[.25,.55,.12]
    rel[0]=[.02,-.65,.08] # nose-like landmark, deliberately not root
    root=np.array([.16,-.09,3.7]); K2=np.array([[760.,0.,161.3],[0.,815.,248.9],[0.,0.,1.]])
    obs=cv2.projectPoints(rel,np.zeros(3),root,K2,None)[0].reshape(-1,2)
    confidence=np.ones(20);obs[-1]=[9999,-9999];confidence[-1]=0
    fitted, info=estimate_root_translation(rel,obs,K2,confidence)
    err=float(abs(fitted-root).max());assert err<1e-10
    results['translation_initializer']=dict(max_root_error=err,accepted_joints=info['used'],
        source_first_joint_is_not_root=True,noncoplanar=True,anisotropic_K=True,
        note='Synthetic initializer only; human prior accuracy not established')
    boxes=np.array([[10,20,110,220],[110,120,210,320]],float)
    assert np.all(bbox_area_xyxy(boxes)==20000)
    a=map_pixels(boxes.reshape(-1,2),tx.edges).reshape(-1,4)
    expected=20000*np.linalg.det(tx.edges[:2,:2])
    assert np.max(abs(bbox_area_xyxy(a)-expected))<1e-9
    invalid_count=0
    for fn,args in [(bbox_area_xyxy,([1,2,1,3],)),
                    (estimate_root_translation,(rel[:3],obs[:3],K2,confidence[:3])),
                    (strict_name_mapping,(['nose','nose'],['nose']))]:
        try:fn(*args)
        except ValueError:invalid_count+=1
    assert invalid_count==3
    results['box_and_rejection']=dict(equal_area=True,affine_area_error=float(max(abs(bbox_area_xyxy(a)-expected))),
        invalid_cases_rejected=invalid_count)
    assert before=={str(p):sha(p) for p in files}
    assert time.monotonic()-t0<120
    save('adapter-check.json',dict(updated=dt.datetime.now(dt.timezone.utc).isoformat(),
        state='passed_cpu_interface_only',results=results,seconds=time.monotonic()-t0,
        source_unchanged=True,candidate_calibration_accepted=False,subject_fit=False,
        new_weights=False,gpu_used=False,root_cause_resolved=False))
    print(json.dumps(dict(state='passed_cpu_interface_only',seconds=time.monotonic()-t0,results=results)),flush=True)

if __name__=='__main__':main()
