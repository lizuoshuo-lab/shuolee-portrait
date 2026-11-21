"""N4 finite algebra audit of new processed-video intrinsics, not calibration.

No video/camera optimization, lens reprocessing, feature extraction or output
camera candidate. Evaluate explicit assumptions on the same normalized rays.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,signal,time,traceback
import cv2
import numpy as np
R=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
I=R/'video-update-20261027';P=I/'output-intrinsics-audit-plan.json';O=I/'output-intrinsics-audit'
Y=Path('/mnt/d/00/工作/数据集/mono_main_calib.yaml');D=Path('/mnt/d/00/工作/数据集/20261027-114356.jpg')
def read(p):return json.loads(p.read_text())
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def resize(sx,sy):return np.array([[sx,0,(sx-1)/2],[0,sy,(sy-1)/2],[0,0,1]],np.float64)
def crop(x,y):return np.array([[1,0,-x],[0,1,-y],[0,0,1]],np.float64)
def project(k,rays):
    p=(k@rays.T).T;return p[:,:2]/p[:,2:3]
def expired(*_):raise TimeoutError('N4 CPU budget exceeded')
def register():
    assert not P.exists() and not O.exists()
    paths=[Y,D,I/'content-review.json',I/'native-temporal-material/seeds.json',R/'pixel-convention-bridge/result.json']
    save(P,dict(stage='N4',registered=now(),kind='Read-only finite projection algebra; no new camera solve',source_sha256=sha(Path(__file__)),
        inputs={str(p):sha(p) for p in paths},
        hypothesis='A change in export resolution is harmless only if fullK and pixel origin follow the same resize/crop. The4000vs4096 ambiguity must not be silently decided by feature residuals.',
        fixed_assumptions=[
            'H0 nominal:first YAML K treated as rectified4000x3000K; height-resize to2704 then ideal centered1520crop',
            'H1 same physical H0 rays:resize4000x3000to4096x3072 WITH K, then resize to2704 and crop1520',
            'H2 conditional alternative:first YAML numeric K assigned unchanged to4096x3072rectifiedimage, then resize/crop',
            'H3 conditional old1080x1920K mapped by exact edge-aligned dimension ratio to1520x2704',
            'H4 dimension rounding:integer3605x2704 intermediate from4000x3000, centered crop needs left1042or1043',
            'H5 explicit mistake-control:compute crop from diagram1532width then apply to actual1520frame'],
        no_optimization=True,no_heldout_residuals=True,no_new_candidate_promotion=True,models='processed output pinhole only; no fisheye coefficients applied',
        samples='Fixed5x5 native image grid and the9 already registered face candidate pixel coordinates only as numerical ray samples, not physical matches or error evidence',
        controls=['H0/H1 commute to floating-point tolerance','H3 bridge identity to old/new pixel-size transform','half-pixel origin conversion consistency','crop typo control is6px horizontal','oldK and raw sources unchanged'],
        budget=dict(runs=1,CPU_seconds=30),
        conclusion_scope='Matrices and same-ray shifts are conditional algebra. None identifies actualK_rect, proves physical calibration, or changes training. Do not select by which gives a smaller landmark residual.'))
    print('N4_PLAN_REGISTERED')
def run():
    plan=read(P);assert plan['source_sha256']==sha(Path(__file__))
    assert not O.exists(),'N4 already exists; inspect only'
    for p,h in plan['inputs'].items():assert sha(Path(p))==h
    O.mkdir();save(O/'progress.json',dict(stage='running',pid=os.getpid(),updated=now()))
    signal.signal(signal.SIGALRM,expired);signal.alarm(plan['budget']['CPU_seconds']);t0=time.monotonic()
    try:
        f=cv2.FileStorage(str(Y),cv2.FILE_STORAGE_READ);k=f.getNode('camera_matrix').mat();kfov=f.getNode('camera_matrix_fov').mat()
        native=(int(f.getNode('image_width').real()),int(f.getNode('image_height').real()));f.release();assert native==(4000,3000)
        w,h=1520,2704;s=h/3000.;A0=crop((4000*s-w)/2,0)@resize(s,s);K0=A0@k
        A1=crop((4096*(h/3072)-w)/2,0)@resize(h/3072,h/3072)@resize(4096/4000,3072/3000)
        A2=crop((4096*(h/3072)-w)/2,0)@resize(h/3072,h/3072)
        old=next(x for x in read(R/'pixel-convention-bridge/result.json')['subjects'] if x['subject']=='portrait')
        oldk=np.asarray(old['conversion']['input_K']);A3=resize(w/1080,h/1920)
        variants=[dict(id='H0',description=plan['fixed_assumptions'][0],K=K0),
            dict(id='H1',description=plan['fixed_assumptions'][1],K=A1@k),
            dict(id='H2',description=plan['fixed_assumptions'][2],K=A2@k),
            dict(id='H3',description=plan['fixed_assumptions'][3],K=A3@oldk),
            dict(id='H4_left1042',description=plan['fixed_assumptions'][4],K=crop(1042,0)@resize(3605/4000,h/3000)@k),
            dict(id='H4_left1043',description=plan['fixed_assumptions'][4],K=crop(1043,0)@resize(3605/4000,h/3000)@k),
            dict(id='H5_wrong1532',description=plan['fixed_assumptions'][5],K=crop((4000*s-1532)/2,0)@resize(s,s)@k)]
        grid=np.array([(x,y) for y in np.linspace(0,h-1,5) for x in np.linspace(0,w-1,5)])
        face=np.array([x['uv'] for x in read(I/'native-temporal-material/seeds.json')['seeds'] if x['group']=='face_candidate'],np.float64)
        output=[]
        for v in variants:
            item=dict(id=v['id'],assumption=v['description'],K=v['K'].tolist(),physical_accuracy_accepted=False)
            for label,uv in [('grid',grid),('face_pixel_samples',face)]:
                rays=(np.linalg.inv(K0)@np.c_[uv,np.ones(len(uv))].T).T
                delta=project(v['K'],rays)-uv;dist=np.linalg.norm(delta,axis=-1)
                item[label]=dict(sample_count=len(uv),same_ray_displacement_max_px=float(dist.max()),same_ray_displacement_min_px=float(dist.min()),
                    delta_xy=delta.tolist(),reference_pixels=uv.tolist(),actual_reprojection_error=False)
            output.append(item)
        checks=[]
        def check(name,passed):
            checks.append(dict(name=name,passed=bool(passed)))
            if not passed:raise AssertionError(name)
        check('H0 and properly rescaled4096K equivalent',np.allclose(A1@k,K0,rtol=0,atol=1e-10))
        check('H3 edge-aligned native rays preserved relative oldK',np.allclose(np.linalg.inv(A3@oldk)@A3,np.linalg.inv(oldk),rtol=0,atol=1e-12))
        check('wrong diagram crop1532 gives exactly6px shift',np.allclose(np.array(output[-1]['grid']['delta_xy']),[6.,0.],rtol=0,atol=1e-10))
        T=np.eye(3);T[:2,2]=.5;rays=(np.linalg.inv(K0)@np.c_[grid,np.ones(len(grid))].T).T
        check('integer and pixel-corner conventions commute',np.max(np.abs(project(T@K0,rays)-(grid+.5)))<1e-10)
        check('all source bytes unchanged',all(sha(Path(p))==h for p,h in plan['inputs'].items()))
        result=dict(stage='N4',state='completed_conditional_algebra_actual_K_unidentified',updated=now(),passed=True,checks=checks,variants=output,
            nominal_contract=dict(width=w,height=h,pixel_convention='integer_centers',projection='pinhole',
                source_YAML_dimensions=list(native),K_rect_assumption='firstcamera_matrix treated as rectified4000x3000K; user workflow alone does not establish actualK_rect',
                ideal_resized_width=4000*s,ideal_crop_left=(4000*s-w)/2,K=K0.tolist(),training_admitted=False),
            unknowns=['Actual rectified output K before resize','Whether4096raster uses numerically unchanged or scaledK','Intermediate rounding, integer crop offsets, resize pixel convention',
                'Whether all newly supplied people use the same exact camera export configuration'],
            original_fov_K_only_recorded=kfov.tolist(),second_fisheye_undistortion=False,conditional_K_exported_to_training=False,
            camera_fit_performed=False,features_changed=False,elapsed_seconds=time.monotonic()-t0)
        save(O/'result.json',result);save(O/'progress.json',dict(stage='completed',pid=None,updated=now()))
        print(json.dumps(dict(state=result['state'],K0=K0.tolist(),displacements=[dict(id=x['id'],gridmax=x['grid']['same_ray_displacement_max_px'],facemax=x['face_pixel_samples']['same_ray_displacement_max_px']) for x in output],training_admitted=False)))
    except BaseException as e:
        save(O/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),updated=now()));save(O/'progress.json',dict(stage='failed',pid=None,updated=now()));raise
    finally:signal.alarm(0)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--register',action='store_true');a=p.parse_args();register() if a.register else run()
