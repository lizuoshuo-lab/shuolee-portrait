"""Once-only registered synthetic output-residual integration and ray oracle."""
from pathlib import Path
from datetime import datetime,timezone
import copy,json,hashlib,time,os,signal,traceback
import numpy as np
O=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/native-residual-camera')
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def linux(p):
    p=p.replace('\\','/')
    return Path('/mnt/'+p[0].lower()+p[2:])

def main():
    if (O/'progress.json').exists(): raise FileExistsError('Registered run already exists; no repeat')
    plan=json.loads((O/'plan.json').read_text(encoding='utf-8'))
    for p,h in plan['source_hashes'].items():
        if sha(linux(p))!=h:raise RuntimeError('Registered source changed: '+p)
    start=time.monotonic();checks=[]
    def progress(stage):save(O/'progress.json',dict(state='running',stage=stage,pid=os.getpid(),updated=now()))
    def check(name,ok,**detail):
        checks.append(dict(name=name,passed=bool(ok),**detail));save(O/'checks.json',checks)
        if not ok:raise AssertionError(name)
    def reject(name,fn):
        try:fn()
        except (ValueError,RuntimeError,KeyError) as e:check(name,True,error=str(e))
        else:check(name,False)
    def timeout(*_):raise TimeoutError('Registered180second budget')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(plan['budget']['seconds'])
    progress('cpu_camera_controls')
    try:
        import cv2
        from scipy.spatial.transform import Rotation
        from native_residual_camera122 import (ResidualRenderContract,projection,with_projection,
            validate_camera,cv_coefficients,distort,rays_world,project_world,sampled_domain_check)
        f=plan['fixture'];rng=np.random.default_rng(f['seed']);w,h=plan['budget']['images']
        lens=projection(f['known_radial'],f['known_tangential'],f['known_prism'],input_domain='synthetic_known_camera')
        k=np.array([[68.,0,43.2],[0,70.,37.1],[0,0,1.]])
        cameras=[]
        for i in range(2):
            e=np.eye(4);e[:3,:3]=Rotation.from_rotvec([.025*i,-.08*i,.02*i]).as_matrix();e[:3,3]=[.10*i,-.04*i,.015*i]
            kk=k.copy();kk[0,2]+=2.3*i;kk[1,2]-=1.2*i
            p=copy.deepcopy(lens)
            if i:p['radial']=[v*.6 for v in p['radial']];p['tangential']=[-.002,.001];p['thin_prism']=[-.0004,.0001,.0006,-.0001]
            cameras.append(with_projection(dict(id=str(i),image_name=f'known_{i}.png',width=w,height=h,K=kk.tolist(),w2c=e.tolist()),p))
        save(O/'cameras-fixture.json',cameras)
        xyz=rng.uniform([-.9,-.8,1.5],[.9,.8,3.0],size=(512,3));errors=[];inverse=[]
        for c in cameras:
            e=np.array(c['w2c']);K=np.array(c['K']);d=cv_coefficients(c['residual_projection'])
            expected=cv2.projectPoints(xyz,cv2.Rodrigues(e[:3,:3])[0],e[:3,3],K,d)[0][:,0]
            uv=project_world(xyz,c);errors.append(float(np.max(np.abs(uv-expected))))
            origin,rays=rays_world(uv,c);want=xyz-origin;want/=np.linalg.norm(want,axis=-1,keepdims=True)
            inverse.append(float(np.max(np.abs(rays-want))))
            sampled_domain_check(c)
        check('Two complete K/E/D project against independent OpenCV',max(errors)<plan['acceptance']['cpu_projection_max_px'],max_abs_px=max(errors))
        check('Two inverse rays retain world pose and subpixel principal point',max(inverse)<plan['acceptance']['inverse_ray_max'],max_abs=max(inverse))
        q=rng.uniform(-.7,.7,size=(100,2));_,j=distort(q,lens,jacobian=True);fd=np.zeros_like(j)
        for a in range(2):
            delta=np.zeros_like(q);delta[:,a]=1e-6
            fd[:,:,a]=(distort(q+delta,lens)-distort(q-delta,lens))/2e-6
        jacerr=float(np.max(np.abs(fd-j)))
        check('Analytic normalized projection Jacobian',jacerr<plan['acceptance']['jacobian_max'],max_abs=jacerr)
        for key in ['radial','tangential','thin_prism','mapping','units','pixels']:
            bad=copy.deepcopy(cameras[0]);del bad['residual_projection'][key]
            reject('Missing '+key+' is not a zero/default',lambda bad=bad:validate_camera(bad))
        bad=copy.deepcopy(cameras[0]);bad['residual_projection']['radial']=[0]*8
        reject('Vendor eight values cannot enter Brown6 radial slot',lambda:validate_camera(bad))
        bad=copy.deepcopy(cameras[0]);bad['residual_projection']['input_domain']='native_fisheye'
        reject('Raw fisheye cannot enter output-residual contract',lambda:validate_camera(bad))
        bad=copy.deepcopy(cameras[0]);bad['K'][0][1]=.01
        reject('K skew is not silently removed',lambda:validate_camera(bad))
        bad=copy.deepcopy(cameras[0]);bad['residual_projection']['radial']=[-20,0,0,0,0,0]
        reject('Noninvertible residual candidate rejected before render',lambda:sampled_domain_check(bad))
        save(O/'cpu-result.json',dict(checks=checks.copy(),passed=True,physical_calibration=False))
        progress('cuda_independent_ray_oracle')
        import torch
        from PIL import Image
        from native_portrait_renderer122_v3 import render_cameras,render_native
        from native_portrait_renderer122_v2 import render_native as old_render,NativeCameraContractV2
        from native_training122_v3 import FreshNativeGaussians,create_view,render_view,accumulate_update,export_native_checkpoint
        from native_training122 import foreground_objective
        from native_scene_package122_v4 import load_scene,render_recorded
        from native_scene_package122_v3 import load_scene as old_load
        t=lambda v:torch.as_tensor(v,dtype=torch.float32,device='cuda')
        contract=ResidualRenderContract('synthetic_known_camera')
        def render(p,cs):return render_cameras({n:t(v) for n,v in p.items()},cs,contract=contract,sh_degree=3)
        def png(name,image):Image.fromarray((np.clip(image,0,1)*255+.5).astype(np.uint8)).save(O/(name+'.png'))
        qxyzw=Rotation.from_rotvec([.3,-.2,.15]).as_quat();quat=np.r_[qxyzw[3],qxyzw[:3]].astype('float32')
        sh=np.zeros((1,16,3),'float32');sh[:,0]=(np.array([.3,.6,.8])-.5)/.28209479177387814
        one=dict(means=np.array([[.15,-.08,2.]],'float32'),quats=quat[None],scales=np.array([[.06,.10,.045]],'float32'),opacities=np.array([.65],'float32'),sh=sh)
        with torch.no_grad():rgb,alpha,meta=render(one,cameras[:1])
        c=cameras[0];yy,xx=np.mgrid[:h,:w];pixels=np.column_stack([xx.ravel()+.5,yy.ravel()+.5])
        # Independent cv2 inverse, not the implementation's inverse/Jacobian.
        norm=cv2.undistortPointsIter(pixels[:,None,:],np.array(c['K']),cv_coefficients(c['residual_projection']),None,None,
            (cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,60,1e-12))[:,0]
        ray=np.column_stack([norm,np.ones(len(norm))]);ray/=np.linalg.norm(ray,axis=-1,keepdims=True)
        rot=Rotation.from_quat(qxyzw).as_matrix();precision=rot@np.diag(1/one['scales'][0].astype(float)**2)@rot.T
        m=one['means'][0].astype(float);distance=(ray@precision@m)/np.einsum('ij,jk,ik->i',ray,precision,ray)
        delta=ray*distance[:,None]-m;power=-.5*np.einsum('ij,jk,ik->i',delta,precision,delta)
        oracle=np.minimum(.999,float(one['opacities'][0])*np.exp(power));oracle[oracle<1/255]=0
        oracle=oracle.reshape(h,w,1);oracle_rgb=oracle*np.array([.3,.6,.8])
        err=max(float(np.max(np.abs(alpha[0].cpu().numpy()-oracle))),float(np.max(np.abs(rgb[0].cpu().numpy()-oracle_rgb))))
        check('Actual Brown CUDA rays against independent full-covariance oracle',err<plan['acceptance']['independent_ray_rgb_alpha_max'],max_abs_rgb_alpha=err)
        png('oracle',oracle_rgb);png('cuda',rgb[0].cpu().numpy());png('oracle-difference-x1000',np.abs(rgb[0].cpu().numpy()-oracle_rgb)*1000)
        zero=copy.deepcopy(cameras[0]);zero['residual_projection']=projection([0]*6,[0]*2,[0]*4,input_domain='synthetic_known_camera')
        with torch.no_grad():
            zr,za,_=render(one,[zero])
            oldr,olda,_=old_render(t(one['means']),t(one['quats']),t(one['scales']),t(one['opacities']),t(one['sh']),t([zero['w2c']]),t([zero['K']]),w,h,contract=NativeCameraContractV2('pinhole','synthetic_known_camera'),sh_degree=3)
        diff=max(float((zr-oldr).abs().max()),float((za-olda).abs().max()))
        check('Explicit zero coefficients preserve v2 native response exactly',diff==0,max_abs=diff)
        p={n:t(v) for n,v in one.items()};rr=t([lens['radial']]).requires_grad_(True)
        reject('Trainable distortion is explicitly refused',lambda:render_native(p['means'],p['quats'],p['scales'],p['opacities'],p['sh'],t([c['w2c']]),t([c['K']]),w,h,contract=contract,sh_degree=3,radial_coeffs=rr,tangential_coeffs=t([lens['tangential']]),thin_prism_coeffs=t([lens['thin_prism']])))
        progress('fresh_training_with_two_recorded_lenses')
        n=plan['budget']['points'];teacher=dict(means=rng.uniform([-.7,-.55,1.9],[.7,.55,2.2],(n,3)).astype('float32'),
            quats=rng.normal(size=(n,4)).astype('float32'),scales=rng.uniform(.05,.10,(n,3)).astype('float32'),
            opacities=np.full(n,.7,'float32'),sh=rng.normal(0,.015,(n,16,3)).astype('float32'))
        teacher['quats']/=np.linalg.norm(teacher['quats'],axis=-1,keepdims=True)
        teacher['sh'][:,0]=(rng.uniform(.2,.8,(n,3))-.5)/.28209479177387814
        views=[]
        with torch.no_grad():
            for c in cameras:
                im,a,_=render(teacher,[c]);im=im[0].cpu().numpy();a=a[0].cpu().numpy()
                views.append(create_view(c,im,a,a>.02,np.ones_like(a,dtype=bool),device='cuda'))
        initial={n:v.copy() for n,v in teacher.items()};initial['means'][:,0]+=.02;initial['sh'][:,0]+=.04
        model=FreshNativeGaussians(initial,device='cuda')
        rates=dict(means=.001,quats=.0002,log_scales=.0005,opacity_logits=.001,sh=.003)
        optimizer=model.optimizer(rates)
        check('Fresh free Gaussian optimizer begins empty',len(optimizer.state)==0)
        def objective():
            with torch.no_grad():return float(torch.stack([foreground_objective(*render_view(model,v,contract)[:2],v)[0] for v in views]).mean())
        before=objective();history=[];camera_before=copy.deepcopy(cameras)
        for step in range(plan['budget']['synthetic_steps']):
            row=accumulate_update(model,views,contract,optimizer);history.append(dict(step=step+1,**row))
        after=objective();save(O/'training-history.json',history)
        check('Native foreground training decreases loss with two fixed distinct lenses',after<before and camera_before==cameras,before=before,after=after,steps=len(history),points=n)
        reject('Second optimizer cannot resume updated parameters',lambda:model.optimizer(rates))
        progress('exact_native_package_replay')
        path=export_native_checkpoint(O/'native-package',model,views,contract,scope='synthetic-control-only')
        scene=load_scene(path);linear=model.numpy_linear()
        check('Canonical native arrays remain bit-identical after export',all(np.array_equal(linear[n],scene.parameters[n]) for n in linear))
        check('Every native K/E/D record remains exact',list(scene.cameras.values())==cameras)
        replay=[];pictures=[]
        for i,v in enumerate(views):
            with torch.no_grad():live,la,_=render_view(model,v,contract)
            im,a,receipt=render_recorded(scene,v.camera['id']);replay.append(max(float(np.abs(im-live.cpu().numpy()).max()),float(np.abs(a-la.cpu().numpy()).max())))
            png('trained-camera'+str(i),im);pictures.append(im)
            np.savez_compressed(O/('camera'+str(i)+'-native-floats.npz'),rgb=im,alpha=a)
        check('Saved native package reproduces both trained responses exactly',max(replay)==0,max_abs=max(replay))
        reject('Legacy v3 reader does not silently discard output residuals',lambda:old_load(path))
        m=json.loads(path.read_text());bad=copy.deepcopy(m);bad['cameras']['sha256']='0'*64
        save(path.parent/'wrong-hash.json',bad);reject('Mismatched camera SHA rejected',lambda:load_scene(path.parent/'wrong-hash.json'))
        missing=copy.deepcopy(cameras);del missing[0]['residual_projection']['tangential'];save(path.parent/'missing-D.json',missing)
        bad=copy.deepcopy(m);bad['cameras']=dict(path='missing-D.json',sha256=sha(path.parent/'missing-D.json'));save(path.parent/'missing-D-scene.json',bad)
        reject('Missing coefficients rejected even with a matching file hash',lambda:load_scene(path.parent/'missing-D-scene.json'))
        for p,digest in plan['preserved_live_sources'].items():
            if sha(linux(p))!=digest:raise RuntimeError('Live legacy source mutated')
        check('Existing live v1/v2/v3 source hashes preserved',True)
        grid=Image.new('RGB',(w*3,h*2),'white')
        for col,name in enumerate(['oracle','cuda','oracle-difference-x1000']):grid.paste(Image.open(O/(name+'.png')),(col*w,0))
        for col in range(2):grid.paste(Image.open(O/('trained-camera'+str(col)+'.png')),(col*w,h))
        grid.save(O/'residual-control.png')
        save(O/'figure-index.json',dict(native_tile=[w,h],text_on_image=False,rescaled=False,
            rows=[['independent anisotropic Brown-ray oracle','native CUDA response','absolute RGB difference x1000'],['trained camera0','trained camera1','unused white cell']],synthetic_only=True))
        result=dict(state='completed',updated=now(),seconds=time.monotonic()-start,checks=checks,
            synthetic_only=True,real_coefficients_fitted=False,human_training_started=False,physical_quality_accepted=False,
            camera_optimization=False,native_package=str(path),legacy_sources_unchanged=True,
            training=dict(points=n,steps=len(history),before=before,after=after),
            scopes_not_verified=['real camera residual coefficients','global invertibility for arbitrary coefficients','full native resolution residual capacity','new residual interactive UI','real person admission'])
        save(O/'result.json',result);save(O/'progress.json',dict(state='completed',pid=os.getpid(),updated=now()))
        print(json.dumps(dict(state='completed',checks=len(checks),seconds=result['seconds'],before=before,after=after)))
    except Exception as e:
        save(O/'failure.json',dict(updated=now(),error=repr(e),traceback=traceback.format_exc(),checks=checks))
        save(O/'progress.json',dict(state='failed',pid=os.getpid(),updated=now()));raise
    finally:signal.alarm(0)

if __name__=='__main__':main()
