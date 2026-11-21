"""One registered repair validation; no new person, camera fit or old run resume."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,copy,hashlib,json,os,signal,subprocess,sys,time,traceback
import numpy as np
H=Path(__file__).parent
P='C:/' if os.name=='nt' else '/mnt/c/'
R=Path(P+'Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
G=Path(P+'Users/Administrator/Desktop/GS-SLAM-Lab/repos/gsplat')
O=R/'native-renderer-v2'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()

def register():
    assert read(R/'queue-state.json')['running'] is None
    previous=R/'native-training-integration/subpixel-culling-diagnosis/result.json'
    assert read(previous)['all_controls_passed']
    O.mkdir(exist_ok=False)
    names=['native_portrait_renderer122_v2.py','native_scene_package122_v2.py','native_training122_v2.py','probe_native_v2_122.py','run_native_v2_122.sh']
    preserved=['native_portrait_renderer122.py','native_scene_package122.py','native_gaussian_ply122.py','native_training122.py','native_pixel_convention122.py','probe_native_culling122.py']
    sources=['gsplat/rendering.py','gsplat/cuda/_wrapper.py','gsplat/cuda/csrc/ProjectionUT3DGSFused.cu','gsplat/cuda/csrc/RasterizeToPixelsFromWorld3DGSFwd.cu','gsplat/cuda/csrc/RasterizeToPixelsFromWorld3DGSBwd.cu']
    inputs=['covariance-sh-ply/original-parameters.npz','covariance-sh-ply/plan.json','native-training-integration/subpixel-culling-diagnosis/frozen-large-arrays.npz',
            'native-training-integration/subpixel-culling-diagnosis/camera.json','native-training-integration/existing-backend-path/fresh-initial-arrays.npz',
            'native-training-integration/existing-backend-path/known-synthetic-target.npz','native-training-integration/existing-backend-path/final-native-package/cameras.json',
            'native-training-integration/plan.json']
    plan=dict(stage='T1',registered=now(),prior_failure_sha256=sha(previous),
        intervention='New explicit v2 adapter calls existing UT projection with opacity=None, retains eps2d=.3,3.33-sigma geometric bounds and original opacity in the unchanged ray-3D kernel. Versioned package/loader and training component use the same adapter.',
        scope='Necessary repair of demonstrated implementation error. No physical K/pose change, human training, data warp, old model continuation, weights or live viewer replacement.',
        preservation='Legacy v1 source and scene contracts remain unchanged. New and old loaders must reject the other schema. Kernel source-tree hash is recorded. UT geometric bounds remain approximate.',
        controls=['Same frozen200k64x64 failure fixture compared bitwise to T0b uncull diagnostic; one fresh Adam step, then nativePLY/camera readback.',
            'Existing6anisotropicGaussians,3full-K cameras,SH3 rendered by new code for pinhole/KB4. Compare with all-Gaussians/all-tiles ray evaluation to detect missed support, and package roundtrip. No scene optimization.',
            'One anisotropic96x96 Gaussian: six selected raw-parameter central differences per projection, fixed epsilon1e-3; no threshold/epsilon sweep.',
            'Existing96point synthetic initial arrays, new empty optimizer, same4recordedcameras,12updates only (3perview); fresh stop-hook2of5 and no1000withoutreview.',
            'Explicit invalid camera/domain/coefficients/SH and incompatible package/source/hash controls. No camera interpolation or hidden defaults.',
            'After GPU driver exits, run new CLI sequentially for onepinhole and oneKB4 recordedcamera; compare savedfloatRGB/alpha to driver. No concurrentGPUjobs.'],
        limits=dict(wall_seconds=300,large_gradient_count=30681,primitive_parity_max_abs=3e-6,all_tiles_reference_max_abs=3e-6,
            small_roundtrip_max_abs=3e-6,large_roundtrip_max_abs=2e-5,gradient_abs=1e-4,gradient_relative=.02,epsilon=.001),
        gradient_fixture=dict(width=96,height=96,K=[[90.,0.,54.25],[0.,93.,41.75],[0.,0.,1.]],w2c=np.eye(4).tolist(),
            means=[[.05,-.03,2.5]],scales=[[.12,.08,.06]],quats_wxyz=[[.96,.17,.12,.08]],opacity=.7,pixel=[56,41],seed=12215,
            parameters=[['means',[0,0]],['means',[0,2]],['log_scales',[0,1]],['quats',[0,2]],['opacity_logits',[0]],['sh',[0,5,1]]]),
        optimizer=read(R/'native-training-integration/plan.json')['optimizer'],
        budget=dict(driver_runs=1,optimizer_steps_small=12,optimizer_steps_stop=2,optimizer_steps_large=1,fd_forwards=24,recorded_cli_renders=2,GPU_tasks_at_once=1),
        stopping='Keep any failure and initial source. Only fix demonstrated implementation mistakes; no scale/opacity/eps/budget/acceptance loosening. Do not repeat completed T0a wholesale.',
        source_hashes={n:sha(H/n) for n in names},preserved_source_hashes={n:sha(H/n) for n in preserved},backend_hashes={n:sha(G/n) for n in sources},
        input_hashes={n:sha(R/n) for n in inputs})
    save(O/'plan.json',plan)
    q=read(R/'queue-state.json');q.update(updated=now(),phase='T1_native_v2_repair_registered',running=None,next='Execute T1 bounded repair validation once; no human training.');save(R/'queue-state.json',q)
    print('T1_REGISTERED')

def all_tiles_reference(p,E,K,width,height,projection,radial):
    import torch,math
    from gsplat.cuda._wrapper import isect_tiles,isect_offset_encode,spherical_harmonics,rasterize_to_pixels_eval3d
    c,n=len(E),len(p['means']);tile=16;tw=math.ceil(width/tile);th=math.ceil(height/tile)
    center=torch.tensor([width*.5,height*.5],device='cuda').expand(c,n,2).contiguous()
    radii=torch.full((c,n,2),2*max(width,height),device='cuda',dtype=torch.int32)
    depth=torch.einsum('ci,ni->cn',E[:,2,:3],p['means'])+E[:,2,3,None]
    _,ids,flat=isect_tiles(center,radii,depth,tile,tw,th,packed=False)
    offsets=isect_offset_encode(ids,c,tw,th).reshape(c,th,tw)
    eye=torch.inverse(E)[:,:3,3];dirs=p['means'][None]-eye[:,None]
    color=(spherical_harmonics(3,dirs,p['sh'][None].expand(c,-1,-1,-1))+.5).clamp_min(0)
    return rasterize_to_pixels_eval3d(p['means'],p['quats'],p['scales'],color,p['opacities'][None].expand(c,-1),E,K,width,height,tile,offsets,flat,
        camera_model='pinhole' if projection=='pinhole' else 'fisheye',radial_coeffs=radial)

def run():
    if (O/'progress.json').exists():raise RuntimeError('Already started; inspect existing process/result')
    plan=read(O/'plan.json')
    for group,root in [('source_hashes',H),('preserved_source_hashes',H),('backend_hashes',G),('input_hashes',R)]:
        for n,d in plan[group].items():assert sha(root/n)==d,n
    start=time.monotonic();result=dict(stage='T1',checks=[],human_training_started=False,physical_admission=False,interactive_UI_tested=False)
    def check(name,passed,**data):
        result['checks'].append(dict(name=name,passed=bool(passed),**data))
        if not passed:raise AssertionError(name)
    def reject(name,fn,contains):
        try:fn()
        except (ValueError,RuntimeError,FileExistsError) as e:
            check(name,contains.lower() in str(e).lower(),error=str(e))
        else:check(name,False)
    def progress(phase):save(O/'progress.json',dict(state='running',phase=phase,pid=os.getpid(),updated=now()));print(phase,flush=True)
    def stop(*_):raise TimeoutError('Registered T1 300-second budget')
    signal.signal(signal.SIGALRM,stop);signal.alarm(plan['limits']['wall_seconds']);progress('initializing')
    try:
        busy=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
        if busy:raise RuntimeError('GPU already busy:'+busy)
        sys.path.insert(0,str(G))
        import torch,gsplat
        from PIL import Image
        from native_portrait_renderer122 import NativeCameraContract
        from native_portrait_renderer122_v2 import NativeCameraContractV2,render_native,REVISION
        from native_scene_package122 import load_scene as load_legacy
        from native_scene_package122_v2 import load_scene,render_recorded,write_scene,runtime_record
        from native_training122_v2 import FreshNativeGaussians,NativeTrainingView,render_view,optimize_fresh,foreground_objective,export_native_checkpoint
        from probe_native_culling122 import low_level
        torch.set_num_threads(2);torch.manual_seed(12215)
        t=lambda x:torch.as_tensor(np.asarray(x),dtype=torch.float32,device='cuda')
        limit=plan['limits'];result['runtime']=runtime_record()
        def image(name,x):Image.fromarray(np.rint(np.asarray(x).clip(0,1)*255).astype('uint8')).save(O/(name+'.png'))
        def direct(p,E,K,w,h,contract,radial=None):
            return render_native(p['means'],p['quats'],p['scales'],p['opacities'],p['sh'],E,K,w,h,contract=contract,radial_coeffs=radial,sh_degree=3)
        progress('normal_covariance_SH_camera_and_package')
        old=np.load(R/'covariance-sh-ply/original-parameters.npz');params={k:old[k] for k in ['means','quats','scales','opacities','sh']}
        E=t(old['viewmats']);K=t(old['Ks']);rp=read(R/'covariance-sh-ply/plan.json');width,height=rp['width'],rp['height']
        cameras=[dict(id=str(i),image_name=f'synthetic_{i}.png',width=width,height=height,K=k.tolist(),w2c=e.tolist()) for i,(k,e) in enumerate(zip(old['Ks'],old['viewmats']))]
        result['normal']={};package_paths={};floats={}
        for projection in ['pinhole','opencv_fisheye4']:
            contract=NativeCameraContractV2(projection,'synthetic_known_camera');radial=None if projection=='pinhole' else t([rp['fisheye_coeffs']]*3)
            p={k:t(v).requires_grad_(True) for k,v in params.items()}
            rgb,alpha,meta=direct(p,E,K,width,height,contract,radial)
            with torch.no_grad():ref,refa=all_tiles_reference(p,E,K,width,height,projection,radial)
            error=max(float((rgb-ref).abs().max()),float((alpha-refa).abs().max()))
            check('Geometric bounds retain tested response '+projection,error<limit['all_tiles_reference_max_abs'],max_abs=error,reference='all Gaussian/image tile intersections, no UT bounds')
            yy,xx=torch.meshgrid(torch.arange(height,device='cuda'),torch.arange(width,device='cuda'),indexing='ij')
            loss=(rgb*(1+xx[None,:,:,None]/width+.7*yy[None,:,:,None]/height)).sum()/100;loss.backward()
            grads={n:dict(finite=v.grad is not None and bool(torch.isfinite(v.grad).all()),max_abs=0 if v.grad is None else float(v.grad.abs().max())) for n,v in p.items()}
            check('Finite nonzero geometry covariance opacity SH gradients '+projection,all(x['finite'] and x['max_abs']>0 for x in grads.values()),gradients=grads)
            package=write_scene(O/('normal-'+projection),params,cameras,contract=contract,radial=None if projection=='pinhole' else rp['fisheye_coeffs'],scope='synthetic-control-only')
            package_paths[projection]=str(package);scene=load_scene(package);parity=[]
            for ci in range(3):
                im,aa,receipt=render_recorded(scene,str(ci),projection)
                mismatch=max(float(np.abs(im-rgb[ci].detach().cpu().numpy()).max()),float(np.abs(aa-alpha[ci].detach().cpu().numpy()).max()));parity.append(mismatch)
                image(f'{projection}_{ci}',im)
            check('Full SH covariance and recorded-camera package roundtrip '+projection,max(parity)<limit['small_roundtrip_max_abs'],max_abs=max(parity))
            np.savez_compressed(O/(projection+'-reference.npz'),rgb=rgb[0].detach().cpu().numpy(),alpha=alpha[0].detach().cpu().numpy())
            floats[projection]=rgb.detach().cpu().numpy();result['normal'][projection]=dict(reference_difference=error,package_difference=max(parity),visible=int((meta['radii']>0).all(-1).sum()))
        grid=Image.new('RGB',(width*3,height*2))
        for row,projection in enumerate(['pinhole','opencv_fisheye4']):
            for col in range(3):grid.paste(Image.open(O/f'{projection}_{col}.png'),(col*width,row*height))
        grid.save(O/'native_v2_views.png')
        save(O/'figure-index.json',dict(rows=['known synthetic pinhole','known synthetic KB4'],columns=['recorded camera0','recorded camera1','recorded camera2'],tile_size=[width,height],images_rescaled=False,text_on_image=False))
        progress('contract_rejections')
        plain=NativeCameraContractV2('pinhole','synthetic_known_camera');p={k:t(v) for k,v in params.items()}
        reject('Legacy contract rejected',lambda:direct(p,E,K,width,height,NativeCameraContract('pinhole','synthetic_known_camera')),'v2')
        reject('Already rectified image cannot acquire fisheye',lambda:NativeCameraContractV2('opencv_fisheye4','rectified').validate(),'rectified')
        reject('Vendor eight coefficients cannot masquerade as KB4',lambda:direct(p,E,K,width,height,NativeCameraContractV2('opencv_fisheye4','synthetic_known_camera'),torch.zeros((3,8),device='cuda')),'four')
        reject('K gradients cannot be silently ignored',lambda:direct(p,E,K.clone().requires_grad_(True),width,height,plain),'freezes')
        reflected=E.clone();reflected[0,0]*=-1
        reject('Reflected pose rejected',lambda:direct(p,reflected,K,width,height,plain),'rigid')
        wrong=dict(p,sh=p['sh'][:,:15]);reject('Incomplete SH rejected',lambda:direct(wrong,E,K,width,height,plain),'SH')
        reject('New loader rejects old packages',lambda:load_scene(R/'native-training-integration/existing-backend-path/final-native-package/scene.json'),'v2')
        reject('Old loader rejects new packages',lambda:load_legacy(package_paths['pinhole']),'schema')
        source=Path(package_paths['pinhole']);original=read(source)
        for name,mutate,expected in [
            ('wrong_bounds',lambda m:m['renderer'].update(bounds='legacy-opacity-compensated'),'renderer'),
            ('wrong_backend',lambda m:m['backend']['source_sha256'].update(adapter='0'*64),'mismatch'),
            ('wrong_asset_hash',lambda m:m['model']['asset'].update(sha256='0'*64),'SHA256'),
            ('path_escape',lambda m:m['model']['asset'].update(path='../point_cloud.ply'),'escapes')]:
            altered=copy.deepcopy(original);mutate(altered);path=source.parent/(name+'.json');save(path,altered)
            reject(name,lambda path=path:load_scene(path),expected)
        shifted=K.clone();shifted[:,0,2]+=1
        with torch.no_grad():base,_,_=direct(p,E,K,width,height,plain);changed,_,_=direct(p,E,shifted,width,height,plain)
        check('Off-center principal point influences actual ray image',float((base-changed).abs().max())>1e-4,max_abs=float((base-changed).abs().max()))
        progress('finite_difference_geometry_gradients')
        f=plan['gradient_fixture'];rng=np.random.default_rng(f['seed'])
        tiny=dict(means=np.array(f['means'],'float32'),quats=np.array(f['quats_wxyz'],'float32'),scales=np.array(f['scales'],'float32'),opacities=np.array([f['opacity']],'float32'),sh=rng.normal(0,.025,(1,16,3)).astype('float32'))
        tiny['sh'][:,0]=(np.array([.4,.6,.7])-.5)/.28209479177387814
        ee=t([f['w2c']]);kk=t([f['K']]);result['gradient_difference']={}
        for projection in ['pinhole','opencv_fisheye4']:
            model=FreshNativeGaussians(tiny,device='cuda');contract=NativeCameraContractV2(projection,'synthetic_known_camera');radial=None if projection=='pinhole' else t([rp['fisheye_coeffs']])
            def scalar():
                im,aa,_=direct(model.parameters_linear(),ee,kk,f['width'],f['height'],contract,radial)
                x,y=f['pixel'];return (im[0,y,x]*t([.2,.3,.5])).sum()+.4*aa[0,y,x,0]
            scalar().backward();rows=[]
            for name,index in f['parameters']:
                value=getattr(model,name);ind=tuple(index);analytic=float(value.grad[ind]);saved=value[ind].detach().clone();eps=limit['epsilon']
                with torch.no_grad():
                    value[ind]=saved+eps;plus=float(scalar());value[ind]=saved-eps;minus=float(scalar());value[ind]=saved
                fd=(plus-minus)/(2*eps);difference=abs(fd-analytic)
                rows.append(dict(parameter=name,index=index,analytic=analytic,finite_difference=fd,abs_difference=difference))
                check('Central difference '+projection+' '+name+str(index),difference<=limit['gradient_abs']+limit['gradient_relative']*abs(fd),**rows[-1])
            result['gradient_difference'][projection]=rows
        progress('same_twenty_hundred_thousand_fixture')
        large=np.load(R/'native-training-integration/subpixel-culling-diagnosis/frozen-large-arrays.npz');large={k:large[k] for k in large.files}
        lc=read(R/'native-training-integration/subpixel-culling-diagnosis/camera.json');le=t(lc['w2c']);lk=t(lc['K'])
        model=FreshNativeGaussians(large,device='cuda');before=model.means.detach().clone();p=model.parameters_linear()
        rgb,alpha,meta=direct(p,le,lk,64,64,plain)
        with torch.no_grad():reference,ra,_=low_level(p,le,lk,64,64,False)
        diff=max(float((rgb-reference).abs().max()),float((alpha-ra).abs().max()))
        check('Same200k fixture reproduces diagnostic repair',diff<limit['primitive_parity_max_abs'] and int((meta['radii']>0).all(-1).sum())==200000,max_abs=diff)
        objective=rgb.square().mean()+alpha.square().mean();objective.backward()
        gradients=int((model.means.grad.norm(dim=-1)>0).sum());finite=all(v.grad is not None and bool(torch.isfinite(v.grad).all()) for v in model.parameters())
        opt=model.optimizer(plan['optimizer']);opt.step();moving=int((torch.linalg.norm(model.means-before,dim=-1)>0).sum())
        result['large']=dict(points=200000,steps=1,position_gradients=gradients,moving_centers=moving,finite=finite,image_size=[64,64],full_resolution_capacity_tested=False)
        check('Same200k free centers really update',finite and gradients==moving==limit['large_gradient_count'],**result['large'])
        camera=dict(id='0',image_name='large_synthetic.png',width=64,height=64,K=lc['K'][0],w2c=lc['w2c'][0])
        package=export_native_checkpoint(O/'large-package',model,[camera],plain,scope='synthetic-control-only')
        im,aa,_=render_recorded(load_scene(package),'0','pinhole')
        with torch.no_grad():r,a,_=direct(model.parameters_linear(),le,lk,64,64,plain)
        delta=max(float(np.abs(im-r.cpu().numpy()[0]).max()),float(np.abs(aa-a.cpu().numpy()[0]).max()))
        check('Updated200k full-attribute PLY and camera roundtrip',delta<limit['large_roundtrip_max_abs'],max_abs=delta)
        image('large_updated',im)
        reject('A second optimizer cannot restart updated parameters',lambda:model.optimizer(plan['optimizer']),'Fresh optimizer')
        del large,model,opt,p,rgb,alpha,reference,ra;torch.cuda.empty_cache()
        progress('fresh_v2_training_component')
        data=np.load(R/'native-training-integration/existing-backend-path/fresh-initial-arrays.npz');initial={k:data[k] for k in data.files}
        data=np.load(R/'native-training-integration/existing-backend-path/known-synthetic-target.npz');target={k:data[k] for k in data.files}
        cs=read(R/'native-training-integration/existing-backend-path/final-native-package/cameras.json');teacher=FreshNativeGaussians(target,device='cuda');views=[]
        for c in cs[:4]:
            zero=np.zeros((c['height'],c['width'],3),'float32');valid=np.ones((c['height'],c['width'],1),bool)
            placeholder=NativeTrainingView.create(c,zero,valid.astype('float32'),valid,valid,device='cuda')
            with torch.no_grad():im,aa=render_view(teacher,placeholder,plain)
            views.append(NativeTrainingView.create(c,im.cpu().numpy(),aa.cpu().numpy(),aa.cpu().numpy()>.05,valid,device='cuda'))
        student=FreshNativeGaussians(initial,device='cuda')
        def evaluate(m):
            with torch.no_grad():return float(np.mean([float(foreground_objective(*render_view(m,v,plain),v)[0]) for v in views]))
        before_loss=evaluate(student)
        receipt,_=optimize_fresh(student,views,plain,steps=12,learning_rates=plan['optimizer'],seed=12214,output=O/'component-run')
        after_loss=evaluate(student);result['component']=dict(receipt=receipt,before=before_loss,after=after_loss)
        check('V2 component starts empty and completes all frozen views',receipt['optimizer_initial_state_entries']==0 and receipt['view_update_counts']==[3]*4 and receipt['frozen_camera_tensors_unchanged'] and after_loss<before_loss,ratio=after_loss/before_loss)
        package=export_native_checkpoint(O/'component-package',student,cs[:4],plain,scope='synthetic-control-only')
        ci,ca,_=render_recorded(load_scene(package),'0','pinhole')
        with torch.no_grad():ri,ra=render_view(student,views[0],plain)
        delta=max(float(np.abs(ci-ri.cpu().numpy()).max()),float(np.abs(ca-ra.cpu().numpy()).max()))
        check('Training and checkpoint loader use identical v2 evaluation',delta<limit['small_roundtrip_max_abs'],max_abs=delta)
        image('component_after12',ci)
        fresh=FreshNativeGaussians(initial,device='cuda')
        reject('V2 missing1000 review rejected before output',lambda:optimize_fresh(fresh,views,plain,steps=1000,learning_rates=plan['optimizer'],seed=12214,output=O/'must-not-start'),'review')
        check('Rejected training leaves no run directory',not (O/'must-not-start').exists())
        stop_result,_=optimize_fresh(fresh,views,plain,steps=5,learning_rates=plan['optimizer'],seed=12214,output=O/'stop-control',review_steps=[2],review=lambda step,path:False)
        check('V2 review rejection stops at requested step',stop_result['steps']==2 and stop_result['state']=='stopped_at_review')
        altered=FreshNativeGaussians(initial,device='cuda')
        with torch.no_grad():altered.means[0,0]+=.01
        reject('Modified initial state cannot receive a new fresh optimizer',lambda:altered.optimizer(plan['optimizer']),'untouched')
        for group,root in [('source_hashes',H),('preserved_source_hashes',H),('backend_hashes',G),('input_hashes',R)]:
            for n,d in plan[group].items():assert sha(root/n)==d,n
        result.update(state='driver_complete_cli_and_review_pending',all_driver_controls_passed=True,seconds=time.monotonic()-start,updated=now(),CLI_tested=False)
        save(O/'result.json',result);save(O/'progress.json',dict(state=result['state'],pid=None,updated=now()))
        save(O/'cli-jobs.json',dict(jobs=[dict(scene=package_paths[x],camera='0',projection=x,output=str(O/('cli-'+x)),reference=str(O/(x+'-reference.npz'))) for x in ['pinhole','opencv_fisheye4']]))
        print(json.dumps(dict(state=result['state'],checks=len(result['checks']),large=result['large'],component=result['component'],seconds=result['seconds'])),flush=True)
    except BaseException as e:
        result.update(state='failed',error=repr(e),traceback=traceback.format_exc(),seconds=time.monotonic()-start,updated=now())
        save(O/'result.json',result);save(O/'progress.json',dict(state='failed',pid=None,error=repr(e),updated=now()));raise
    finally:signal.alarm(0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','run']);a=p.parse_args()
    register() if a.action=='register' else run()
