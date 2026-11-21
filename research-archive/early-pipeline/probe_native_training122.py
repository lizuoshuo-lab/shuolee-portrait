"""One registered synthetic integration suite, never a person training entry."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, os, sys, time, traceback, signal, subprocess
import numpy as np

H=Path(__file__).resolve().parent
WIN=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
R=WIN if os.name=='nt' else Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
O=R/'native-training-integration'
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def register():
    O.mkdir(exist_ok=False)
    previous=read(R/'queue-state.json')
    if previous['running'] is not None:raise RuntimeError('Existing recorded run; inspect before registration')
    names=['native_training122.py','probe_native_training122.py','native_portrait_renderer122.py',
           'native_pixel_convention122.py','native_gaussian_ply122.py','native_scene_package122.py','run_native_training122.sh']
    old=Path('D:/GS-SLAM-Data/shuolee-portrait-new-capture-118/src/shuolee_portrait')
    plan=dict(stage='T0a',registered=now(),kind='synthetic_native_training_integration_only',
        causal_variable='Actual parameter optimization through pinned native UT/ray3D instead of119screen-space training. This is an engineering integration, not a calibration/ghosting intervention.',
        source_audit=dict(old119='WorldXYZ already free, but renderer is gaussian_renderer screen-space; alpha obtained by separate white-color render. Native122adapter was rendering/gradient-only, with no optimizer loop.',
            reuse='Existing validated native adapter, pixel-origin bridge, standard PLY codec and SHA-bound package loader. Reuse119maskedSSIM formulation with explicit observed/occluded validity.',
            new='Fresh direct world means/full covariance/SH parameters, emptyAdam, fixed recorded cameras, nativeRGB/alpha, exact masks, deterministic framecycles and mandatory1000review hook.'),
        fixture=dict(seed=12214,points=96,width=160,height=160,axes=[.58,.78,.40],
             camera_radius=3.,camera_height=.20,train_azimuth_degrees=[-45,45,135,225],heldout_azimuth_degrees=[0,90],
             K_integer=[[160.,0.,91.25],[0.,164.,75.25],[0.,0.,1.]],
             target_scale=[.12,.10,.06],target_opacity=.94,initial_offset=[.04,.02,0.],initial_jitter=.008,
             initial_scale_ratio=1.15,initial_opacity=.65,initial_rgb=.45,rgb_mask='known synthetic target alpha >0.05',
             all_pixels_observable=True,SH_degree=3,steps=120),
        optimizer=dict(means=.003,quats=.002,log_scales=.005,opacity_logits=.02,sh=.03),
        limits=dict(train_objective_ratio=.8,heldout_objective_ratio=.9,roundtrip_max_abs=3e-6,
             outside_supervision_difference=1e-7,wall_seconds=300,smoke_points=200000,smoke_width=64,smoke_height=64,
             smoke_steps=1,smoke_minimum_moving_centers=100),
        controls=['No outside-mask RGB effect orgradient;invalid occluder alpha/RGB ignored.',
          'Native fixed offcenterK/cameras unchanged;alltrainingviews seen;noheldoutframe optimization.',
          'New96point run120steps and onefreshreview-stop run2of5steps. Review callback missing at1000 rejects beforeoutput.',
          'One200000point forward/backward/Adamupdate onananalyticellipsoid entirelyinfrustum, nooffscreenpadding oroldbodyinit;somepoints may beoccluded.',
          'OneKB4synthetic forward/backward;notapplying fisheye totheMP4.',
          'Final completecovariance/SH PLY+samecameras reloadthroughR1c and compare6nativeimages.',
          '12novel syntheticviews rendered forreview;no claim of person reconstruction or unseenrealvideo robustness.'],
        budget=dict(runs=1,max_training_steps=120,review_stop_steps=2,large_step=1,KB4_backward=1,GPU_tasks=1,new_downloads=0),
        stopping='Preserve anyfailure andalloutputs. Fixonlydemonstratedimplementation errors. Nohyperparameter sweeps, physicalcamerachanges,personinitializationortraining.',
        scope_limits=['Synthetic teacher and student use same known backend, not independentphysicalGT.',
           'No adaptive density, realpersonfreshsurface/SMPLfit, fullresolution dataloader orphysicaltrainingadmission integration yet.',
           'Model package is synthetic-control-only; do notpromote latestpointcloud orlabelit asnewperson.'],
        source_hashes={n:sha(H/n) for n in names},historical_source_hashes={str(old/n):sha(old/n) for n in ['clothed_training.py','clothed_surface.py','training.py']})
    save(O/'plan.json',plan)
    previous.update(updated=now(),phase='T0a_synthetic_native_training_registered',next='Execute registered T0a once, review numeric controls andsyntheticimages. No human training or new cameras.',running=None)
    save(R/'queue-state.json',previous);print('T0A_REGISTERED',flush=True)


def fixture(n,axes,scale,opacity,seed):
    i=np.arange(n,dtype=float);y=1-2*(i+.5)/n;phi=np.pi*(3-np.sqrt(5))*i
    unit=np.c_[np.sqrt(1-y*y)*np.cos(phi),y,np.sqrt(1-y*y)*np.sin(phi)]
    means=unit*np.asarray(axes)
    from scipy.spatial.transform import Rotation
    quats=Rotation.from_rotvec(np.c_[.25*np.sin(phi),.4*np.cos(phi),.3*y]).as_quat()[:,[3,0,1,2]]
    colors=np.c_[.25+.45*(unit[:,0]>0),.25+.45*(unit[:,1]>0),.30+.30*(unit[:,2]>0)]
    sh=np.zeros((n,16,3));sh[:,0]=(colors-.5)/.28209479177387814
    sh[:,1:]=np.random.default_rng(seed).normal(0,.025,(n,15,3))
    return dict(means=means.astype('float32'),quats=quats.astype('float32'),scales=np.tile(scale,(n,1)).astype('float32'),opacities=np.full(n,opacity,'float32'),sh=sh.astype('float32'))


def pose(angle,radius,height):
    radians=np.deg2rad(angle);eye=np.array([np.sin(radians)*radius,height,np.cos(radians)*radius])
    f=-eye/np.linalg.norm(eye);right=np.cross(f,[0,1,0]);right/=np.linalg.norm(right);down=np.cross(f,right)
    E=np.eye(4);E[:3,:3]=[right,down,f];E[:3,3]=-E[:3,:3]@eye;return E


def run():
    if (O/'progress.json').exists():raise RuntimeError('This suite has already started;inspect instead ofrerunning')
    plan=read(O/'plan.json')
    effective_sources=dict(plan['source_hashes'])
    if (O/'implementation-repair.json').exists():
        repair=read(O/'implementation-repair.json')
        assert repair['original_plan_sha256']==sha(O/'plan.json')
        assert repair['failure_sha256']==sha(O/'launch-failure.log')
        assert set(repair['replacement_source_hashes'])=={'probe_native_training122.py'}
        effective_sources.update(repair['replacement_source_hashes'])
    for name,digest in effective_sources.items():assert sha(H/name)==digest,name
    for name,digest in plan['historical_source_hashes'].items():
        path=Path('/mnt/'+name[0].lower()+name[2:].replace('\\','/'))
        assert sha(path)==digest,path
    start=time.monotonic();result=dict(stage='T0a',updated=now(),checks=[],human_training_started=False,physical_admission=False)
    def progress(stage,**extra):
        save(O/'progress.json',dict(stage=stage,pid=os.getpid(),command=sys.argv,updated=now(),**extra));print(stage,flush=True)
    def check(name,passed,**extra):
        row=dict(name=name,passed=bool(passed),**extra);result['checks'].append(row)
        if not passed:raise AssertionError(name)
    def timeout(*_):raise TimeoutError('Registered300second budgetexceeded')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(plan['limits']['wall_seconds'])
    progress('starting')
    try:
        gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
        if gpu:raise RuntimeError('GPU alreadyhascomputeprocess:'+gpu)
        import torch,gsplat
        torch.set_num_threads(2);torch.manual_seed(plan['fixture']['seed']);torch.cuda.manual_seed_all(plan['fixture']['seed'])
        from dataclasses import replace
        from PIL import Image
        from native_training122 import NativeTrainingView,FreshNativeGaussians,foreground_objective,render_view,optimize_fresh,export_native_checkpoint
        from native_portrait_renderer122 import NativeCameraContract
        from native_pixel_convention122 import to_native_pixels
        from native_scene_package122 import load_scene,render_recorded
        result['runtime']=dict(torch=torch.__version__,gsplat=gsplat.__version__,GPU=torch.cuda.get_device_name())
        f=plan['fixture'];target=fixture(f['points'],f['axes'],f['target_scale'],f['target_opacity'],f['seed'])
        initial={k:v.copy() for k,v in target.items()};rng=np.random.default_rng(f['seed']+1)
        initial['means']+=np.asarray(f['initial_offset'])+rng.normal(0,f['initial_jitter'],initial['means'].shape)
        initial['scales']*=f['initial_scale_ratio'];initial['opacities'][:]=f['initial_opacity'];initial['sh'][:]=0
        initial['sh'][:,0]=(f['initial_rgb']-.5)/.28209479177387814
        np.savez_compressed(O/'fresh-initial-arrays.npz',**initial);np.savez_compressed(O/'known-synthetic-target.npz',**target)
        contract=NativeCameraContract('pinhole','synthetic_known_camera');teacher=FreshNativeGaussians(target,device='cuda')
        K,_,bridge=to_native_pixels(np.array(f['K_integer']),np.zeros((0,2)),intrinsic_convention='integer_centers',observation_convention='integer_centers')
        save(O/'pixel-origin.json',bridge);views=[];targets=[]
        zeros=np.zeros((f['height'],f['width'],3),np.float32);ones=np.ones((f['height'],f['width'],1),bool)
        for index,angle in enumerate(f['train_azimuth_degrees']+f['heldout_azimuth_degrees']):
            camera=dict(id=str(index),image_name=f'synthetic_{index}.png',width=f['width'],height=f['height'],K=K.tolist(),w2c=pose(angle,f['camera_radius'],f['camera_height']).tolist())
            placeholder=NativeTrainingView.create(camera,zeros,ones.astype('float32'),ones,ones,device='cuda')
            with torch.no_grad():rgb,alpha=render_view(teacher,placeholder,contract)
            rgb0=rgb.cpu().numpy();a0=alpha.cpu().numpy();mask=a0>.05
            view=NativeTrainingView.create(camera,rgb0,a0,mask,ones,device='cuda');views.append(view);targets.append((rgb0,a0))
        model=FreshNativeGaussians(initial,device='cuda');means0=model.means.detach().clone()
        v=views[0];rgb,alpha=render_view(model,v,contract)
        photo,terms=foreground_objective(rgb,alpha,v)
        other=replace(v,rgb=torch.where(v.rgb_valid,v.rgb,.917))
        outside=float(abs(photo-foreground_objective(rgb,alpha,other)[0]))
        check('Outside-mask RGB never enters nativeobjective',outside<=plan['limits']['outside_supervision_difference'],difference=outside)
        pred=rgb.detach().requires_grad_(True);a=alpha.detach().requires_grad_(True)
        probe,_=foreground_objective(pred,a,v);probe.backward()
        check('Outside-mask RGB has exactlyzero objectivegradient',bool((pred.grad[~v.rgb_valid[...,0]]==0).all()))
        observed=v.observation_valid.clone();observed[65:95,65:95]=False
        occ=replace(v,observation_valid=observed,rgb_valid=v.rgb_valid&observed)
        altered=replace(occ,rgb=torch.where(observed,occ.rgb,.999),alpha=torch.where(observed,occ.alpha,.999))
        check('Externaloccluder area excludedfromRGBandalpha',float(abs(foreground_objective(rgb,alpha,occ)[0]-foreground_objective(rgb,alpha,altered)[0]))<=1e-7)
        def evaluate(m):
            values=[];images=[]
            with torch.no_grad():
                for view in views:
                    im,aa=render_view(m,view,contract);loss,parts=foreground_objective(im,aa,view)
                    values.append(dict(frame=view.camera['id'],objective=float(loss),**{k:float(x) for k,x in parts.items()}));images.append(im.cpu().numpy())
            return values,images
        before,initial_images=evaluate(model)
        for label,frames in [('target',[x[0] for x in targets]),('initial',initial_images)]:
            for index,img in enumerate(frames):Image.fromarray(np.rint(img.clip(0,1)*255).astype('uint8')).save(O/f'{label}_{index}.png')
        progress('optimizing_known_synthetic_fixture',steps=f['steps'])
        trained,history=optimize_fresh(model,views[:4],contract,steps=f['steps'],learning_rates=plan['optimizer'],seed=f['seed'],output=O/'small-run')
        after,final_images=evaluate(model)
        result['metrics']=dict(before=before,after=after,train_ratio=float(np.mean([x['objective'] for x in after[:4]])/np.mean([x['objective'] for x in before[:4]])),heldout_ratio=float(np.mean([x['objective'] for x in after[4:]])/np.mean([x['objective'] for x in before[4:]])))
        for index,img in enumerate(final_images):Image.fromarray(np.rint(img.clip(0,1)*255).astype('uint8')).save(O/f'final_{index}.png')
        result['small_run']=trained
        check('Alltrainingviews optimized exactly30times;heldoutneveroptimized',trained['view_update_counts']==[30]*4)
        check('Emptyfreshoptimizer andfiniteupdates',trained['optimizer_initial_state_entries']==0 and trained['optimizer_final_state_entries']==5)
        movement=torch.linalg.norm(model.means-means0,dim=-1).detach().cpu().numpy()
        result['movement']=dict(min=float(movement.min()),median=float(np.median(movement)),max=float(movement.max()))
        check('Freeworldcenters actuallymove',bool((movement>1e-5).all()))
        path=export_native_checkpoint(O/'final-native-package',model,[v.camera for v in views],contract,scope='synthetic-control-only')
        loaded=load_scene(path);parity=[]
        for index,view in enumerate(views):
            im,aa,_=render_recorded(loaded,view.camera['id'],'pinhole')
            with torch.no_grad():native,nativea=render_view(model,view,contract)
            parity.append(dict(frame=index,rgb_max=float(np.abs(im-native.cpu().numpy()).max()),alpha_max=float(np.abs(aa-nativea.cpu().numpy()).max())))
        result['native_roundtrip']=parity
        check('TrainedfullcovarianceSHPLYandcamera reload reproduce all6nativeviews',max(max(r['rgb_max'],r['alpha_max']) for r in parity)<plan['limits']['roundtrip_max_abs'])
        orbit=[]
        with torch.no_grad():
            for angle in range(0,360,30):
                camera=dict(views[0].camera);camera['id']=f'orbit_{angle}';camera['image_name']=camera['id']+'.png';camera['w2c']=pose(angle,3.,.20).tolist()
                view=NativeTrainingView.create(camera,zeros,ones.astype('float32'),ones,ones,device='cuda')
                im,_=render_view(model,view,contract);out=Image.fromarray(np.rint(im.cpu().numpy().clip(0,1)*255).astype('uint8'));out.save(O/f'orbit_{angle:03d}.png');orbit.append(out)
        orbit[0].save(O/'synthetic_orbit.gif',save_all=True,append_images=orbit[1:],duration=200,loop=0)
        grid=Image.new('RGB',(480,320))
        for row,index in enumerate([0,4]):
            for col,label in enumerate(['target','initial','final']):grid.paste(Image.open(O/f'{label}_{index}.png'),(col*160,row*160))
        grid.save(O/'synthetic_comparison.png')
        save(O/'figure-index.json',dict(rows=['trainingcamera0','heldoutcamera4'],columns=['knownsynthetic_target','fresh_initial','after120steps'],native_tile_size=[160,160],text_on_images=False,images_rescaled=False))
        # Review rejection must stop this fresh control before step3;not a resumed run.
        fresh=FreshNativeGaussians(initial,device='cuda')
        stop,_=optimize_fresh(fresh,views[:4],contract,steps=5,learning_rates=plan['optimizer'],seed=f['seed'],output=O/'review-stop-control',review_steps=[2],review=lambda step,receipt:False)
        check('Reviewdenial stops updates before nextstep',stop['steps']==2 and stop['state']=='stopped_at_review')
        try:optimize_fresh(fresh,views[:4],contract,steps=1000,learning_rates=plan['optimizer'],seed=f['seed'],output=O/'must-not-start')
        except ValueError:rejected=True
        else:rejected=False
        check('Missing1000review rejected before outputcreation',rejected and not (O/'must-not-start').exists())
        try:optimize_fresh(fresh,views[:4],contract,steps=1,learning_rates=plan['optimizer'],seed=f['seed'],output=O/'small-run')
        except FileExistsError:rejected=True
        else:rejected=False
        check('Existingoutput cannot becontinued oroverwritten',rejected)
        fresh.zero_grad(set_to_none=True);fish=NativeCameraContract('opencv_fisheye4','synthetic_known_camera')
        frgb,falpha=render_view(fresh,views[0],fish,radial=[.1,-.03,.003,-.0001]);(frgb.mean()+falpha.mean()).backward()
        check('ExplicitKB4synthetic path hasfinitegeometrygradients',fresh.means.grad is not None and bool(torch.isfinite(fresh.means.grad).all()) and bool((fresh.means.grad.abs()>0).any()))
        del fresh,teacher,pred,a,rgb,alpha,photo,probe;torch.cuda.empty_cache()
        progress('two_hundred_thousand_free_points_one_step')
        big=fixture(200000,f['axes'],[.004,.005,.003],.1,f['seed'])
        bigmodel=FreshNativeGaussians(big,device='cuda');opt=bigmodel.optimizer(plan['optimizer']);beforebig=bigmodel.means.detach().clone()
        camera=dict(views[0].camera,width=64,height=64,K=[[60.,0.,33.5],[0.,62.,30.5],[0.,0.,1.]])
        view=NativeTrainingView.create(camera,np.zeros((64,64,3),np.float32),np.ones((64,64,1),np.float32),np.ones((64,64,1),bool),np.ones((64,64,1),bool),device='cuda')
        torch.cuda.reset_peak_memory_stats();bigrgb,biga=render_view(bigmodel,view,contract);objective=bigrgb.square().mean()+biga.square().mean();objective.backward()
        finite=all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in bigmodel.parameters())
        gradient_centers=int((bigmodel.means.grad.norm(dim=-1)>0).sum());opt.step();torch.cuda.synchronize()
        moving=int((torch.linalg.norm(bigmodel.means-beforebig,dim=-1)>0).sum())
        result['large_smoke']=dict(points=200000,steps=1,nonzero_position_gradients=gradient_centers,moving_centers=moving,finite_all_parameter_gradients=finite,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),image_size=[64,64],full_resolution_human_capacity_tested=False)
        check('200000freshfreepoints trainone step withfinitegradient',finite and moving>=plan['limits']['smoke_minimum_moving_centers'],**result['large_smoke'])
        for name,digest in effective_sources.items():assert sha(H/name)==digest,name
        check('Nativefixedcamera optimization reducesknownsynthetictrainobjective',result['metrics']['train_ratio']<=plan['limits']['train_objective_ratio'],ratio=result['metrics']['train_ratio'])
        check('Unoptimizedsyntheticheldoutview objectiveimproves',result['metrics']['heldout_ratio']<=plan['limits']['heldout_objective_ratio'],ratio=result['metrics']['heldout_ratio'])
        result.update(state='complete_pending_actual_review',all_controls_passed=True,seconds=time.monotonic()-start,updated=now(),person_training=False)
        save(O/'result.json',result);progress('complete_pending_actual_review',worker_finished=True)
        save(O/'progress.json',dict(state='complete_pending_actual_review',pid=None,updated=now()));print(json.dumps(result),flush=True)
    except BaseException as exc:
        result.update(state='failed',error=repr(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-start,updated=now())
        save(O/'result.json',result);save(O/'progress.json',dict(state='failed',pid=None,error=repr(exc),updated=now()));raise
    finally:signal.alarm(0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','run']);a=p.parse_args()
    register() if a.action=='register' else run()
