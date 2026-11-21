"""Native MSV2 fast recipe, full-scene demo inputs at native, Lanczos2, SwinIR4."""
from pathlib import Path
from datetime import datetime, timezone
from types import SimpleNamespace
import argparse, copy, hashlib, importlib.util, json, math, os, random, shutil, sys, time, traceback
import numpy as np
from PIL import Image

def host_path(value):
    value=str(value).replace('\\','/')
    if os.name!='nt' and len(value)>1 and value[1]==':':
        return Path('/mnt/'+value[0].lower()+value[2:])
    if os.name=='nt' and value.startswith('/mnt/') and len(value)>6:
        return Path(value[5].upper()+':'+value[6:])
    if os.name=='nt' and value.startswith('/opt/'):
        return Path('\\\\wsl.localhost\\Ubuntu-20.04'+value.replace('/','\\'))
    return Path(value)

CONFIG_INPUT=os.environ.get('MSV2_EXPERIMENT_CONFIG','')
CONFIG_PATH=host_path(CONFIG_INPUT) if CONFIG_INPUT else None
CONFIG=json.loads(CONFIG_PATH.read_text(encoding='utf-8-sig')) if CONFIG_PATH else {}
if CONFIG:
    assert CONFIG['schema']=='msv2-fullframe-experiment-v1'
    assert len(CONFIG['frames'])==CONFIG['expected_frames']
EXPECTED_FRAMES=CONFIG.get('expected_frames')
MODEL_ID_TEMPLATE=CONFIG.get('viewer_model_id_template','msv2-demo-fullframe-{variant}-20261104')
ROOT = host_path(CONFIG.get('run_root','/opt/gs-slam-lab/msv2-demo-fullframe-standard-20261104'))
PREP = host_path(CONFIG.get('prep_root','/mnt/d/GS-SLAM-Data/msv2-demo-fullframe-standard-20261104'))
LAB = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
NATIVE = Path('/opt/gs-slam-lab/mini-splatting2-native-001')
REPO = LAB/'repos/mini-splatting2'
RECORD = host_path(CONFIG.get('record_root','/mnt/d/00/记录/2026-11-04_MSV2_完整图_原始_插值2倍_超分4倍'))
WEIGHT = Path('/opt/gs-slam-lab/model-cache/swinir/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth')
WEIGHT_SHA = '99adfa91350a84c99e946c1eb3d8fce34bc28f57d807b09dc8fe40a316328c0a'
COMMIT = '5ec9202c2db7d900728bfc09733e85467cff885e'
NAMES = {1:'original', 2:'interpolation2x', 4:'swinir4x'}
FOURX_MEMORY_LIMIT_GIB = 13.5

def model_id(scale):return MODEL_ID_TEMPLATE.format(variant=NAMES[scale])

def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p, value):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');t.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def windows_path(p):
    return host_path(p)
def import_file(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def runtime():
    import resource
    soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
    target=16384 if hard==resource.RLIM_INFINITY else min(16384,hard)
    if soft!=resource.RLIM_INFINITY and soft<target:
        resource.setrlimit(resource.RLIMIT_NOFILE,(target,hard))
    sys.path[:0]=[str(ROOT/'source'),str(NATIVE/'deps')]
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    return torch
def native_rows():
    audit=read(PREP/'pixel-lineage.json');assert audit['passed'] and audit['no_spatial_resizing']
    rows=audit['records'];assert len({r['image'] for r in rows})==len(rows)
    if EXPECTED_FRAMES is not None:
        assert audit['config_sha256']==sha(CONFIG_PATH),'Input configuration changed after pixel verification'
        assert len(rows)==EXPECTED_FRAMES,(len(rows),EXPECTED_FRAMES)
        assert {r['image']:r['sha256'] for r in rows}=={r['name']:r['source_sha256'] for r in CONFIG['frames']}
    return audit['records']

def initialize():
    rows=native_rows()
    ROOT.mkdir(exist_ok=False)
    source=ROOT/'source';source.mkdir()
    checks=[]
    # These files control all training, densification, depth and sparse optimization.
    for part in ('msv2','scene','utils','gaussian_renderer','arguments','config','lpipsPyTorch'):
        src=REPO/part
        shutil.copytree(src,source/part,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        for p in sorted((source/part).rglob('*.py')):
            relative=p.relative_to(source); baseline=NATIVE/'repo'/relative
            assert baseline.exists(),relative
            assert p.read_text()==baseline.read_text(),relative
            checks.append(dict(path=str(relative),sha256=sha(p),matches_previous_native_reproduction=True))
    shutil.copy2(REPO/'config/fast',ROOT/'fast.config')
    for s,n in NAMES.items():(ROOT/n).mkdir()
    srsource=(host_path(CONFIG['reuse_superresolution_root'])/'swinir-source'
        if CONFIG.get('reuse_superresolution_root') else Path('/mnt/d/00/工作/SwinIR'))
    (ROOT/'swinir-source').mkdir()
    for part in ('models','utils'):
        shutil.copytree(srsource/part,ROOT/'swinir-source'/part,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(srsource/'main_test_swinir.py',ROOT/'swinir-source/main_test_swinir.py')
    shutil.copy2(__file__,ROOT/'experiment.py')
    plan=dict(created=now(),state='preparing',repository='https://github.com/fatPeter/mini-splatting2',commit=COMMIT,
        previous_reproduction=dict(remote_final_points=149413,local_final_points=192013,source=str(LAB/'outputs/mini-splatting2-mip-depth-001/native-baseline-18000-s0')),
        dataset_training_script=str(Path('/mnt/d/00/工作/minisplatting2/local_pipeline/test_gs.py')),
        recipe=dict(method='msv2 sparse',imp_metric='outdoor',config='config/fast',iterations=18000,seed=0,resolution=1,
            sampling_factor=.6,num_depth_factor=1,warn_until_iter=3000,depth_reinit_iter=2000,simp_iteration1=3000,simp_iteration2=8000,
            aggressive_clone_from_iter=500,aggressive_clone_interval=250,densify_until_iter=3000,eval=True,llff_holdout=8),
        variants={n:dict(scale=s,target_wh=[2704*s,1520*s],warmup_wh=[1352*s,760*s]) for s,n in NAMES.items()},
        explicit_user_input_policy='Native full video RGB, no spatial reduction before enlargement; no crop or mask; full background is supervised',
        data_recipe=f"{CONFIG.get('sample_fps',8)} fps source identities; single PINHOLE camera; affine and DSP SIFT; guided sequential matching; principal point bundle adjustment",
        frame_count=len(rows),selection_policy=CONFIG.get('selection_policy','Saved 8-fps source identities'),
        experiment_config=CONFIG_INPUT or None,experiment_config_sha256=sha(CONFIG_PATH) if CONFIG_PATH else None,
        viewer_model_id_template=MODEL_ID_TEMPLATE,
        input_source=str(PREP),source_sha256=sha(PREP/'pixel-lineage.json'),source_checks=checks,fast_sha256=sha(ROOT/'fast.config'),
        training_algorithm_changes=[],gaussian_count_cap=None,
        explicit_user_metric_override='Keep outdoor as requested; previous reproductions and dataset test_gs.py use indoor',
        runtime_adapters=['Explicitly closed uint8 NPY reads and exact contiguous CHW targets; process FD limit 16384',
            'Retain calibrated principal point in projection matrix',
            '4x allocator cap 13.5 GiB and cache trim after full-size backward; native model/loss/optimizer unchanged'],
        warmup_explanation='Official half-resolution training warmup is applied AFTER preparing each native/2x/4x input; no source downsampling before upscaling',
        fresh_gaussians=True,fresh_optimizer=True,old_fitted_state_loaded=False,mask_used=False,
        quality_accepted=False,training_completed=False,
        stop_conditions=['Nonfinite model/loss','Pixel or calibrated projection contract failure','Unsupported camera registration','Insufficient memory'])
    save(ROOT/'plan.json',plan);save(RECORD/'标准配置核查.json',plan)
    if CONFIG_PATH:
        shutil.copy2(CONFIG_PATH,ROOT/'experiment-config.json')
        shutil.copy2(CONFIG_PATH,RECORD/'实验配置.json')
    save(RECORD/'高斯数量核查.json',dict(updated=now(),prior_remote_points=149413,prior_native_points=192013,
        correction='Earlier no-warmup/custom-initialization experiments are superseded. The user explicitly retained outdoor; other schedule values follow prior config/fast. No fixed Gaussian budget is present in MSV2.',
        adopted_recipe=plan['recipe'],current_training_completed=False,quality_accepted=False))
    print('STANDARD_RECIPE_LOCKED',flush=True)

def seed_superresolution():
    """Reuse only byte-verified SR images with exactly equal full RGB inputs."""
    previous_root=CONFIG.get('reuse_superresolution_root')
    if not previous_root:return
    run=ROOT/NAMES[4];state_path=run/'superresolution.json'
    if state_path.exists():return
    previous_root=host_path(previous_root);previous_plan=read(previous_root/'plan.json')
    previous_prep=host_path(previous_plan['input_source'])
    assert read(previous_prep/'state.json')['video_sha256']==CONFIG['video_sha256']
    previous=read(previous_root/'swinir4x/superresolution.json')
    assert previous['state']=='completed' and previous['weight_sha256']==WEIGHT_SHA
    assert previous['precision']=='float32' and previous['tile']==256 and previous['tile_overlap']==32
    for rel in ('main_test_swinir.py','models/network_swinir.py'):
        assert sha(ROOT/'swinir-source'/rel)==sha(previous_root/'swinir-source'/rel)
    prior_pixels={r['rgb_md5']:r for r in read(previous_prep/'pixel-lineage.json')['records']}
    prior_sr={r['name']:r for r in previous['records']}
    out=run/'dataset/images';out.mkdir(parents=True,exist_ok=True);records=[]
    for row in native_rows():
        old=prior_pixels.get(row['rgb_md5'])
        if old is None:continue
        new_source=PREP/'dataset/images'/row['image'];old_source=previous_prep/'dataset/images'/old['image']
        assert sha(new_source)==row['sha256'] and sha(old_source)==old['sha256']
        with Image.open(new_source) as im:a=np.array(im)
        with Image.open(old_source) as im:b=np.array(im)
        assert a.shape==(1520,2704,3) and np.array_equal(a,b)
        reused=prior_sr[old['image']];assert reused['native_sha256']==old['sha256']
        old_output=previous_root/'swinir4x/dataset/images'/old['image'];assert sha(old_output)==reused['output_sha256']
        with Image.open(old_output) as im:assert im.mode=='RGB' and im.size==(10816,6080)
        target=out/row['image'];assert not target.exists();os.link(old_output,target)
        records.append(dict(name=row['image'],native_sha256=row['sha256'],output_sha256=reused['output_sha256'],seconds=0.,
            reused_from=str(old_output),reuse_storage='independent hard link to immutable image',
            previous_native_sha256=old['sha256'],full_rgb_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
            full_rgb_bitwise_equal=True))
    state=dict(state='running',started=now(),updated=now(),records=records,processed=len(records),reused_frames=len(records),
        weight_sha256=WEIGHT_SHA,precision='float32',tile=256,tile_overlap=32,
        reuse_policy='Identical decoded full RGB and identical public SwinIR source/weights; no learned Gaussian state reused')
    save(state_path,state);save(run/'superresolution-reuse.json',state)
    print('FULL_RGB_SUPERRESOLUTION_REUSED',len(records),len(native_rows()),flush=True)

def stage_pixels(scale):
    run=ROOT/NAMES[scale];out=run/'dataset/images';out.mkdir(parents=True,exist_ok=True)
    rows=native_rows();items=[]
    for i,r in enumerate(rows):
        src=PREP/'dataset/images'/r['image'];assert sha(src)==r['sha256']
        dst=out/src.name
        if not dst.exists():
            if scale==1:os.symlink(src,dst)
            elif scale==2:
                with Image.open(src) as im:im.resize((5408,3040),Image.Resampling.LANCZOS).save(dst,compress_level=3)
            else:raise RuntimeError('Run superresolution before stage_pixels(4)')
        with Image.open(dst) as im:assert im.mode=='RGB' and im.size==(2704*scale,1520*scale)
        items.append(dict(name=src.name,native_sha256=r['sha256'],image_sha256=sha(dst)))
        if i%20==0:print('PIXELS_READY',scale,i+1,flush=True)
    save(run/'pixels.json',dict(scale=scale,count=len(rows),items=items,full_rgb=True,no_mask=True,source_resolution=[2704,1520]))

def superresolve():
    torch=runtime();assert sha(WEIGHT)==WEIGHT_SHA
    # The fixed FP32/256-tile recipe measured below 2.8 GiB; bound its cache
    # while native training or the interactive viewer shares the device.
    torch.cuda.set_per_process_memory_fraction(4*1024**3/torch.cuda.mem_get_info()[1])
    sys.path.insert(0,str(ROOT/'swinir-source'))
    official=import_file('main_test_swinir',ROOT/'swinir-source/main_test_swinir.py')
    args=SimpleNamespace(task='real_sr',scale=4,large_model=True,model_path=str(WEIGHT),tile=256,tile_overlap=32)
    model=official.define_model(args).eval().cuda()
    run=ROOT/NAMES[4];out=run/'dataset/images';out.mkdir(parents=True,exist_ok=True)
    p=run/'superresolution.json'
    state=read(p) if p.exists() else dict(state='running',started=now(),records=[],weight_sha256=WEIGHT_SHA,precision='float32',tile=256,tile_overlap=32)
    done={r['name']:r for r in state['records']}
    with torch.inference_mode():
        for row in native_rows():
            control=ROOT/'pause-superresolution'
            if control.exists() and control.read_text(encoding='utf-8-sig').strip()!='run':
                state.update(state='paused_for_native_training',updated=now());save(p,state)
                print('SWINIR_PAUSED_AT_FRAME_BOUNDARY',len(state['records']),flush=True);return
            src=PREP/'dataset/images'/row['image'];dst=out/row['image']
            if row['image'] in done:
                assert done[row['image']]['native_sha256']==row['sha256']
                assert sha(dst)==done[row['image']]['output_sha256'];continue
            t=time.monotonic();assert sha(src)==row['sha256']
            rgb=np.asarray(Image.open(src).convert('RGB'))
            x=torch.from_numpy(rgb.copy()).to(device='cuda',dtype=torch.float32).permute(2,0,1)[None]/255.
            h,w=rgb.shape[:2];hp=(h//8+1)*8-h;wp=(w//8+1)*8-w
            x=torch.cat([x,torch.flip(x,[2])],2)[:,:,:h+hp,:]
            x=torch.cat([x,torch.flip(x,[3])],3)[:,:,:,:w+wp]
            pred=official.test(x,model,args,8)[...,:h*4,:w*4]
            assert pred.shape==(1,3,6080,10816) and torch.isfinite(pred).all()
            arr=np.rint(pred[0].clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
            temporary=dst.with_name(dst.name+'.tmp')
            Image.fromarray(arr).save(temporary,format='PNG',compress_level=3)
            temporary.replace(dst)
            item=dict(name=row['image'],native_sha256=row['sha256'],output_sha256=sha(dst),seconds=time.monotonic()-t)
            state['records'].append(item);state.update(updated=now(),processed=len(state['records']),state='running',peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
            save(p,state);print('SWINIR_FULLFRAME',len(state['records']),json.dumps(item),flush=True)
            del x,pred,arr
    state.update(state='completed',finished=now());save(p,state);stage_pixels(4)

def stage_cameras(scale):
    torch=runtime();assert read(PREP/'state.json')['state']=='completed'
    from scene.colmap_loader import read_intrinsics_binary
    from collections import namedtuple
    import struct
    run=ROOT/NAMES[scale];data=run/'dataset';sp=data/'sparse/0';sp.mkdir(parents=True,exist_ok=True)
    for name in ('images.bin','points3D.bin'):shutil.copy2(PREP/'dataset/sparse/0'/name,sp/name)
    cameras=read_intrinsics_binary(str(PREP/'dataset/sparse/0/cameras.bin'))
    with (sp/'cameras.bin').open('wb') as f:
        f.write(struct.pack('<Q',len(cameras)))
        for cid,c in cameras.items():
            assert c.model=='PINHOLE'
            f.write(struct.pack('<iiQQ',cid,1,c.width*scale,c.height*scale))
            f.write(struct.pack('<4d',*(np.asarray(c.params)*scale)))
    loaded=read_intrinsics_binary(str(sp/'cameras.bin'))
    for cid,c in cameras.items():
        assert loaded[cid].width==c.width*scale and loaded[cid].height==c.height*scale
        assert np.array_equal(loaded[cid].params,np.asarray(c.params)*scale)
    assert sha(sp/'images.bin')==sha(PREP/'dataset/sparse/0/images.bin')
    assert sha(sp/'points3D.bin')==sha(PREP/'dataset/sparse/0/points3D.bin')
    rows=copy.deepcopy(read(PREP/'camera-inputs.json')['records'])
    if EXPECTED_FRAMES is not None:assert len(rows)==EXPECTED_FRAMES
    pixel={r['name']:r for r in read(run/'pixels.json')['items']}
    for i,r in enumerate(rows):
        r['native_image']=str(PREP/'dataset/images'/r['name']);r['image']=str(data/'images'/r['name']);r['wh']=[2704*scale,1520*scale]
        r['native_sha256']=r['source_sha256'];r['source_sha256']=pixel[r['name']]['image_sha256']
        r['K']=(np.diag([scale,scale,1.])@np.asarray(r['K'])).tolist();r['split']='test' if i%8==0 else 'train'
        k=np.asarray(r['K']).copy();k[:2,2]-=.5;r['K_integer_centers']=k.tolist()
    save(run/'training-inputs.json',dict(records=rows,mask_used=False,crop_used=False,background_trained=True,scale=scale,
        initial_sparse_sha256=sha(sp/'points3D.bin'),camera_poses_sha256=sha(sp/'images.bin'),scaled_intrinsics_readback_verified=True))
    # Exactly PILtoTorch's default RGB resize, including the official warmup level.
    for level in (1,2):
        cache=run/f'rgb-cache-{level}';cache.mkdir(exist_ok=True)
        for i,r in enumerate(rows):
            dst=cache/(r['frame']+'.npy');wh=(round(r['wh'][0]/level),round(r['wh'][1]/level))
            if dst.exists():
                assert np.load(dst,mmap_mode='r').shape==(wh[1],wh[0],3);continue
            with Image.open(r['image']) as im:arr=np.array(im.resize(wh))
            np.save(dst,arr)
        print('CAMERA_CACHE_READY',scale,level,len(rows),flush=True)

class LazyCamera:
    cached_id=None
    cached_tensor=None
    last=None
    def __init__(self,row,uid,level=1):
        import torch
        from utils.graphics_utils import getWorld2View2
        self.row=row;self.uid=uid;self.colmap_id=1;self.image_name=row['frame'];self.level=level
        self.image_width,self.image_height=[round(v/level) for v in row['wh']]
        self.width=self.image_width;self.height=self.image_height
        self.K=np.diag([self.width/row['wh'][0],self.height/row['wh'][1],1.])@np.asarray(row['K'])
        w=np.asarray(row['w2c']);self.R=w[:3,:3].T;self.T=w[:3,3]
        self.FoVx=2*math.atan(self.width/(2*self.K[0,0]));self.FoVy=2*math.atan(self.height/(2*self.K[1,1]))
        self.znear=.01;self.zfar=100.
        self.world_view_transform=torch.tensor(getWorld2View2(self.R,self.T)).transpose(0,1).cuda()
        p=np.zeros((4,4));p[0,0]=2*self.K[0,0]/self.width;p[1,1]=2*self.K[1,1]/self.height
        p[0,2]=2*self.K[0,2]/self.width-1;p[1,2]=2*self.K[1,2]/self.height-1
        p[2,2]=self.zfar/(self.zfar-self.znear);p[2,3]=-self.zfar*self.znear/(self.zfar-self.znear);p[3,2]=1
        self.projection_matrix=torch.tensor(p,dtype=torch.float32,device='cuda').T.contiguous()
        self.full_proj_transform=self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0)).squeeze(0)
        self.camera_center=self.world_view_transform.inverse()[3,:3]
        self.rgb_path=Path(row['image']).parents[2]/f'rgb-cache-{level}'/(row['frame']+'.npy')
        header=np.load(self.rgb_path,mmap_mode='r',allow_pickle=False)
        try:assert header.shape==(self.height,self.width,3) and header.dtype==np.uint8
        finally:header._mmap.close()
    @property
    def rgb(self):
        # Each read owns a uint8 array; no camera retains an open file mapping.
        with self.rgb_path.open('rb') as stream:
            return np.load(stream,allow_pickle=False)
    @property
    def original_image(self):
        import torch
        LazyCamera.last=self
        if LazyCamera.cached_id!=id(self):
            LazyCamera.cached_tensor=None
            # Division (not multiplication by reciprocal) matches native PILtoTorch.
            LazyCamera.cached_tensor=torch.from_numpy(self.rgb).to(device='cuda',dtype=torch.float32).div_(255.).permute(2,0,1).contiguous()
            LazyCamera.cached_id=id(self)
        return LazyCamera.cached_tensor

def module_for(scale):
    torch=runtime();run=ROOT/NAMES[scale];rows=read(run/'training-inputs.json')['records'];by_name={r['frame']:r for r in rows}
    # Match the native entrypoint import order (scene imports camera_utils).
    import scene
    import utils.camera_utils as cu
    def load_camera(args,uid,info,level):
        assert args.resolution==1;row=by_name[info.image_name]
        assert (info.width,info.height)==tuple(row['wh'])
        assert np.allclose(info.R,np.asarray(row['w2c'])[:3,:3].T) and np.allclose(info.T,np.asarray(row['w2c'])[:3,3])
        info.image.close()
        return LazyCamera(row,uid,int(level))
    cu.loadCam=load_camera
    native_json=cu.camera_to_JSON
    def json_camera(uid,info):
        out=native_json(uid,info);row=by_name[info.image_name];out.update(K=row['K'],cx=row['K'][0][2],cy=row['K'][1][2],pixel_centers='COLMAP .5 based',split=row['split']);return out
    cu.camera_to_JSON=json_camera
    import scene
    scene.camera_to_JSON=json_camera
    native_scene=scene.Scene
    class RecordedScene(native_scene):
        latest=None
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw);RecordedScene.latest=self
    module=import_file('native_msv2_train',ROOT/'source/msv2/train.py')
    module.Scene=RecordedScene
    module.network_gui.try_connect=lambda:None
    # No GUI socket is needed for local batch runs. Training functions stay native.
    return module,RecordedScene

def options(scale):
    from arguments import ModelParams,OptimizationParams,PipelineParams
    from argparse import Namespace
    parser=argparse.ArgumentParser();lp=ModelParams(parser);op=OptimizationParams(parser);pp=PipelineParams(parser)
    args=parser.parse_args([])
    config=eval((ROOT/'fast.config').read_text(),{'Namespace':Namespace})
    vars(args).update(vars(config));args.imp_metric='outdoor';args.num_depth_factor=1;args.sampling_factor=.6
    args.source_path=str(ROOT/NAMES[scale]/'dataset');args.model_path=str(ROOT/NAMES[scale]/'model')
    args.resolution=1;args.data_device='cpu';args.eval=True
    return args,lp.extract(args),op.extract(args),pp.extract(args)

def save_image(p,tensor):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    arr=np.rint(tensor.detach().clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
    Image.fromarray(arr).save(p)

def preflight(scale):
    torch=runtime();run=ROOT/NAMES[scale];rows=read(run/'training-inputs.json')['records']
    from scene.gaussian_model import GaussianModel
    from scene.dataset_readers import read_points3D_binary
    from utils.graphics_utils import BasicPointCloud
    from gaussian_renderer import render_imp, render_depth
    from fused_ssim import fused_ssim
    args,data,opt,pipe=options(scale)
    xyz,rgb,_=read_points3D_binary(str(run/'dataset/sparse/0/points3D.bin'))
    assert len(xyz)>1000 and np.isfinite(xyz).all()
    centers=np.asarray([np.linalg.inv(r['w2c'])[:3,3] for r in rows if r['split']=='train'])
    extent=float(np.linalg.norm(centers-centers.mean(0),axis=1).max()*1.1)
    g=GaussianModel(0);g.create_from_pcd(BasicPointCloud(xyz,rgb/255.,np.zeros_like(xyz)),extent);g.training_setup(opt)
    checks=[]
    for level in (2,1):
        for index in (0,len(rows)//2,len(rows)-1):
            c=LazyCamera(rows[index],index,level);k=c.K;w=np.asarray(c.row['w2c']);q=xyz@w[:3,:3].T+w[:3,3]
            good=q[:,2]>.2;q=q[good];xx=xyz[good]
            expected=q@k.T;expected=expected[:,:2]/expected[:,2:]-.5
            hp=torch.cat([torch.as_tensor(xx,dtype=torch.float32,device='cuda'),torch.ones((len(xx),1),device='cuda')],1)@c.full_proj_transform
            uv=(hp[:,:2]/(hp[:,3:]+1e-7)+1)*torch.tensor([c.width,c.height],device='cuda')/2-.5
            inside=(expected[:,0]>=0)&(expected[:,0]<c.width)&(expected[:,1]>=0)&(expected[:,1]<c.height)
            error=float(np.max(np.abs(uv.cpu().numpy()[inside]-expected[inside])))
            assert error<.1,(index,level,error)
            assert sha(c.row['image'])==c.row['source_sha256']
            with Image.open(c.row['image']) as im:reference=np.asarray(im.resize((c.width,c.height)))
            assert np.array_equal(c.rgb,reference)
            checks.append(dict(frame=c.image_name,level=level,wh=[c.width,c.height],projection_max_error_px=error,cache_pixel_error=0))
    # At 4x the first native training stage is the official half-size warmup.
    # Validate that actual fresh-model stage; full-size pixel/projection checks
    # above and the separate full-size capacity probe retain their own scope.
    backward_level=2 if scale==4 else 1
    c=LazyCamera(rows[len(rows)//2],0,backward_level);torch.cuda.reset_peak_memory_stats();start=time.monotonic()
    output=render_imp(c,g,pipe,torch.zeros(3,device='cuda'));target=c.original_image.cuda();image=output['render']
    l1=(image-target).abs().mean();loss=.8*l1+.2*(1-fused_ssim(image[None],target[None]));loss.backward();torch.cuda.synchronize()
    assert torch.isfinite(loss) and torch.isfinite(g._xyz.grad).all()
    peak=torch.cuda.max_memory_reserved()/1024**3
    save_image(run/'preflight.png',image)
    result=dict(passed=True,scale=scale,checks=checks,initial_points=len(xyz),native_loss=float(loss),native_backward_seconds=time.monotonic()-start,
        backward_level=backward_level,backward_wh=[c.width,c.height],full_target_wh=rows[0]['wh'],
        peak_reserved_gib=peak,model_and_loss_unmodified=True,training_completed=False)
    save(run/'preflight.json',result);print('STANDARD_PREFLIGHT_PASSED',json.dumps(result),flush=True)

def validate_runtime(scale):
    """Check descriptor ownership and exact target pixels before retrying 4x."""
    import gc,resource
    assert scale==4
    torch=runtime();torch.zeros(1,device='cuda')
    run=ROOT/NAMES[scale];rows=read(run/'training-inputs.json')['records']
    plan=read(ROOT/'plan.json')
    for source in plan['source_checks']:
        assert sha(ROOT/'source'/source['path'])==source['sha256'],source['path']
    count=lambda:len(os.listdir('/proc/self/fd'))
    baseline=count()
    cameras={(level,i):LazyCamera(row,i,level) for level in (1,2) for i,row in enumerate(rows)}
    after_construction=count()
    assert after_construction-baseline<32,(baseline,after_construction)
    checks=[]
    with torch.no_grad():
        for level in (1,2):
            for i in (0,len(rows)//2,len(rows)-1):
                camera=cameras[level,i]
                mapped=np.load(camera.rgb_path,mmap_mode='r',allow_pickle=False)
                try:old=np.array(mapped)
                finally:mapped._mmap.close()
                current=camera.rgb
                assert np.array_equal(old,current)
                expected=torch.from_numpy(old).to(device='cuda',dtype=torch.float32).div_(255.).permute(2,0,1)
                actual=camera.original_image
                assert torch.equal(expected,actual)
                checks.append(dict(frame=camera.image_name,level=level,wh=[camera.width,camera.height],
                    uint8_equal=True,target_float32_bitwise_equal=True,open_file_descriptors=count()))
                del old,current,expected,actual
                print('RUNTIME_TARGET_PARITY',level,camera.image_name,flush=True)
    LazyCamera.cached_tensor=None;LazyCamera.cached_id=None;LazyCamera.last=None
    del camera,cameras;gc.collect();torch.cuda.empty_cache()
    result=dict(passed=True,started=now(),nofile_limits=list(resource.getrlimit(resource.RLIMIT_NOFILE)),
        descriptors_before_cameras=baseline,descriptors_after_camera_construction=after_construction,
        camera_instances=2*len(rows),
        descriptors_after_targets=count(),checks=checks,native_sources_unchanged=True,
        model_and_loss_unmodified=True,optimizer_updates=0)
    probe=import_file('native_fourx_capacity',Path(__file__).with_name('probe_standard_msv2_memory4.py'))
    probe.main()
    capacity=read(ROOT/'native-4x-memory-probe.json')
    assert capacity['state']=='passed',capacity
    result.update(finished=now(),full_resolution_capacity=capacity,descriptors_after_full_backward=count())
    save(ROOT/'runtime-fd-validation.json',result)
    save(RECORD/'四倍运行资源核验.json',result)
    print('FOURX_RUNTIME_VALIDATED',json.dumps(result),flush=True)

def train(scale):
    torch=runtime();run=ROOT/NAMES[scale]
    if scale==4:
        total=torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(FOURX_MEMORY_LIMIT_GIB*1024**3/total)
    assert read(run/'preflight.json')['passed'] and not (run/'model').exists()
    plan=read(ROOT/'plan.json')
    assert sha(ROOT/'fast.config')==plan['fast_sha256']
    for source in plan['source_checks']:
        assert sha(ROOT/'source'/source['path'])==source['sha256'],source['path']
    shutil.copy2(__file__,run/'training-adapter.py')
    m,Scene=module_for(scale);args,data,opt,pipe=options(scale)
    assert args.imp_metric=='outdoor' and args.warn_until_iter==3000 and args.iterations==18000
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.set_device(0)
    save(run/'resolved-options.json',vars(args))
    import resource
    state=dict(state='training',pid=os.getpid(),started=now(),recipe=plan['recipe'],quality_accepted=False,
        nofile_limits=list(resource.getrlimit(resource.RLIMIT_NOFILE)),
        gpu_allocator_limit_gib=FOURX_MEMORY_LIMIT_GIB if scale==4 else None,
        target_layout='contiguous CHW, unchanged float32 pixel values',
        allocator_cache_trim='after each full-resolution backward' if scale==4 else 'native',
        native_sources_unchanged=True,adapter_sha256=sha(run/'training-adapter.py'))
    save(run/'execution.json',state);native_report=m.training_report;start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    def report(tb,iteration,l1,loss,*rest):
        native_report(tb,iteration,l1,loss,*rest)
        c=LazyCamera.last
        if scale==4 and c is not None and c.level==1:
            torch.cuda.empty_cache()
        if iteration%100==0 or iteration in (1,1999,2001,2999,3001,8001,13001,18000):
            g=Scene.latest.gaussians;c=LazyCamera.last
            item=dict(iteration=iteration,points=len(g.get_xyz),loss=float(loss),l1=float(l1),seconds=time.monotonic()-start,updated=now(),wh=[c.width,c.height],image=c.image_name,warmup_level=c.level,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
            item['open_file_descriptors']=len(os.listdir('/proc/self/fd'))
            item['peak_allocated_gib']=torch.cuda.max_memory_allocated()/1024**3
            item['allocated_gib']=torch.cuda.memory_allocated()/1024**3
            item['reserved_gib']=torch.cuda.memory_reserved()/1024**3
            assert math.isfinite(item['loss']);save(run/'progress.json',item)
            with (run/'progress.jsonl').open('a') as f:f.write(json.dumps(item)+'\n')
    m.training_report=report
    try:
        # Identical native training function, loss, optimizer, render and depth kernels.
        m.training(data,opt,pipe,[-1],[18000],[],None,-1,args)
        torch.cuda.synchronize();scene=Scene.latest;g=scene.gaussians
        assert all(torch.isfinite(v).all() for v in (g._xyz,g._opacity,g._scaling,g._rotation,g._features_dc,g._features_rest))
        ply=run/'model/point_cloud/iteration_18000/point_cloud.ply';assert ply.exists()
        restored=m.GaussianModel(3);restored.load_ply(str(ply));restored.active_sh_degree=g.active_sh_degree
        checks=[]
        with torch.no_grad():
            cams=scene.getTestCameras()
            for i in (0,len(cams)//2,len(cams)-1):
                c=cams[i];a=m.render(c,g,pipe,torch.zeros(3,device='cuda'))['render'];b=m.render(c,restored,pipe,torch.zeros(3,device='cuda'))['render']
                d=(a-b).abs();checks.append(dict(frame=c.image_name,max_abs=float(d.max()),mean_abs=float(d.mean())))
                save_image(run/'preview'/f'{c.image_name}.png',a);del a,b,d
        parity=all(c['max_abs']<.005 for c in checks);assert parity,checks
        save(run/'ply-parity.json',dict(passed=parity,checks=checks,sha256=sha(ply)))
        state.update(state='training_completed',finished=now(),iterations=18000,optimizer_updates=17999,points=len(g.get_xyz),seconds=time.monotonic()-start,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,train_frames=len(scene.getTrainCameras()),test_frames=len(scene.getTestCameras()),ply_sha256=sha(ply))
        save(run/'execution.json',state);print('STANDARD_TRAINING_COMPLETED',json.dumps(state),flush=True)
    except BaseException as e:
        state.update(state='failed',finished=now(),error=repr(e));save(run/'execution.json',state);raise

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['init','seed_sr','pixels','sr','cameras','preflight','validate_runtime','train']);p.add_argument('--scale',type=int,choices=[1,2,4],default=1);args=p.parse_args()
    if args.action=='init':initialize()
    elif args.action=='seed_sr':seed_superresolution()
    elif args.action=='pixels':stage_pixels(args.scale)
    elif args.action=='sr':superresolve()
    elif args.action=='cameras':stage_cameras(args.scale)
    elif args.action=='preflight':preflight(args.scale)
    elif args.action=='validate_runtime':validate_runtime(args.scale)
    elif args.action=='train':train(args.scale)
if __name__=='__main__':main()
