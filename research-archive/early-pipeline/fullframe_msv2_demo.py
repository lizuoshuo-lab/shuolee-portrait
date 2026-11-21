"""Fresh full-frame, unmasked MSV2 comparison. No cropped RGB or mask is read."""
from pathlib import Path
from datetime import datetime, timezone
from types import SimpleNamespace
import argparse, copy, hashlib, importlib.util, json, math, os, random, shutil, subprocess, sys, time, traceback
import numpy as np
from PIL import Image

ROOT = Path('/opt/gs-slam-lab/msv2-demo-fullframe-20261104')
PARENT = Path('/opt/gs-slam-lab/static-person-upsampling-20261102/demo-static-input')
UPSTREAM = Path('/opt/gs-slam-lab/msv2-demo-upsample3x-20261104/source')
DEPS = Path('/opt/gs-slam-lab/mini-splatting2-native-001/deps')
RECORD = Path('/mnt/d/00/记录/2026-11-04_MSV2_完整图_原始_插值2倍_超分4倍')
VIDEO = Path('/mnt/d/00/工作/数据集/demo.mp4')
WEIGHT = Path('/opt/gs-slam-lab/model-cache/swinir/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth')
WEIGHT_SHA = '99adfa91350a84c99e946c1eb3d8fce34bc28f57d807b09dc8fe40a316328c0a'
NAMES = {1:'original', 2:'interpolation2x', 4:'swinir4x'}
RUN = ROOT / NAMES[1]
SCALE = 1
SELECTED = [0, 11, 23, 34, 46, 58]
CONFIG = dict(iterations=18000,densify_until_iter=3000,aggressive_clone_from_iter=500,
              aggressive_clone_interval=250,warn_until_iter=0,depth_reinit_iter=2000,
              simp_iteration1=3000,simp_iteration2=8000,imp_metric='outdoor',
              num_depth_factor=1.,sampling_factor=.6)

def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p, obj):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8'); t.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''): h.update(b)
    return h.hexdigest()
def choose(s):
    global RUN,SCALE
    SCALE=s; RUN=ROOT/NAMES[s]
def source_rows():
    rows=read(PARENT/'data/training-inputs.json')['records']
    # Keep only actual frame/camera/RGB metadata, never carry a mask dependency.
    keys=['frame','name','source_frame_index','time_seconds','image','source_sha256','wh','K','K_integer_centers','w2c','camera_kind']
    return [{k:r[k] for k in keys} for r in rows]

def create():
    ROOT.mkdir(exist_ok=False)
    shutil.copy2(__file__,ROOT/'experiment.py')
    shutil.copytree(UPSTREAM,ROOT/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree('/mnt/d/00/工作/SwinIR',ROOT/'swinir-source',ignore=shutil.ignore_patterns('.git','model_zoo','testsets','figs','__pycache__','*.pyc'))
    save(ROOT/'source-manifest.json',{p.relative_to(ROOT/'source').as_posix():sha(p) for p in (ROOT/'source').rglob('*') if p.is_file()})
    save(ROOT/'swinir-source-manifest.json',{p.relative_to(ROOT/'swinir-source').as_posix():sha(p) for p in (ROOT/'swinir-source').rglob('*') if p.is_file()})
    rows=source_rows(); assert len(rows)==59 and all(r['wh']==[2704,1520] for r in rows)
    save(ROOT/'native-inputs.json',dict(records=rows))
    for s,n in NAMES.items():
        (ROOT/n).mkdir();save(ROOT/n/'config.json',CONFIG)
    save(ROOT/'plan.json',dict(created=now(),source_video=str(VIDEO),video_sha256=sha(VIDEO),
        hypothesis='Full RGB scene supervision avoids synthetic background compositing; enlarged full images may improve detail. Quality and geometry remain to be measured.',
        arms={str(s):dict(method=n,wh=[2704*s,1520*s]) for s,n in NAMES.items()},
        frames=59,mask_used=False,alpha_loss=False,foreground_filter=False,background_trained=True,crop_used=False,
        model='Official Mini-Splatting2 msv2, commit 5ec9202c2db7d900728bfc09733e85467cff885e',
        loss='0.8 full-image L1 + 0.2 (1 - full-image fused SSIM)',config=CONFIG,
        fresh_gaussians=True,fresh_optimizer=True,previous_trained_state_loaded=False,
        initialization='Fresh all-image SIFT triangulation with fixed existing scene cameras, plus original untrained scene SfM points; same file for all arms; no masks',
        camera_policy='Full native camera K, first two rows multiplied by scale; integer centers subtract 0.5 after scaling; same w2c',
        memory_policy='Full-resolution rendering; SSIM evaluation/backward row strips with exact 5-pixel neighborhood, depth ray correction in row strips. No image resizing or sampling.',
        fixed=['59 frame identities','w2c','native full K','fresh shared sparse initialization','seed 6060','MSV2 model and schedule'],
        interpretation='Compared with old masked crops, support, initialization and loss all change. Among these three new arms only input resolution/SR method changes.',
        stop_conditions=['Nonfinite gradients or parameters','Camera/pixel contract failure','Unsafe memory demand','18000 labels'],
        quality_accepted=False))
    print('FULLFRAME_CREATED',ROOT,flush=True)

def prepare():
    import cv2
    cv2.setNumThreads(4)
    rows=read(ROOT/'native-inputs.json')['records']
    # Independent FFmpeg decode: no scale/crop filter anywhere in this command.
    indices=[r['source_frame_index'] for r in rows]
    select='+'.join(f'eq(n\\,{i})' for i in indices)
    cmd=['ffmpeg','-nostdin','-v','error','-threads','2','-i',str(VIDEO),'-vf','select='+select,
         '-vsync','0','-f','rawvideo','-pix_fmt','rgb24','pipe:1']
    proc=subprocess.Popen(cmd,stdout=subprocess.PIPE)
    checks=[]; features=[]; images=[]; sift=cv2.SIFT_create(nfeatures=12000)
    for i,r in enumerate(rows):
        payload=proc.stdout.read(2704*1520*3);assert len(payload)==2704*1520*3
        decoded=np.frombuffer(payload,np.uint8).reshape(1520,2704,3)
        rgb=np.asarray(Image.open(r['image']).convert('RGB'));assert np.array_equal(decoded,rgb)
        assert sha(r['image'])==r['source_sha256']
        images.append(rgb)
        key,desc=sift.detectAndCompute(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),None)
        features.append((np.array([p.pt for p in key]),desc))
        checks.append(dict(frame=r['frame'],source_frame_index=r['source_frame_index'],max_pixel_error=0,sha256=r['source_sha256']))
        # Upscale directly from the original complete frame, including background.
        p=ROOT/NAMES[2]/'images'/r['name'];p.parent.mkdir(exist_ok=True)
        Image.fromarray(rgb).resize((5408,3040),Image.Resampling.LANCZOS).save(p)
        if i%10==0 or i==58: print('FULL_NATIVE_VERIFIED',i+1,flush=True)
    assert proc.stdout.read()==b'' and proc.wait()==0
    save(ROOT/'native-pixel-audit.json',dict(passed=True,frames=59,checks=checks,decode_command=cmd,downsampling=False,cropping=False))
    # Match full frames without foreground or background exclusion.
    sfm=np.load(PARENT/'camera/solver-output.npz'); xyz=[sfm['points'].copy()]
    rgb0=np.zeros((len(xyz[0]),3)); counts=np.zeros(len(xyz[0]))
    for ci,pi,uv in zip(sfm['obs_cam'],sfm['obs_point'],sfm['observations']):
        x,y=np.rint(uv).astype(int)
        if 0<=x<2704 and 0<=y<1520: rgb0[pi]+=images[ci][y,x]/255.;counts[pi]+=1
    assert (counts>0).all(); colors=[rgb0/counts[:,None]]
    matches=[];matcher=cv2.FlannBasedMatcher(dict(algorithm=1,trees=5),dict(checks=64))
    for i in range(len(rows)):
        for step in (2,6,12):
            j=i+step
            if j>=len(rows): continue
            k1,d1=features[i];k2,d2=features[j]
            pairs=[m for m,n in matcher.knnMatch(d1,d2,k=2) if m.distance < .7*n.distance]
            if len(pairs)<8:continue
            u=np.array([k1[m.queryIdx] for m in pairs]);v=np.array([k2[m.trainIdx] for m in pairs])
            a=np.array(rows[i]['w2c']);b=np.array(rows[j]['w2c'])
            ki=np.array(rows[i]['K_integer_centers']);kj=np.array(rows[j]['K_integer_centers'])
            h=cv2.triangulatePoints(ki@a[:3],kj@b[:3],u.T,v.T).T
            with np.errstate(divide='ignore',invalid='ignore'):
                pts=h[:,:3]/h[:,3:];ca=pts@a[:3,:3].T+a[:3,3];cb=pts@b[:3,:3].T+b[:3,3]
                qa=ca@ki.T;qb=cb@kj.T
                err=np.maximum(np.linalg.norm(qa[:,:2]/qa[:,2:]-u,axis=1),np.linalg.norm(qb[:,:2]/qb[:,2:]-v,axis=1))
                oa=np.linalg.inv(a)[:3,3];ob=np.linalg.inv(b)[:3,3]
                ra=pts-oa;rb=pts-ob;angle=np.degrees(np.arccos(np.clip((ra*rb).sum(1)/(np.linalg.norm(ra,axis=1)*np.linalg.norm(rb,axis=1)),-1,1)))
                good=np.isfinite(pts).all(1)&(ca[:,2]>.05)&(cb[:,2]>.05)&(err<1.5)&(angle>.5)&(np.linalg.norm(pts,axis=1)<200)
            if good.any():
                pixels=np.rint(u[good]).astype(int);xyz.append(pts[good]);colors.append(images[i][pixels[:,1],pixels[:,0]]/255.)
            matches.append(dict(i=i,j=j,matched=len(pairs),triangulated=int(good.sum())))
        if i%10==0:print('FULLFRAME_TRIANGULATION',i, sum(len(x) for x in xyz),flush=True)
    xyz=np.concatenate(xyz);colors=np.concatenate(colors)
    assert np.isfinite(xyz).all() and np.isfinite(colors).all()
    # Merge repeated triangulations in tiny world-space voxels; no semantic filtering.
    extent=np.linalg.norm(np.percentile(xyz,95,axis=0)-np.percentile(xyz,5,axis=0))
    _,ids=np.unique(np.floor(xyz/(extent*1e-4)).astype(np.int64),axis=0,return_index=True)
    ids=np.sort(ids);xyz=xyz[ids].astype(np.float32);colors=colors[ids].astype(np.float32)
    assert len(xyz)>5000
    np.savez(ROOT/'initial-points.npz',xyz=xyz,rgb=colors)
    save(ROOT/'initialization.json',dict(created=now(),points=len(xyz),sha256=sha(ROOT/'initial-points.npz'),
        fixed_camera_source=str(PARENT/'camera/solver-output.npz'),all_image_features=True,masks_read=False,
        learned_gaussians_loaded=False,match_checks=matches,voxel_size=extent*1e-4))
    for s in (1,2): stage_inputs(s)
    print('FULLFRAME_INPUTS_1_2_READY',len(xyz),flush=True)

def stage_inputs(s):
    rows=copy.deepcopy(read(ROOT/'native-inputs.json')['records'])
    for r in rows:
        r['native_image']=r['image'];r['native_sha256']=r['source_sha256']
        if s>1:r['image']=str(ROOT/NAMES[s]/'images'/r['name'])
        r['wh']=[2704*s,1520*s];r['source_sha256']=sha(r['image'])
        k=np.diag([s,s,1.])@np.array(r['K']);ki=k.copy();ki[:2,2]-=.5
        r['K']=k.tolist();r['K_integer_centers']=ki.tolist();r['scale']=s
        with Image.open(r['image']) as im:assert list(im.size)==r['wh']
    save(ROOT/NAMES[s]/'training-inputs.json',dict(records=rows,mask_used=False,crop_used=False,background_trained=True))

def runtime():
    sys.path[:0]=[str(ROOT/'source'),str(DEPS)]
    import torch
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    return torch

def superresolve(first_only=False):
    torch=runtime();assert sha(WEIGHT)==WEIGHT_SHA
    sys.path.insert(0,str(ROOT/'swinir-source'));import main_test_swinir as official
    args=SimpleNamespace(task='real_sr',scale=4,large_model=True,model_path=str(WEIGHT),tile=256,tile_overlap=32)
    free,total=torch.cuda.mem_get_info();assert free>7*1024**3;torch.cuda.set_per_process_memory_fraction(8*1024**3/total)
    model=official.define_model(args).eval().cuda()
    p=ROOT/'superresolution.json';state=read(p) if p.exists() else dict(state='running',started=now(),records=[])
    rows=read(ROOT/'native-inputs.json')['records'];done={x['frame']:x for x in state['records']}
    with torch.inference_mode():
        for r in rows[:1] if first_only else rows:
            outpath=ROOT/NAMES[4]/'images'/r['name'];outpath.parent.mkdir(exist_ok=True)
            if r['frame'] in done:
                assert sha(outpath)==done[r['frame']]['output_sha256'];continue
            assert sha(r['image'])==r['source_sha256'];t=time.monotonic()
            rgb=np.array(Image.open(r['image']).convert('RGB'))
            x=torch.from_numpy(rgb).to(device='cuda',dtype=torch.float32).permute(2,0,1)[None]/255.
            h,w=rgb.shape[:2];hp=(h//8+1)*8-h;wp=(w//8+1)*8-w
            x=torch.cat([x,torch.flip(x,[2])],2)[:,:,:h+hp,:]
            x=torch.cat([x,torch.flip(x,[3])],3)[:,:,:,:w+wp]
            pred=official.test(x,model,args,8)[...,:h*4,:w*4]
            assert pred.shape==(1,3,6080,10816) and torch.isfinite(pred).all()
            arr=pred[0].clamp(0,1).permute(1,2,0).cpu().numpy()
            arr=np.rint(arr*255).astype(np.uint8);Image.fromarray(arr).save(outpath)
            item=dict(frame=r['frame'],native_sha256=r['source_sha256'],output_sha256=sha(outpath),wh=[10816,6080],seconds=time.monotonic()-t)
            state['records'].append(item);state.update(state='probe_completed' if first_only else 'running',updated=now(),processed=len(state['records']),weight_sha256=WEIGHT_SHA,full_frame=True,mask_used=False,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
            save(p,state);print('FULLFRAME_SWINIR',len(state['records']),json.dumps(item),flush=True)
            del x,pred,arr
    if not first_only:
        assert len(state['records'])==59;state.update(state='completed',finished=now());save(p,state);stage_inputs(4)

class Camera:
    cache={}
    def __init__(self,row,uid):
        import torch
        self.row=row;self.uid=uid;self.colmap_id=uid;self.image_name=row['frame']
        self.image_width,self.image_height=row['wh'];self.width=self.image_width;self.height=self.image_height
        # The upstream MSV2 depth kernel hardcodes near=.01 and far=100.
        self.K=np.array(row['K']);w=np.array(row['w2c']);self.R=w[:3,:3].T;self.T=w[:3,3];self.znear=.01;self.zfar=100.
        self.FoVx=2*math.atan(self.width/(2*self.K[0,0]));self.FoVy=2*math.atan(self.height/(2*self.K[1,1]))
        self.world_view_transform=torch.tensor(w,dtype=torch.float32,device='cuda').T.contiguous()
        p=np.zeros((4,4));p[0,0]=2*self.K[0,0]/self.width;p[1,1]=2*self.K[1,1]/self.height
        p[0,1]=2*self.K[0,1]/self.width;p[0,2]=2*self.K[0,2]/self.width-1;p[1,2]=2*self.K[1,2]/self.height-1
        p[2,2]=self.zfar/(self.zfar-self.znear);p[2,3]=-self.zfar*self.znear/(self.zfar-self.znear);p[3,2]=1
        self.projection_matrix=torch.tensor(p,dtype=torch.float32,device='cuda').T.contiguous()
        self.full_proj_transform=self.world_view_transform@self.projection_matrix
        self.camera_center=self.world_view_transform.inverse()[3,:3]
        # Memory mapped uncompressed RGB avoids keeping 59 full float images in RAM.
        cache=Path(row['image']).parent.parent/'rgb-cache'/Path(row['image']).with_suffix('.npy').name
        if cache.exists():self.rgb=np.load(cache,mmap_mode='r')
        else:self.rgb=np.asarray(Image.open(row['image']).convert('RGB'))
        assert self.rgb.shape==(self.height,self.width,3)
    @property
    def original_image(self):
        import torch
        if id(self) not in Camera.cache:
            Camera.cache.clear()
            Camera.cache[id(self)]=torch.from_numpy(np.array(self.rgb,copy=True)).to(device='cuda',dtype=torch.float32).permute(2,0,1).contiguous().div_(255.)
        return Camera.cache[id(self)]

class Scene:
    latest=None
    def __init__(self,args,gaussians,**unused):
        from utils.graphics_utils import BasicPointCloud
        rows=read(RUN/'training-inputs.json')['records']
        self.model_path=args.model_path;self.gaussians=gaussians;self.cameras=[Camera(r,i) for i,r in enumerate(rows)]
        centers=np.array([np.linalg.inv(np.array(r['w2c']))[:3,3] for r in rows])
        self.cameras_extent=float(np.linalg.norm(centers-centers.mean(0),axis=1).max()*1.1)
        init=np.load(ROOT/'initial-points.npz')
        gaussians.create_from_pcd(BasicPointCloud(points=init['xyz'],colors=init['rgb'],normals=np.zeros_like(init['xyz'])),self.cameras_extent)
        random.shuffle(self.cameras);Scene.latest=self
    def getTrainCameras(self,*args,**kwargs):return self.cameras
    def getTrainCameras_warn_up(self,*args,**kwargs):return self.cameras
    def getTestCameras(self,*args,**kwargs):return []
    def save(self,iteration):
        p=Path(self.model_path)/f'point_cloud/iteration_{iteration}/point_cloud.ply'
        self.gaussians.save_ply(str(p));save(p.parent/'snapshot.json',dict(iteration_label=iteration,optimizer_updates=iteration-1,points=len(self.gaussians.get_xyz),sha256=sha(p)))
        if iteration in (1000,18000):
            import torch
            from gaussian_renderer import render
            pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
            native=read(ROOT/'native-inputs.json')['records']
            with torch.no_grad():
                for i in SELECTED:
                    c=Camera(native[i],i);im=render(c,self.gaussians,pipe,torch.zeros(3,device='cuda'))['render']
                    save_image(p.parent/(c.image_name+'.png'),im)
            if iteration==18000:
                from scene import GaussianModel
                restored=GaussianModel(self.gaussians.max_sh_degree);restored.load_ply(str(p));restored.active_sh_degree=self.gaussians.active_sh_degree
                cams={c.uid:c for c in self.cameras};checks=[]
                with torch.no_grad():
                    for i in (0,34,58):
                        a=render(cams[i],self.gaussians,pipe,torch.zeros(3,device='cuda'))['render']
                        b=render(cams[i],restored,pipe,torch.zeros(3,device='cuda'))['render']
                        d=(a-b).abs();checks.append(dict(frame=cams[i].image_name,wh=[cams[i].width,cams[i].height],max_abs=float(d.max()),mean_abs=float(d.mean())))
                        del a,b,d
                passed=all(x['max_abs']<.005 for x in checks)
                save(p.parent/'ply-parity.json',dict(passed=passed,checks=checks,sha256=sha(p)))
                assert passed,checks

def save_image(p,t):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    a=np.rint(t.detach().clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
    Image.fromarray(a).save(p)

def options():
    from arguments import ModelParams,OptimizationParams,PipelineParams
    parser=argparse.ArgumentParser();lp=ModelParams(parser);op=OptimizationParams(parser);pp=PipelineParams(parser)
    args=parser.parse_args([])
    for k,v in CONFIG.items():setattr(args,k,v)
    args.source_path=str(RUN);args.model_path=str(RUN/'model');args.resolution=1;args.data_device='cpu'
    args.imp_metric=CONFIG['imp_metric'];args.num_depth_factor=CONFIG['num_depth_factor'];args.sampling_factor=CONFIG['sampling_factor']
    return args,lp.extract(args),op.extract(args),pp.extract(args)

def bounded_ssim(a,b):
    """Identical full-image SSIM, evaluate/backprop exact 11x11 neighborhoods in strips."""
    import torch
    from fused_ssim_cuda import fusedssim, fusedssim_backward
    class Strips(torch.autograd.Function):
        @staticmethod
        def forward(ctx,x,y):
            ctx.save_for_backward(x,y);h=x.shape[-2];total=torch.zeros((),device=x.device,dtype=torch.float64)
            for y0 in range(0,h,128):
                y1=min(h,y0+128);a0=max(0,y0-5);a1=min(h,y1+5)
                m,*_=fusedssim(.01**2,.03**2,x[:,:,a0:a1].contiguous(),y[:,:,a0:a1].contiguous(),False)
                total+=m[:,:,y0-a0:y1-a0].sum(dtype=torch.float64)
            return (total/x.numel()).to(x.dtype)
        @staticmethod
        def backward(ctx,g):
            x,y=ctx.saved_tensors;h=x.shape[-2];grad=torch.zeros_like(x)
            for y0 in range(0,h,128):
                y1=min(h,y0+128);a0=max(0,y0-5);a1=min(h,y1+5)
                xx=x[:,:,a0:a1].contiguous();yy=y[:,:,a0:a1].contiguous()
                m,d1,d2,d3=fusedssim(.01**2,.03**2,xx,yy,True)
                dm=torch.zeros_like(m);dm[:,:,y0-a0:y1-a0]=g/x.numel()
                patch=fusedssim_backward(.01**2,.03**2,xx,yy,dm,d1,d2,d3)
                grad[:,:,a0:a1]+=patch
            return grad,None
    return Strips.apply(a,b)

def full_image_loss(a,b,weight=.2):
    """The unchanged full-frame L1/DSSIM objective with one accumulated image gradient."""
    import torch
    from fused_ssim_cuda import fusedssim, fusedssim_backward
    class Photo(torch.autograd.Function):
        @staticmethod
        def forward(ctx,x,y):
            ctx.save_for_backward(x,y);n=x.numel();h=x.shape[-2]
            sl=torch.zeros((),device=x.device,dtype=torch.float64);ss=torch.zeros_like(sl)
            for y0 in range(0,h,128):
                y1=min(h,y0+128);a0=max(0,y0-5);a1=min(h,y1+5)
                xx=x[None,:,a0:a1].contiguous();yy=y[None,:,a0:a1].contiguous()
                m,*_=fusedssim(.01**2,.03**2,xx,yy,False)
                ss+=m[:,:,y0-a0:y1-a0].sum(dtype=torch.float64)
                sl+=(x[:,y0:y1]-y[:,y0:y1]).abs().sum(dtype=torch.float64)
            l1=(sl/n).to(x.dtype);loss=((1-weight)*sl/n+weight*(1-ss/n)).to(x.dtype)
            return l1,loss
        @staticmethod
        def backward(ctx,gl1,gloss):
            x,y=ctx.saved_tensors;n=x.numel();h=x.shape[-2];grad=torch.zeros_like(x)
            for y0 in range(0,h,128):
                y1=min(h,y0+128);a0=max(0,y0-5);a1=min(h,y1+5)
                xx=x[None,:,a0:a1].contiguous();yy=y[None,:,a0:a1].contiguous()
                m,d1,d2,d3=fusedssim(.01**2,.03**2,xx,yy,True)
                dm=torch.zeros_like(m);dm[:,:,y0-a0:y1-a0]=-gloss*weight/n
                grad[:,a0:a1]+=fusedssim_backward(.01**2,.03**2,xx,yy,dm,d1,d2,d3)[0]
                grad[:,y0:y1]+=(x[:,y0:y1]-y[:,y0:y1]).sign()*((gl1+(1-weight)*gloss)/n)
            return grad,None
    return Photo.apply(a,b)

def fullk_depth(camera,g,pipe,bg,**kwargs):
    import torch
    from gaussian_renderer import render_depth
    result=render_depth(camera,g,pipe,bg,**kwargs)
    pts=result['out_pts'];alpha=result['accum_alpha']
    w=torch.tensor(camera.row['w2c'],dtype=torch.float64,device='cuda');ki=torch.tensor(np.linalg.inv(camera.K),dtype=torch.float64,device='cuda')
    # In-place row correction: identical ray formula without full-frame float64 temporaries.
    for y0 in range(0,camera.height,64):
        y1=min(camera.height,y0+64);p=pts[:,y0:y1].permute(1,2,0)
        cp=p.double()@w[:3,:3].T+w[:3,3]
        good=torch.isfinite(cp).all(2)&(cp[:,:,2]>.01)&(p.square().sum(2)>0)
        yy,xx=torch.meshgrid(torch.arange(y0,y1,device='cuda',dtype=torch.float64)+.5,torch.arange(camera.width,device='cuda',dtype=torch.float64)+.5,indexing='ij')
        rays=torch.stack((xx,yy,torch.ones_like(xx)),-1)@ki.T
        corrected=((rays*cp[:,:,2:3]-w[:3,3])@w[:3,:3]).float()
        pts[:,y0:y1]=torch.where(good[:,:,None],corrected,torch.zeros_like(corrected)).permute(2,0,1)
        aa=alpha.reshape(camera.height,camera.width)[y0:y1];aa[~good]=1.
    return result

def loss_equivalence():
    torch=runtime();from fused_ssim import fused_ssim
    torch.manual_seed(17);a=torch.rand((1,3,337,491),device='cuda',requires_grad=True);b=torch.rand_like(a)
    x=fused_ssim(a,b);gx=torch.autograd.grad(x,a)[0];y=bounded_ssim(a,b);gy=torch.autograd.grad(y,a)[0]
    err=float((gx-gy).abs().max());rel=float((gx-gy).abs().mean()/gx.abs().mean())
    assert abs(float(x-y))<1e-6 and err<1e-7 and rel<1e-4,(float(x-y),err,rel)
    from utils.loss_utils import l1_loss
    ref=.8*l1_loss(a[0],b[0])+.2*(1-fused_ssim(a,b));refgrad=torch.autograd.grad(ref,a)[0]
    l,loss=full_image_loss(a[0],b[0]);newgrad=torch.autograd.grad(loss,a)[0]
    perr=float((refgrad-newgrad).abs().max());prel=float((refgrad-newgrad).abs().mean()/refgrad.abs().mean())
    assert abs(float(loss-ref))<1e-6 and perr<1e-7 and prel<1e-4
    save(ROOT/'loss-equivalence.json',dict(passed=True,value_error=abs(float(x-y)),gradient_max_error=err,gradient_mean_relative_error=rel,
        combined_photo_value_error=abs(float(loss-ref)),combined_photo_gradient_max_error=perr,combined_photo_gradient_mean_relative_error=prel,full_pixels=True,mask_used=False))

def preflight(first4=False,stress=False):
    torch=runtime();loss_equivalence();verify_source()
    from scene import GaussianModel
    from gaussian_renderer import render_imp,render_simp
    from utils.loss_utils import l1_loss
    free,total=torch.cuda.mem_get_info();assert free>10*1024**3;torch.cuda.set_per_process_memory_fraction(12*1024**3/total)
    args,data,opt,pipe=options();g=GaussianModel(0)
    if first4:
        from utils.graphics_utils import BasicPointCloud
        r=copy.deepcopy(read(ROOT/'native-inputs.json')['records'][0]);r['image']=str(ROOT/NAMES[4]/'images'/r['name']);r['wh']=[10816,6080];r['K']=(np.diag([4,4,1])@np.array(r['K'])).tolist()
        c=Camera(r,0);init=np.load(ROOT/'initial-points.npz');g.create_from_pcd(BasicPointCloud(points=init['xyz'],colors=init['rgb'],normals=np.zeros_like(init['xyz'])),1.)
        cams=[c]
        if stress:
            # Read-only capacity check, never used to initialize a new training run.
            stress_ply=ROOT/'original/model/point_cloud/iteration_3000/point_cloud.ply'
            g.load_ply(str(stress_ply));g.active_sh_degree=0
    else:
        scene=Scene(data,g);allcams={c.uid:c for c in scene.cameras};cams=[allcams[i] for i in [0,29,58]]
    g.training_setup(opt);g.init_culling(59);bg=torch.zeros(3,device='cuda');checks=[]
    for c in cams:
        pkg=render_imp(c,g,pipe,bg,culling=g._culling[:,c.uid]);im=pkg['render'];gt=c.original_image
        ll1,loss=full_image_loss(im,gt);loss.backward()
        assert tuple(im.shape)==(3,c.height,c.width) and torch.isfinite(loss) and torch.isfinite(g._xyz.grad).all()
        checks.append(dict(frame=c.image_name,wh=[c.width,c.height],loss=float(loss),finite_gradient=True))
        del pkg,im,gt,loss;g.optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        c=cams[-1];d=fullk_depth(c,g,pipe,bg,culling=g._culling[:,c.uid])
        pts=d['out_pts'][:,::23,::23].permute(1,2,0).reshape(-1,3)
        yy,xx=torch.meshgrid(torch.arange(0,c.height,23,device='cuda'),torch.arange(0,c.width,23,device='cuda'),indexing='ij')
        expected=torch.stack((xx,yy),-1).reshape(-1,2);valid=pts.square().sum(1)>0
        w=torch.tensor(c.row['w2c'],dtype=torch.float64,device='cuda');k=torch.tensor(c.K,dtype=torch.float64,device='cuda')
        p=pts[valid].double();cp=p@w[:3,:3].T+w[:3,3];q=cp@k.T;uv=q[:,:2]/q[:,2:]-.5
        residual=(uv-expected[valid]).abs();error=float(residual.max())
        # Quantized world coordinates lose sub-pixel precision extremely near the camera.
        # Check the error against an independent half-ULP projection bound, and <0.05 native pixels.
        spacing=torch.from_numpy(np.abs(np.spacing(pts[valid].cpu().numpy()))).to(device='cuda',dtype=torch.float64)*.5
        jac=(k[None,:2,:]/q[:,2,None,None] - q[:,:2,None]*k[None,2:3,:]/q[:,2,None,None].square())@w[:3,:3]
        bound=(jac.abs()*spacing[:,None,:]).sum(2)
        quantization_excess=float((residual-bound).max())
        assert quantization_excess<1e-5 and error/SCALE<.05,(error,quantization_excess)
        del d,pts,q,uv
        simp=render_simp(c,g,pipe,bg,culling=g._culling[:,c.uid]);assert torch.isfinite(simp['accum_weights']).all()
    save(RUN/('large-scene-capacity.json' if stress else 'first-frame-preflight.json' if first4 else 'preflight.json'),dict(passed=True,created=now(),checks=checks,depth_ray_error_px=error,depth_error_native_pixels=error/SCALE,
        float32_quantization_bound_excess_px=quantization_excess,depth_check_note='Double precision projection of final float32 world points; errors bounded by world-coordinate half-ULP and 0.05 native pixels',
        peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,optimizer_updates=0,mask_used=False,stress_point_count=len(g.get_xyz),training_initialization_modified=False))
    print('FULLFRAME_PREFLIGHT_PASSED',SCALE,flush=True)

def verify_source():
    for p,h in read(ROOT/'source-manifest.json').items():assert sha(ROOT/'source'/p)==h,p

def verify_inputs():
    native=read(ROOT/'native-inputs.json')['records'];rows=read(RUN/'training-inputs.json')['records'];assert len(rows)==len(native)==59
    for r,n in zip(rows,native):
        assert not any('mask' in k for k in r)
        assert r['frame']==n['frame'] and r['source_frame_index']==n['source_frame_index'] and r['w2c']==n['w2c']
        assert r['wh']==[2704*SCALE,1520*SCALE] and sha(r['image'])==r['source_sha256']
        np.testing.assert_array_equal(np.diag([SCALE,SCALE,1])@np.array(n['K']),r['K'])
        ki=np.array(r['K']);ki[:2,2]-=.5;np.testing.assert_array_equal(ki,r['K_integer_centers'])
    assert sha(ROOT/'initial-points.npz')==read(ROOT/'initialization.json')['sha256']
    save(RUN/'input-contract.json',dict(passed=True,frames=59,wh=[2704*SCALE,1520*SCALE],full_image=True,mask_used=False,background_trained=True,
        camera_scaled=True,extrinsics_unchanged=True,native_pixel_audit_passed=read(ROOT/'native-pixel-audit.json')['passed'],
        adapter_sha256=sha(__file__),initial_points_sha256=sha(ROOT/'initial-points.npz')))

def cache_images():
    rows=read(RUN/'training-inputs.json')['records']
    if SCALE==1:return
    p=RUN/'rgb-cache';p.mkdir(exist_ok=True)
    for r in rows:
        dst=p/Path(r['name']).with_suffix('.npy')
        if not dst.exists():np.save(dst,np.asarray(Image.open(r['image']).convert('RGB')))

def train():
    torch=runtime();verify_source();verify_inputs();assert read(RUN/'preflight.json')['passed'];assert read(ROOT/'loss-equivalence.json')['passed']
    assert not (RUN/'model').exists(),'Fresh run only'
    if Path(__file__).resolve() != (RUN/'experiment.py').resolve():shutil.copy2(__file__,RUN/'experiment.py')
    free,total=torch.cuda.mem_get_info();assert free>10*1024**3;torch.cuda.set_per_process_memory_fraction(12*1024**3/total)
    random.seed(6060);np.random.seed(6060);torch.manual_seed(6060)
    path=ROOT/'source/msv2/train.py'
    source=path.read_text();old='''        Ll1 = l1_loss(image, gt_image)
        ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)'''
    assert source.count(old)==1
    adapted=source.replace(old,'        Ll1, loss = full_image_loss(image, gt_image, opt.lambda_dssim)')
    import types,difflib
    (RUN/'training-memory-adapter.diff').write_text(''.join(difflib.unified_diff(source.splitlines(True),adapted.splitlines(True),fromfile='upstream/train.py',tofile='memory_adapter/train.py')))
    m=types.ModuleType('fullframe_upstream');m.__file__=str(path);m.full_image_loss=full_image_loss;exec(compile(adapted,str(path),'exec'),m.__dict__)
    m.Scene=Scene;m.render_depth=fullk_depth;m.network_gui.try_connect=lambda:None
    start=time.monotonic()
    def report(tb,iteration,Ll1,loss,*unused):
        assert torch.isfinite(loss),'Nonfinite loss'
        if iteration%100==0 or iteration==1:
            g=Scene.latest.gaussians
            out=dict(state='training',updated=now(),scale=SCALE,iteration_label=iteration,optimizer_updates=iteration-1,points=len(g.get_xyz),loss=float(loss),l1=float(Ll1),seconds=time.monotonic()-start,gpu_reserved_gib=torch.cuda.memory_reserved()/1024**3,quality_accepted=False)
            save(RUN/'progress.json',out);print(json.dumps(out),flush=True)
    m.training_report=report;args,data,opt,pipe=options()
    save(RUN/'execution.json',dict(state='running',started=now(),pid=os.getpid(),mask_used=False))
    try:
        m.training(data,opt,pipe,[],[1000,3000,8000,18000],[],None,-1,args)
        g=Scene.latest.gaussians;torch.cuda.synchronize();assert torch.isfinite(g.get_xyz).all()
        save(RUN/'execution.json',dict(state='training_completed',finished=now(),iterations=18000,optimizer_updates=17999,points=len(g.get_xyz),seconds=time.monotonic()-start,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,mask_used=False,quality_accepted=False))
    except BaseException:
        save(RUN/'execution.json',dict(state='failed',finished=now(),error=traceback.format_exc(),quality_accepted=False));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['create','prepare','sr-probe','superresolve','preflight','first4','stress4','cache','train','loss-check']);p.add_argument('--scale',type=int,choices=[1,2,4],default=1);a=p.parse_args();choose(a.scale)
    if a.action=='sr-probe':superresolve(True)
    elif a.action=='first4':preflight(True)
    elif a.action=='stress4':preflight(True,True)
    elif a.action=='cache':cache_images()
    elif a.action=='loss-check':loss_equivalence()
    else:globals()[a.action]()
