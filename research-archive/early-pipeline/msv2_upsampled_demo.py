"""Run the official MSV2 optimizer on the existing 3x demo pixel/camera contract.

The adapter supplies full off-centre intrinsics and keeps images on the CPU.
The upstream training loop, loss, CUDA rasterizers and densification are frozen.
"""
from pathlib import Path
from datetime import datetime, timezone
from types import SimpleNamespace
from collections import OrderedDict
import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import time
import traceback

import numpy as np
from PIL import Image

RUN = Path('/opt/gs-slam-lab/msv2-demo-upsample3x-20261104')
BASE = Path('/opt/gs-slam-lab/static-person-upsampling-20261102/demo')
UPSTREAM = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/repos/mini-splatting2')
DEPS = Path('/opt/gs-slam-lab/mini-splatting2-native-001/deps')
RECORD = Path('/mnt/d/00/记录/2026-11-04_MSV2_人体3倍上采样/demo')
SELECTED = [0, 11, 23, 34, 46, 58]


def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    tmp.replace(p)
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4*1024**2), b''): h.update(b)
    return h.hexdigest()


def prepare():
    RUN.mkdir(exist_ok=False)
    shutil.copytree(UPSTREAM, RUN/'source', ignore=shutil.ignore_patterns('.git', '__pycache__', 'assets', '*.pyc', 'build'))
    shutil.copy2(__file__, RUN/'experiment.py')
    rows = read(BASE/'upsampled3x/data/training-inputs.json')['records']
    native = read(BASE/'original/data/training-inputs.json')['records']
    assert len(rows) == len(native) == 59
    for r, b in zip(rows, native):
        assert r['frame'] == b['frame'] and r['w2c'] == b['w2c']
        assert r['wh'] == [1428, 2880] and b['wh'] == [476, 960]
        assert sha(r['image']) == r['source_sha256'] and sha(r['mask']) == r['mask_sha256']
        np.testing.assert_allclose(np.array(r['K']), np.diag([3.,3.,1.])@np.array(b['K']), atol=1e-10, rtol=0)
        with Image.open(r['image']) as im: assert im.size == (1428,2880)
    save(RUN/'training-inputs.json', dict(records=rows, coordinate_contract='K in pixel-corner coordinates'))
    seeds = BASE/'original/clothing-init/points.npz'
    xyz = np.load(seeds, allow_pickle=False)['xyz'].copy()
    assert xyz.shape == (200000,3)
    colors = np.zeros_like(xyz, dtype=np.float64); count = np.zeros(len(xyz))
    for r in native:
        w2c=np.array(r['w2c']); k=np.array(r['K'])
        cp=xyz@w2c[:3,:3].T+w2c[:3,3]
        q=cp@k.T; uv=q[:,:2]/q[:,2:3]-.5
        ij=np.rint(uv).astype(np.int64)
        w,h=r['wh']; good=(cp[:,2]>.01)&(ij[:,0]>=0)&(ij[:,0]<w)&(ij[:,1]>=0)&(ij[:,1]<h)
        ids=np.flatnonzero(good)
        with Image.open(r['image']) as im: rgb=np.asarray(im.convert('RGB'))
        with Image.open(r['mask']) as im: mask=np.asarray(im.convert('L'))
        ids=ids[mask[ij[ids,1],ij[ids,0]]>127]
        colors[ids]+=rgb[ij[ids,1],ij[ids,0]]/255.; count[ids]+=1
    colors[count>0]/=count[count>0,None]; colors[count==0]=.5
    np.savez_compressed(RUN/'initial-points.npz', xyz=xyz, rgb=colors.astype(np.float32))
    config=dict(iterations=18000,densify_until_iter=3000,aggressive_clone_from_iter=500,
                aggressive_clone_interval=250,warn_until_iter=0,depth_reinit_iter=2000,
                simp_iteration1=3000,simp_iteration2=8000)
    save(RUN/'config.json',config)
    source_hashes={p.relative_to(RUN/'source').as_posix():sha(p) for p in (RUN/'source').rglob('*') if p.is_file()}
    save(RUN/'source-manifest.json',source_hashes)
    save(RUN/'plan.json',dict(created=now(),subject='demo',method='Mini-Splatting2 / msv2',
        upstream='https://github.com/fatPeter/mini-splatting2',commit='5ec9202c2db7d900728bfc09733e85467cff885e',
        inputs='Same 59 native demo crops enlarged directly 3x with Lanczos; bilinear masks',
        training_wh=[1428,2880],native_wh=[476,960],pixel_count_scale=9,learned_superresolution=False,
        source_video=str(Path('/mnt/d/00/工作/数据集/demo.mp4')),
        source_video_sha256=sha('/mnt/d/00/工作/数据集/demo.mp4'),
        camera_policy='All original full K and background-estimated w2c retained; K first two rows scaled 3x',
        initialization='Same unoptimized 200000-point silhouette-hull seed as the prior input-resolution experiment; newly sampled source RGB, new Gaussian scales/opacity/SH and empty optimizer',
        cached_unoptimized_hull=True,cached_hull_sha256=sha(seeds),old_trained_gaussians_loaded=False,
        prior_subject_fit_or_mesh_optimized=False,source_originals_changed=False,
        loss='Upstream MSV2 0.8 L1 + 0.2 (1 - fused SSIM), original RGB alpha-composited onto black',
        comparability='Backend/workflow comparison; upstream MSV2 uses whole composite-image loss, unlike the previous masked static sofa trainer. Not a single-variable causal ablation.',
        official_fast_config=read_fast(),config=config,
        adapter_changes=['Full off-centre projection matrix','Explicit full 3x pixels at every iteration (no half-size warmup)',
                         'CPU image storage with a bounded GPU cache','JSON camera/input adapter and progress/snapshot output'],
        stop_conditions=['Nonfinite loss/parameters','CUDA error or insufficient memory','18000 iteration labels maximum'],
        official_last_iteration_has_no_optimizer_step=True,expected_optimizer_updates=17999,
        quality_accepted=False,production_deployment_changed=False))
    print('PREPARED', RUN, flush=True)


def read_fast(): return (UPSTREAM/'config/fast').read_text(encoding='utf-8').strip()


def runtime():
    sys.path[:0]=[str(RUN/'source'),str(DEPS)]
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    return torch


class Camera:
    cache=OrderedDict()
    def __init__(self,row,uid):
        import torch
        self.row=row;self.uid=uid;self.colmap_id=uid;self.image_name=row['frame']
        self.image_width,self.image_height=row['wh'];self.width=self.image_width;self.height=self.image_height
        self.K=np.array(row['K']); w=np.array(row['w2c'])
        self.R=w[:3,:3].T;self.T=w[:3,3];self.znear=.01;self.zfar=100.
        self.FoVx=2*math.atan(self.width/(2*self.K[0,0]));self.FoVy=2*math.atan(self.height/(2*self.K[1,1]))
        self.world_view_transform=torch.tensor(w,dtype=torch.float32,device='cuda').T.contiguous()
        p=np.zeros((4,4));p[0,0]=2*self.K[0,0]/self.width;p[1,1]=2*self.K[1,1]/self.height
        p[0,1]=2*self.K[0,1]/self.width;p[0,2]=2*self.K[0,2]/self.width-1;p[1,2]=2*self.K[1,2]/self.height-1
        p[2,2]=self.zfar/(self.zfar-self.znear);p[2,3]=-self.zfar*self.znear/(self.zfar-self.znear);p[3,2]=1
        self.projection_matrix=torch.tensor(p,dtype=torch.float32,device='cuda').T.contiguous()
        self.full_proj_transform=self.world_view_transform@self.projection_matrix
        self.camera_center=self.world_view_transform.inverse()[3,:3]
        with Image.open(row['image']) as im:rgb=np.array(im.convert('RGB'),copy=True)
        with Image.open(row['mask']) as im:mask=np.array(im.convert('L'),copy=True)
        self.rgb=rgb;self.mask=mask
    @property
    def original_image(self):
        import torch
        key=id(self)
        if key not in Camera.cache:
            while len(Camera.cache)>=2:Camera.cache.popitem(last=False)
            rgb=torch.from_numpy(self.rgb).to('cuda',dtype=torch.float32).permute(2,0,1)/255.
            a=torch.from_numpy(self.mask).to('cuda',dtype=torch.float32)[None]/255.
            Camera.cache[key]=rgb*a
        Camera.cache.move_to_end(key)
        return Camera.cache[key]


class Scene:
    latest=None
    def __init__(self,args,gaussians,**kwargs):
        from utils.graphics_utils import BasicPointCloud
        self.model_path=args.model_path;self.gaussians=gaussians
        rows=read(RUN/'training-inputs.json')['records']
        self.cameras=[Camera(r,i) for i,r in enumerate(rows)]
        centers=np.array([np.linalg.inv(np.array(r['w2c']))[:3,3] for r in rows])
        self.cameras_extent=float(np.linalg.norm(centers-centers.mean(axis=0),axis=1).max()*1.1)
        initial=np.load(RUN/'initial-points.npz',allow_pickle=False)
        gaussians.create_from_pcd(BasicPointCloud(points=initial['xyz'],colors=initial['rgb'],normals=np.zeros_like(initial['xyz'])),self.cameras_extent)
        random.shuffle(self.cameras)
        Scene.latest=self
    def getTrainCameras(self,scale=1.0):return self.cameras
    def getTrainCameras_warn_up(self,*args,**kwargs):return self.cameras
    def getTestCameras(self,scale=1.0):return []
    def save(self,iteration):
        p=Path(self.model_path)/f'point_cloud/iteration_{iteration}/point_cloud.ply'
        self.gaussians.save_ply(str(p))
        save(p.parent/'snapshot.json',dict(iteration_label=iteration,optimizer_updates=iteration-1,points=len(self.gaussians.get_xyz),sha256=sha(p),quality_accepted=False))
        if iteration in (1000,18000):
            import torch
            from gaussian_renderer import render
            pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
            cams={c.uid:c for c in self.cameras}
            restored=None;parity=[]
            if iteration==18000:
                from scene import GaussianModel
                restored=GaussianModel(self.gaussians.max_sh_degree)
                restored.load_ply(str(p))
            for i in SELECTED:
                with torch.no_grad():im=render(cams[i],self.gaussians,pipe,torch.zeros(3,device='cuda'))['render']
                save_image(p.parent/(cams[i].image_name+'.png'),im)
                if restored is not None:
                    with torch.no_grad():back=render(cams[i],restored,pipe,torch.zeros(3,device='cuda'))['render']
                    diff=(im-back).abs()
                    parity.append(dict(frame=cams[i].image_name,mean_abs=float(diff.mean()),max_abs=float(diff.max())))
            if restored is not None:
                save(p.parent/'ply-parity.json',dict(checks=parity,passed=all(x['max_abs']<.005 for x in parity),sha256=sha(p)))


def save_image(p,t):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    a=(t.detach().clamp(0,1).permute(1,2,0).cpu().numpy()*255+.5).astype(np.uint8)
    Image.fromarray(a).save(p)


def options(iterations=18000):
    from arguments import ModelParams,OptimizationParams,PipelineParams
    parser=argparse.ArgumentParser();lp=ModelParams(parser);op=OptimizationParams(parser);pp=PipelineParams(parser)
    args=parser.parse_args([])
    for k,v in read(RUN/'config.json').items():setattr(args,k,v)
    args.source_path=str(RUN);args.model_path=str(RUN/'model');args.resolution=1;args.data_device='cpu'
    args.iterations=iterations;args.imp_metric='indoor';args.num_depth_factor=1.;args.sampling_factor=.6
    return args,lp.extract(args),op.extract(args),pp.extract(args)


def verify_source():
    for p,h in read(RUN/'source-manifest.json').items():assert sha(RUN/'source'/p)==h,p


def fullk_depth(camera,gaussians,pipe,background,**kwargs):
    """Keep MSV2's recovered camera-Z depth on the calibrated pixel-centre ray.

    Its CUDA backprojection inverts full_proj per pixel in float32 without
    homogeneous division. Reconstruct with full K and w2c instead; retain its
    depth and importance distribution on valid hits. No-hit zero points have
    zero sampling probability rather than creating points at the world origin.
    """
    import torch
    from gaussian_renderer import render_depth
    result=render_depth(camera,gaussians,pipe,background,**kwargs)
    pts=result['out_pts'].permute(1,2,0)
    w=torch.tensor(camera.row['w2c'],dtype=torch.float64,device='cuda')
    cp=pts.double()@w[:3,:3].T+w[:3,3]
    good=torch.isfinite(cp).all(dim=2)&(cp[:,:,2]>.01)&(pts.square().sum(dim=2)>0)
    yy,xx=torch.meshgrid(torch.arange(camera.height,device='cuda',dtype=torch.float64)+.5,
                         torch.arange(camera.width,device='cuda',dtype=torch.float64)+.5,indexing='ij')
    k=torch.tensor(camera.K,dtype=torch.float64,device='cuda')
    rays=torch.stack((xx,yy,torch.ones_like(xx)),dim=-1)@torch.linalg.inv(k).T
    corrected=((rays*cp[:,:,2:3]-w[:3,3])@w[:3,:3]).float()
    result['out_pts']=torch.where(good[:,:,None],corrected,torch.zeros_like(corrected)).permute(2,0,1).contiguous()
    result['accum_alpha']=torch.where(good,result['accum_alpha'].squeeze(),torch.ones_like(good,dtype=torch.float32)).reshape_as(result['accum_alpha'])
    return result


def preflight():
    torch=runtime();verify_source()
    from scene import GaussianModel
    from gaussian_renderer import render_imp,render_depth
    from utils.loss_utils import l1_loss
    from fused_ssim import fused_ssim
    args,data,opt,pipe=options();g=GaussianModel(0);scene=Scene(data,g);g.training_setup(opt);g.init_culling(59)
    xyz=np.load(RUN/'initial-points.npz')['xyz'][::400]
    max_projection=0.
    for c in scene.cameras:
        pts=torch.tensor(np.c_[xyz,np.ones(len(xyz))],dtype=torch.float32,device='cuda')
        h=pts@c.full_proj_transform;uv=(h[:,:2]/h[:,3:4]+1)*torch.tensor([c.width,c.height],device='cuda')/2-.5
        w=np.array(c.row['w2c']);cp=xyz@w[:3,:3].T+w[:3,3];q=cp@c.K.T
        expected=q[:,:2]/q[:,2:3]-.5
        error=float(np.abs(uv.cpu().numpy()-expected).max());max_projection=max(max_projection,error)
    assert max_projection<.01,max_projection
    tests=[];cams={c.uid:c for c in scene.cameras};bg=torch.zeros(3,device='cuda')
    for i in [0,29,58]:
        c=cams[i];pkg=render_imp(c,g,pipe,bg,culling=g._culling[:,c.uid]);im=pkg['render'];gt=c.original_image
        loss=.8*l1_loss(im,gt)+.2*(1-fused_ssim(im[None],gt[None]));loss.backward()
        assert tuple(im.shape)==(3,2880,1428) and torch.isfinite(loss)
        assert g._xyz.grad is not None and torch.isfinite(g._xyz.grad).all()
        save_image(RUN/'preflight'/f'{c.image_name}.png',im)
        tests.append(dict(frame=c.image_name,loss=float(loss),render_wh=[1428,2880]))
        g.optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        c=cams[29];d=fullk_depth(c,g,pipe,bg,culling=g._culling[:,c.uid])
        pts=d['out_pts'].permute(1,2,0);alpha=d['accum_alpha'].squeeze()
        flat=pts.reshape(-1,3);valid=torch.isfinite(flat).all(dim=1)&(alpha.reshape(-1)>.5)&(flat.square().sum(dim=1)>0)
        ids=torch.nonzero(valid).flatten()[::100]
        w=torch.tensor(c.row['w2c'],dtype=torch.float32,device='cuda');k=torch.tensor(c.K,dtype=torch.float32,device='cuda')
        cp=flat[ids]@w[:3,:3].T+w[:3,3];q=cp@k.T;uv=q[:,:2]/q[:,2:3]-.5
        expected=torch.stack((ids%c.width,ids//c.width),dim=1)
        error=float((uv-expected).abs().max());assert error<.03,error
    save(RUN/'preflight.json',dict(passed=True,created=now(),frames=59,full_projection_max_error_px=max_projection,
        depth_backprojection_max_error_px=error,gpu_render_backward=tests,optimizer_updates=0,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,quality_accepted=False))
    print('MSV2_PREFLIGHT_PASSED',max_projection,error,flush=True)


def train():
    torch=runtime();verify_source();assert read(RUN/'preflight.json')['passed']
    assert not (RUN/'model').exists(),'Never overwrite or silently resume a previous run'
    free,total=torch.cuda.mem_get_info();assert free>11*1024**3,(free,total)
    torch.cuda.set_per_process_memory_fraction(12*1024**3/total)
    random.seed(6060);np.random.seed(6060);torch.manual_seed(6060)
    spec=importlib.util.spec_from_file_location('upstream_msv2_train',RUN/'source/msv2/train.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.Scene=Scene;module.network_gui.try_connect=lambda:None
    module.render_depth=fullk_depth
    start=time.monotonic()
    def report(tb,iteration,Ll1,loss,*unused):
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
        if iteration%100==0 or iteration==1:
            g=Scene.latest.gaussians
            value=dict(updated=now(),state='training',iteration_label=iteration,optimizer_updates=iteration-1,
                points=len(g.get_xyz),loss=float(loss),l1=float(Ll1),seconds=time.monotonic()-start,
                gpu_reserved_gib=torch.cuda.memory_reserved()/1024**3,quality_accepted=False)
            save(RUN/'progress.json',value);print(json.dumps(value),flush=True)
    module.training_report=report
    args,data,opt,pipe=options()
    save(RUN/'execution.json',dict(state='running',started=now(),pid=os.getpid(),quality_accepted=False))
    try:
        module.training(data,opt,pipe,[],[1000,3000,8000,18000],[],None,-1,args)
        g=Scene.latest.gaussians
        torch.cuda.synchronize()
        save(RUN/'execution.json',dict(state='training_completed',finished=now(),pid=os.getpid(),iterations=18000,
             optimizer_updates=17999,points=len(g.get_xyz),seconds=time.monotonic()-start,
             peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,quality_accepted=False))
    except BaseException:
        save(RUN/'execution.json',dict(state='failed',finished=now(),pid=os.getpid(),error=traceback.format_exc(),quality_accepted=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','preflight','train']);a=p.parse_args()
    globals()[a.action]()
