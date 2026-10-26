"""Bounded local research adaptation of PGSR, using only real static views.

Original PGSR RGB, planar-depth renderer, min-scale, normal, NCC and geometric
losses are retained. Adaptations: all-frame foreground input, interior masks,
point/memory bounds, portable standard PLY, and exact input provenance.
No prior subject Gaussian or previous optimizer is loaded.
"""
from pathlib import Path
import sys, os, json, time, argparse, math, hashlib
from types import SimpleNamespace
import numpy as np
from PIL import Image, ImageDraw
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
B=LAB/'outputs/mini-splatting2-mip-depth-001/shuolee'
PRIVATE=B/'current/novel-view-face/head-structure/private-runtime'
sys.path[1:1]=['/opt/gs-slam-lab/shuolee-pgsr/target',str(PRIVATE/'python-deps'),
    '/opt/gs-slam-lab/shuolee-ga-raster/target','/opt/gs-slam-lab/mini-splatting2-native-001/deps']
import torch
import torch.nn.functional as F
from camera_contract import require_registered_cameras
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from gaussian_renderer import render
from utils.graphics_utils import BasicPointCloud,patch_offsets,patch_warp
from utils.loss_utils import ssim,ssim2,lncc,get_img_grad_weight
from arguments import OptimizationParams
BG=torch.ones(3,device='cuda');PIPE=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');t.replace(p)
def pil(t):return Image.fromarray((t.detach().clamp(0,1).permute(1,2,0).cpu().numpy()*255+.5).astype(np.uint8))

def masked_photometric(rgb,target,alpha):
    """Color supervision only on confident foreground; alpha loss stays separate.

    L1 retains every confident foreground pixel. SSIM uses only complete 11x11
    foreground windows, so background cannot leak into local statistics. The
    original segmentation is not eroded or rewritten for alpha supervision.
    """
    assert rgb.ndim==3 and target.shape==rgb.shape and alpha.shape==rgb[:1].shape
    valid=alpha[0]>.99
    outside=(~valid).float()[None,None]
    support=(F.max_pool2d(F.pad(outside,(5,5,5,5),value=1),11,1)[0,0]==0)
    assert bool(valid.any()) and bool(support.any()),'Insufficient foreground support'
    l1=(rgb-target).abs()[:,valid].mean()
    # Explicitly disconnect invalid pixels, including numerical local-statistics paths.
    safe_rgb=torch.where(valid[None],rgb,torch.zeros_like(rgb))
    safe_gt=torch.where(valid[None],target,torch.zeros_like(target))
    structural=1-ssim2(safe_rgb,safe_gt)[support].mean()
    return .8*l1+.2*structural

class RawCamera(Camera):
    def get_image(self):
        rgb=self.rgb.cuda().float()/255;alpha=self.alpha.cuda().float()/255
        rgb=(self.associated_rgb.cuda().float() if hasattr(self,'associated_rgb') else rgb*alpha)+1-alpha
        gray=(rgb*torch.tensor([.299,.587,.114],device='cuda')[:,None,None]).sum(0,keepdim=True)
        return rgb,gray
    def foreground(self):return self.alpha.cuda().float()/255
    def interior(self):return (1-F.max_pool2d(1-self.foreground()[None],9,1,4))[0,0]>.99
    def associated(self):return self.associated_rgb.cuda().float()
    def contour(self):
        a=self.foreground()[None]
        return ((F.max_pool2d(a,11,1,5)-(1-F.max_pool2d(1-a,11,1,5)))[0,0]>.01)

def cameras(rows,height,alpha_composite=False):
    cams=[];records=[]
    for i,r in enumerate(rows):
        assert sha(r['image'])==r['source_sha256'] and sha(r['mask'])==r['mask_sha256']
        im=Image.open(r['image']).convert('RGB');mask=Image.open(r['mask']).convert('L')
        assert im.size==mask.size==tuple(r['wh'])
        W=round(im.width*height/im.height);H=height
        k=np.diag([W/im.width,H/im.height,1.])@np.array(r['K']);e=np.array(r['w2c'])
        assert abs(k[0,2]-W/2)<1e-5 and abs(k[1,2]-H/2)<1e-5,'PGSR depth expects centered full image'
        cam=RawCamera(i,e[:3,:3].T,e[:3,3],2*np.arctan(W/(2*k[0,0])),2*np.arctan(H/(2*k[1,1])),W,H,r['image'],r['frame'],i,preload_img=False)
        a=np.asarray(mask.resize((W,H),Image.Resampling.BILINEAR)).copy()
        rgb=np.asarray(im.resize((W,H),Image.Resampling.LANCZOS)).copy()
        cam.rgb=torch.from_numpy(rgb).permute(2,0,1);cam.alpha=torch.from_numpy(a)[None]
        if alpha_composite:
            raw=np.asarray(im,dtype=np.float32)/255;original_alpha=np.asarray(mask,dtype=np.float32)/255
            associated=np.stack([np.asarray(Image.fromarray(raw[:,:,j]*original_alpha,mode='F').resize((W,H),Image.Resampling.BILINEAR)) for j in range(3)])
            cam.associated_rgb=torch.from_numpy(associated.copy()).half()
        yy,xx=np.where(a>127);pad=round(.1*(yy.max()-yy.min()+1))
        cam.roi=(max(0,yy.min()-pad),min(H,yy.max()+pad+1),max(0,xx.min()-pad),min(W,xx.max()+pad+1))
        # CUDA RGB NDC coordinates correspond to K projection minus half a pixel.
        pts=np.array([[.1,.2,.3],[-.3,.3,.5],[.5,.5,.4]],dtype=np.float32)
        q=(pts@e[:3,:3].T+e[:3,3])@k.T;uv=q[:,:2]/q[:,2:]-.5
        clip=np.c_[pts,np.ones(len(pts))]@cam.full_proj_transform.cpu().numpy()
        actual=((clip[:,:2]/clip[:,3:]+1)*[W,H]-1)/2
        err=float(abs(actual-uv).max());assert err<.01,(i,err)
        records.append(dict(frame=i,width=W,height=H,K=k.tolist(),w2c=e.tolist(),projection_error_px=err))
        cams.append(cam)
    # Use 4-24 degree viewing baselines, not nearly identical adjacent frames.
    source=np.stack([np.linalg.inv(np.array(r['w2c']))[:3,2] for r in rows])
    for i,cam in enumerate(cams):
        angles=np.rad2deg(np.arccos(np.clip(source@source[i],-1,1)))
        candidates=np.where((angles>=4)&(angles<=24))[0]
        assert len(candidates)>=4,(i,len(candidates))
        ranked=sorted(candidates,key=lambda j:abs(angles[j]-12))[:16]
        cam.nearest_id=[int(v) for v in ranked]
        records[i]['neighbor_ids']=cam.nearest_id
    return cams,records

def standard_render(cam,m):
    from diff_gaussian_rasterization import GaussianRasterizer,GaussianRasterizationSettings
    s=GaussianRasterizationSettings(image_height=cam.image_height,image_width=cam.image_width,
        tanfovx=math.tan(cam.FoVx*.5),tanfovy=math.tan(cam.FoVy*.5),bg=BG,scale_modifier=1.,
        viewmatrix=cam.world_view_transform,projmatrix=cam.full_proj_transform,sh_degree=m.active_sh_degree,
        campos=cam.camera_center,prefiltered=False,debug=False)
    return GaussianRasterizer(raster_settings=s)(means3D=m.get_xyz,means2D=torch.zeros_like(m.get_xyz),
        shs=m.get_features,colors_precomp=None,opacities=m.get_opacity,scales=m.get_scaling,
        rotations=m.get_rotation,cov3D_precomp=None)[0]

@torch.no_grad()
def parity(m,cams):
    checks=[]
    for i in [0,33,69,99,135,174]:
        a=render(cams[i],m,PIPE,BG,return_plane=False,return_depth_normal=False)['render'];b=standard_render(cams[i],m)
        d=(a-b).abs();rec=dict(frame=i,mean_abs=float(d.mean()),max_abs=float(d.max()))
        checks.append(rec)
        assert rec['mean_abs']<2e-6 and rec['max_abs']<.005,rec
    return checks

def multiview(m,cam,near,pkg,opt):
    nr=render(near,m,PIPE,BG,return_plane=True,return_depth_normal=False)
    H,W=cam.image_height,cam.image_width
    iy,ix=torch.meshgrid(torch.arange(H,device='cuda'),torch.arange(W,device='cuda'),indexing='ij')
    pixels=torch.stack([ix,iy],-1).float()
    pts=m.get_points_from_depth(cam,pkg['plane_depth'])
    cp=pts@near.world_view_transform[:3,:3]+near.world_view_transform[3,:3]
    z,valid=m.get_points_depth_in_depth_map(near,nr['plane_depth'],cp)
    near_uv=torch.stack([cp[:,0]*near.Fx/cp[:,2].clamp_min(1e-5)+near.Cx,cp[:,1]*near.Fy/cp[:,2].clamp_min(1e-5)+near.Cy],-1)
    ng=near_uv/torch.tensor([W-1,H-1],device='cuda')*2-1
    nmask=F.grid_sample(near.interior()[None,None].float(),ng[None,:,None],align_corners=True)[0,0,:,0]>.99
    cp=cp/cp[:,2:3].clamp_min(1e-5)*z.squeeze()[:,None]
    wr=(cp-torch.as_tensor(near.T,device='cuda',dtype=torch.float32))@torch.as_tensor(near.R.T,device='cuda',dtype=torch.float32)
    cr=wr@cam.world_view_transform[:3,:3]+cam.world_view_transform[3,:3]
    uv=torch.stack([cr[:,0]*cam.Fx/cr[:,2].clamp_min(1e-5)+cam.Cx,cr[:,1]*cam.Fy/cr[:,2].clamp_min(1e-5)+cam.Cy],-1)
    noise=torch.norm(uv-pixels.reshape(-1,2),dim=-1)
    valid=valid&nmask&cam.interior().flatten()&(noise<opt.multi_view_pixel_noise_th)&(pkg['rendered_alpha'].flatten()>.9)&(cr[:,2]>.1)
    weights=torch.exp(-noise).detach()
    ids=torch.where(valid)[0];zero=m._xyz.sum()*0
    if not len(ids):return zero,zero,0
    geo=opt.multi_view_geo_weight*(weights[ids]*noise[ids]).mean()
    if len(ids)>opt.multi_view_sample_num:ids=ids[torch.randperm(len(ids),device='cuda')[:opt.multi_view_sample_num]]
    with torch.no_grad():
        pix=pixels.reshape(-1,2)[ids];off=patch_offsets(opt.multi_view_patch_size,pix.device)
        patch=pix[:,None]+off.float();g=patch/torch.tensor([W-1,H-1],device='cuda')*2-1
        _,gray=cam.get_image();ref=F.grid_sample(gray[None],g.reshape(1,-1,1,2),align_corners=True).reshape(len(ids),-1)
        r=near.world_view_transform[:3,:3].T@cam.world_view_transform[:3,:3]
        t=-r@cam.world_view_transform[3,:3]+near.world_view_transform[3,:3]
    n=pkg['rendered_normal'].permute(1,2,0).reshape(-1,3)[ids]
    d=pkg['rendered_distance'].flatten()[ids]
    hom=r[None]-t[None,:,None]*n[:,None,:]/d[:,None,None].clamp_min(1e-5)
    hom=near.get_k()[None]@hom@cam.get_inv_k()
    g=patch_warp(hom,patch);g=g/torch.tensor([W-1,H-1],device='cuda')*2-1
    _,gray=near.get_image();other=F.grid_sample(gray[None],g.reshape(1,-1,1,2),align_corners=True).reshape(len(ids),-1)
    # Entire warped patches must stay inside the observed person.
    fullmask=F.grid_sample(near.foreground()[None],g.reshape(1,-1,1,2),align_corners=True).reshape(len(ids),-1).min(-1).values>.99
    ncc,ok=lncc(ref,other);ok=ok.flatten()&fullmask&torch.isfinite(ncc.flatten())
    photo=opt.multi_view_ncc_weight*(ncc.flatten()[ok]*weights[ids][ok]).mean() if ok.any() else zero
    return geo,photo,int(valid.sum())

@torch.no_grad()
def review(m,cams,run,step):
    tiles=[];metrics=[]
    for i in [0,33,69,99,135,174]:
        gt,_=cams[i].get_image();rgb=render(cams[i],m,PIPE,BG,return_plane=False,return_depth_normal=False)['render']
        y0,y1,x0,x1=cams[i].roi;metrics.append(dict(frame=i,training_roi_mae=float((rgb-gt)[:,y0:y1,x0:x1].abs().mean())))
        ims=[pil(x[:,y0:y1,x0:x1]) for x in [gt,rgb]]
        for im in ims:im.thumbnail((300,440))
        p=Image.new('RGB',(600,475),'white');p.paste(ims[0],((300-ims[0].width)//2,30));p.paste(ims[1],(300+(300-ims[1].width)//2,30))
        ImageDraw.Draw(p).text((8,6),f'raw {i+1:03d} | input / PGSR static | {step}',fill='black');tiles.append(p)
    sheet=Image.new('RGB',(1800,950),'white')
    for i,p in enumerate(tiles):sheet.paste(p,((i%3)*600,(i//3)*475))
    sheet.save(run/'review-current.jpg',quality=94);save(run/'review-metrics.json',dict(step=step,metrics=metrics,training_views=True,quality_accepted=False))

def main(root,steps,height,only_preflight=False,scale_weight=None,multiview_factor=1.,alpha_composite=False,mask_foreground=False,observed_path=None,observed_weight=.002,surface_weight=0.,pixel_centers_consistent=False,ncc_factor=1.,trace_surface=False,whole_body_trace=False,stop_after=None,shared_face_path=None,shared_face_weight=.002,shared_face_mode='plane',native_person_path=None):
    global render
    assert not (alpha_composite and mask_foreground),'Do not combine independent photometric experiments'
    run=root/'gaussian-run';run.mkdir(exist_ok=True)
    assert not (run/'current.pt').exists(),'Fresh only; never resume a prior subject'
    torch.manual_seed(6060);np.random.seed(6060)
    document=read(root/'data/training-inputs.json');camera_contract=require_registered_cameras(document)
    rows=document['records'];assert len(rows)==175
    points=np.load(root/'clothing-init/points.npz')['xyz'];assert len(points)==200000
    surface_enabled=surface_weight>0
    camera_factory=cameras
    assert ncc_factor in [0.,1.], 'Only the preregistered NCC on/off comparison is supported'
    if pixel_centers_consistent:
        assert not surface_enabled, 'Surface-moment renderer has not been audited with corrected centers'
        from shuolee_pixel_center import render as consistent_render, cameras as consistent_cameras
        render=consistent_render;camera_factory=consistent_cameras
    if surface_enabled:
        from functools import partial
        sys.path.insert(1,'/opt/gs-slam-lab/shuolee-surface-moments/target')
        from gaussian_renderer.surface import render as surface_render
        from surface_concentration import concentration_loss
        precheck=read(B/'current/static-depth-mixture-audit-078/surface-cuda-preflight.json')
        assert precheck['numerical_preflight_passed'] and precheck['model_unchanged'] and precheck['optimizer_updates']==0
        surface_origin=torch.tensor(points.mean(0),device='cuda')
        _,_,surface_axes=np.linalg.svd((points-points.mean(0))[::20],full_matrices=False)
        surface_vertical=points@surface_axes[0]
        surface_span=float(np.quantile(surface_vertical,.995)-np.quantile(surface_vertical,.005))
        assert surface_span>0 and surface_weight==.01,'Only the preregistered single candidate is allowed'
        render=partial(surface_render,moment_origin=surface_origin)
    scale=float(np.load(root/'clothing-init/motion.npz')['world_scale'])
    cams,records=camera_factory(rows,height,alpha_composite);save(run/'training-cameras.json',dict(records=records))
    pair_supervision=None
    if observed_path is not None:
        from observed_correspondence import ObservedPairs
        assert observed_weight>0
        pair_supervision=ObservedPairs(observed_path,cams,rows)
    shared_face=None
    if shared_face_path is not None:
        assert pixel_centers_consistent and mask_foreground and not surface_enabled and not alpha_composite
        assert shared_face_weight==.002 and not whole_body_trace and not trace_surface
        from functools import partial
        assert shared_face_mode in ['plane','finite_support']
        if shared_face_mode=='finite_support':
            from shared_face_support import SharedFaceSupport as SharedFaceWorld, render as shared_render
            admission=read(B/'current/shared-finite-support-preflight-102/result.json')
        else:
            from shared_face_world import SharedFaceWorld, render as shared_render
            admission=read(B/'current/face-shared-world-points-097/consumer-preflight.json')
        assert admission['numerical_preflight_passed'] and admission['optimizer_updates']==0
        for path,digest in admission['consumer_sources'].items():assert sha(path)==digest, path
        shared_face=SharedFaceWorld(shared_face_path,cams,rows)
        render_options=dict(moment_origin=shared_face.xyz.mean(0))
        if shared_face_mode=='finite_support':render_options['support_sigma']=shared_face.support_sigma
        render=partial(shared_render,**render_options)
    model_class=GaussianModel
    if whole_body_trace:
        from scene.gaussian_model_095 import GaussianModel as TracedGaussianModel
        model_class=TracedGaussianModel
        assert steps==18000 and stop_after==1800, '095 diagnostic must preserve18000step schedule and stop1800'
        assert read(root/'trace-noninterference.json')['passed'], 'Trace must pass noninterference before training'
    else:
        assert stop_after is None, 'Early diagnostic stop requires its validated recorder'
    m=model_class(3);m.create_from_pcd(BasicPointCloud(points,np.full_like(points,.5),np.zeros_like(points)),scale)
    with torch.no_grad():m._scaling.clamp_(max=np.log(.012*scale));m._features_dc.zero_();m._features_rest.zero_()
    native_person=None;native_initialization=None
    if native_person_path is not None:
        assert pixel_centers_consistent and mask_foreground and shared_face is None and not surface_enabled and not alpha_composite
        assert not whole_body_trace and not trace_surface and steps==18000 and height==1536
        from native_person_alignment_113 import initialize_surface,NativePersonAlignment
        admission=read(root/'native-volume-preflight/result.json');assert admission['passed'] and admission['optimizer_updates']==0
        protocol=read(root/'native-volume-preflight/protocol.json')
        for path,digest in protocol['source_sha256'].items():assert sha(path)==digest,path
        assert sha(native_person_path)==admission['target_sha256']
        assert sha(root/'surface-init/normals.npz')==admission['normals_sha256']
        native_initialization=initialize_surface(m,root/'surface-init/normals.npz',sha(root/'clothing-init/points.npz'))
        native_person=NativePersonAlignment(native_person_path,cams,rows,geometry='conditional_volume')
    parser=argparse.ArgumentParser();opt=OptimizationParams(parser).extract(parser.parse_args([]))
    opt.iterations=steps;opt.position_lr_max_steps=steps
    opt.max_all_points=500000;opt.max_abs_split_points=25000;opt.multi_view_sample_num=8192
    opt.exposure_compensation=False;opt.use_virtul_cam=False;opt.use_multi_view_trim=True
    if scale_weight is not None:opt.scale_loss_weight=float(scale_weight)
    assert opt.scale_loss_weight>=0 and 0<=multiview_factor<=1
    opt.multi_view_geo_weight*=multiview_factor;opt.multi_view_ncc_weight*=multiview_factor
    opt.multi_view_ncc_weight*=ncc_factor
    m.training_setup(opt);assert len(m.optimizer.state)==0
    config=dict(steps=steps,initial_points=200000,frames=175,max_height=height,from_scratch=True,
        old_gaussian_loaded=False,old_subject_fit_loaded=False,body_used_only_for_fresh_initialization_box=True,
        old_optimizer_loaded=False,representation='shared_static_world',body_deformation_used=False,
        camera_source=camera_contract,
        original_repository='https://github.com/zju3dv/PGSR',commit='de24f1a38b350387e8d8fe381b2cd70c1ae946e7',
        adapter_sha256=sha(__file__),points_sha256=sha(root/'clothing-init/points.npz'),
        input_sha256=sha(root/'data/training-inputs.json'),options=vars(opt),
        controlled_ablation=dict(scale_loss_weight=opt.scale_loss_weight,multiview_factor=multiview_factor,
            reference_round='064',single_view_normal_unchanged=True,
            disabled_losses_still_computed_to_preserve_random_sampling=True),
        alpha_composite=dict(enabled=alpha_composite,paired_black_white_supervision=alpha_composite,
            associated_rgb_resized_before_compositing=alpha_composite,roi_and_contour_alpha=alpha_composite,
            original_masks_unchanged=True,no_color_threshold_pruning=True),
        foreground_photometric=dict(enabled=mask_foreground,valid='original resized mask > 0.99',
            l1='mean over valid foreground RGB components',ssim='mean over complete 11x11 foreground windows',
            outside_rgb_gradient_zero=True if mask_foreground else False,
            targets_and_resampling_unchanged=True,alpha_loss_full_image_weight=.1,
            geometry_losses_unchanged=True,reference_round='064'),
        differences=['foreground alpha loss0.1','RGB loss on foreground bounding crop of full-camera render',
            'full centered images with source intrinsics; no image crop camera','interior masks for real-view geometric/NCC losses',
            '8192 NCC sample cap;500000 point budget','same new silhouette init as062, not sparse points only',
            'complete18000 optimizer steps; official full iteration budget is30000'],quality_accepted=False)
    config['observed_correspondence']=dict(enabled=pair_supervision is not None,weight=observed_weight if pair_supervision else 0.,
        start=1000,ramp_until=3000,robust_delta_training_px=2,max_head_pairs=32,max_body_pairs=96,
        source=str(observed_path),sha256=sha(observed_path) if pair_supervision else None,
        camera_fixed=True,held_diagnostic_features_excluded=True if pair_supervision else None,
        no_residual_admission=True,reference_round='075',not_official_PGSR=True)
    config['pixel_center_consistency']=dict(enabled=pixel_centers_consistent,
        correction_source_sha256=sha(Path(__file__).with_name('shuolee_pixel_center.py')) if pixel_centers_consistent else None,
        observed_source_sha256=sha(Path(__file__).with_name('observed_correspondence.py')),
        RGB_rasterization_unchanged=True,diagnostic='091 native CUDA and092 observed-interface checks')
    config['ncc_ablation']=dict(factor=ncc_factor,geometric_weight_unchanged=True,
        disabled_NCC_still_computed=True,only_between_arm_variable='NCC loss weight',
        reference='093 corrected-coordinate matched fresh control' if pixel_centers_consistent else 'legacy')
    config['trace_surface']=dict(enabled=trace_surface,
        checkpoints=[0,1800,2999,6999,7001,7500,9001,18000],
        fixed_queries='084 raw100 ghost query; no point ancestry inferred')
    config['whole_body_trace']=dict(enabled=whole_body_trace,nominal_schedule_steps=steps,stop_after=stop_after,
        recording_does_not_change_objective=True,sampled_roots_per_height_band=256,
        model_source_sha256=sha(Path(__file__).parent/'scene/gaussian_model_095.py') if whole_body_trace else None,
        recorder_source_sha256=sha(Path(__file__).with_name('trace_whole_body_095.py')) if whole_body_trace else None)
    config['surface_concentration']=dict(enabled=surface_enabled,weight=surface_weight,start=3000,
        formulation='weighted squared point-to-plane residual at detached blended hit, pseudo-Huber',
        tolerance_fraction_initial_principal_extent=.01,reference_round='077',not_official_PGSR_or_2DGS=True,
        initial_extent=surface_span if surface_enabled else None,
        origin=surface_origin.cpu().tolist() if surface_enabled else None,
        mask='9x9 foreground interior, alpha>0.8, plane ray denominator>0.1, positive depth',
        source_sha256=sha(Path(__file__).with_name('surface_concentration.py')) if surface_enabled else None)
    config['shared_face_world']=dict(enabled=shared_face is not None,weight=shared_face_weight if shared_face else 0.,start=1,
        source=str(shared_face_path),sha256=sha(shared_face_path) if shared_face else None,
        consumer_sha256=sha(Path(__file__).with_name('shared_face_support.py' if shared_face_mode=='finite_support' else 'shared_face_world.py')) if shared_face else None,
        global_points=len(shared_face.xyz) if shared_face else 0,all_visible_observations=True,
        one_world_point_per_feature=True,per_camera_targets=False,cameras_fixed=True,
        mode=shared_face_mode,
        support_sigma_world=shared_face.support_sigma if shared_face and shared_face_mode=='finite_support' else None,
        formulation=('Expected full-rank finite3Dsupport Mahalanobis to fixed sharedXYZ, covariance metric detached; uncertainty-scaled pseudo-Huber.' if shared_face_mode=='finite_support' else 'Expected squared contributor-plane distance to fixed shared XYZ, normalized by image-derived uncertainty; pseudo-Huber.'),
        only_new_training_term=True,not_ground_truth=True,not_official_PGSR=True)
    config['native_person_alignment']=dict(enabled=native_person is not None,weight=.002,start=1,
        geometry='per-contributor finite3DGaussian ray mode, exact native screen weights, fixed assignment for geometry only',
        initialization=native_initialization,queries_per_view_step=3,cameras_fixed=True,target_not_ground_truth=True,
        source=str(native_person_path) if native_person else None,source_sha256=sha(native_person_path) if native_person else None,
        consumer_sha256=sha(Path(__file__).with_name('native_person_alignment_113.py')) if native_person else None,
        admission_sha256=sha(root/'native-volume-preflight/result.json') if native_person else None,
        experiment='113 combined oriented initialization and native alignment; not single-factor attribution')
    save(run/'config.json',config)
    if shared_face and shared_face_mode=='finite_support':
        support_records=[]
        with torch.no_grad():
            for camera_index,view in enumerate(cams):
                initial_pkg=render(view,m,PIPE,BG,return_plane=True,return_depth_normal=False)
                initial_value,initial_detail=shared_face.loss(initial_pkg,camera_index)
                support_records.append(dict(frame=camera_index+1,observations=initial_detail['observation_count'],available=initial_detail['available_count'],finite=bool(torch.isfinite(initial_value))))
                del initial_pkg,initial_value,initial_detail
        save(run/'all-camera-shared-support.json',dict(records=support_records,optimizer_updates=0))
        assert all(q['finite'] and q['available']==q['observations'] for q in support_records),'Missing initial support; do not start training'
    check=parity(m,cams)
    p=render(cams[69],m,PIPE,BG,return_plane=True,return_depth_normal=True)
    geo,ncc,n=multiview(m,cams[69],cams[cams[69].nearest_id[0]],p,opt)
    gt,_=cams[69].get_image();loss=(p['render']-gt).abs().mean()+.015*(p['depth_normal']-p['rendered_normal']).abs().mean()+geo+ncc
    if mask_foreground:
        y0,y1,x0,x1=cams[69].roi
        loss=masked_photometric(p['render'][:,y0:y1,x0:x1],gt[:,y0:y1,x0:x1],cams[69].foreground()[:,y0:y1,x0:x1])+.1*(p['rendered_alpha']-cams[69].foreground()).abs().mean()+.015*(p['depth_normal']-p['rendered_normal']).abs().mean()+geo+ncc
    composite_error=None
    if alpha_composite:
        direct=render(cams[69],m,PIPE,torch.zeros_like(BG),return_plane=False,return_depth_normal=False)['render']
        associated=p['render']-(1-p['rendered_alpha'])
        composite_error=float((direct-associated).abs().max());assert composite_error<3e-6,composite_error
        loss=loss+(associated-cams[69].associated()).abs().mean()+(p['rendered_alpha']-cams[69].foreground()).abs().mean()
    correspondence_check=None
    if pair_supervision:
        cor,detail=pair_supervision.loss(p,69,0);assert torch.isfinite(cor) and int(detail['valid'].sum())>5
        correspondence_check=dict(raw=70,pairs=len(detail['valid']),valid=int(detail['valid'].sum()),loss=float(cor.detach()))
        loss=loss+observed_weight*cor
    surface_check=None
    if surface_enabled:
        raw_surface,detail=concentration_loss(p,cams[69],p['moment_origin_camera'],surface_span)
        assert torch.isfinite(raw_surface) and int(detail['valid'].sum())>100
        surface_check=dict(raw=70,valid=int(detail['valid'].sum()),loss=float(raw_surface.detach()))
        loss=loss+surface_weight*raw_surface
    shared_check=None
    if shared_face:
        raw_shared,detail=shared_face.loss(p,69)
        assert torch.isfinite(raw_shared) and detail['available_count']==detail['observation_count']
        shared_check=dict(raw=70,loss=float(raw_shared.detach()),observations=detail['observation_count'],valid=detail['available_count'])
        loss=loss+shared_face_weight*raw_shared
    native_check=None
    if native_person:
        raw_native,detail=native_person.loss(m,69,PIPE)
        assert torch.isfinite(raw_native) and detail['available_count']>0
        native_check=dict(raw=70,loss=float(raw_native.detach()),observations=detail['observation_count'],valid=detail['available_count'])
        loss=loss+.002*raw_native
    assert torch.isfinite(loss);loss.backward()
    for name in ['_xyz','_scaling','_rotation','_opacity','_features_dc']:
        g=getattr(m,name).grad;assert g is not None and torch.isfinite(g).all(),name
    m.optimizer.zero_grad(set_to_none=True)
    save(run/'preflight.json',dict(native_vs_standard=check,finite_all_losses_and_backward=True,
        multi_view_valid_pixels=n,new_optimizer_updates=0,initial_points=len(points),world_scale=scale,observed_correspondence=correspondence_check,surface_concentration=surface_check,shared_face_world=shared_check,native_person_alignment=native_check))
    if alpha_composite:save(run/'alpha-preflight.json',dict(black_via_alpha_vs_direct_max_abs=composite_error,paired_backgrounds=True,finite_backward=True))
    del p,loss,gt,geo,ncc
    review(m,cams,run,0)
    if trace_surface:
        from trace_static_surface_093 import record as record_surface
        record_surface(m,cams,run,0,render,PIPE,BG)
    if only_preflight:return
    lineage=None
    if whole_body_trace:
        from trace_whole_body_095 import LineageTrace,MILESTONES
        lineage=LineageTrace(m,cams,run/'lineage')
        lineage.snapshot(m,0,render,PIPE,BG)
    save(run/'started.json',dict(started=time.time(),pid=os.getpid(),steps=steps))
    state=read(B/'work-status.json');state['local_training_running']=True
    state['latest_local_execution'].update(state='fresh_static_pgsr_training',running=True,steps=steps);save(B/'work-status.json',state)
    extent=float(np.linalg.norm(points.max(0)-points.min(0)));counts=np.zeros(175,int);stack=[];start=time.monotonic();schedule=[]
    native_window=[0,0];native_totals=np.zeros((175,4,2),dtype=np.int64);native_history=[]
    try:
        with (run/'loss.jsonl').open('x',encoding='utf8') as log:
            for step in range(1,steps+1):
                m.update_learning_rate(step)
                if step%1000==0:m.oneupSHdegree()
                if not stack:stack=np.random.permutation(175).tolist()
                i=stack.pop();counts[i]+=1;cam=cams[i];gt,_=cam.get_image();alpha=cam.foreground()
                p=render(cam,m,PIPE,BG,return_plane=True,return_depth_normal=step>opt.single_view_weight_from_iter)
                y0,y1,x0,x1=cam.roi;rgb=p['render'][:,y0:y1,x0:x1];target=gt[:,y0:y1,x0:x1]
                photo=.8*(rgb-target).abs().mean()+.2*(1-ssim(rgb,target))
                mask=(p['rendered_alpha']-alpha).abs().mean()
                if mask_foreground:
                    photo=masked_photometric(rgb,target,alpha[:,y0:y1,x0:x1])
                if alpha_composite:
                    pred_alpha=p['rendered_alpha'][:,y0:y1,x0:x1];ref_alpha=alpha[:,y0:y1,x0:x1]
                    black=rgb-(1-pred_alpha);fg=cam.associated()[:,y0:y1,x0:x1]
                    pairs=torch.stack([black,rgb]);targets=torch.stack([fg,fg+1-ref_alpha])
                    photo=.8*(pairs-targets).abs().mean()+.2*(1-ssim(pairs,targets))
                    ae=(p['rendered_alpha']-alpha).abs();ring=cam.contour()
                    mask=.5*ae[:,y0:y1,x0:x1].mean()+.5*ae[0][ring].mean()
                visible=p['visibility_filter'];flatten=opt.scale_loss_weight*m.get_scaling[visible].min(-1).values.mean()
                normal=photo*0;geo=photo*0;ncc=photo*0;valid=0;near_id=-1
                if step>opt.single_view_weight_from_iter:
                    iw=(1-get_img_grad_weight(gt)).clamp(0,1).nan_to_num().detach().square();inside=cam.interior()
                    err=(p['depth_normal']-p['rendered_normal']).abs().sum(0)
                    normal=opt.single_view_weight*(iw*err)[inside].mean()
                if step>opt.multi_view_weight_from_iter:
                    near_id=int(np.random.choice(cam.nearest_id));near=cams[near_id];geo,ncc,valid=multiview(m,cam,near,p,opt)
                schedule.append([i,near_id])
                correspondence=photo*0;correspondence_valid=0
                if pair_supervision is not None and step>1000:
                    raw_cor,detail=pair_supervision.loss(p,i,int(counts[i]-1));correspondence_valid=int(detail['valid'].sum())
                    correspondence=observed_weight*min(1.,(step-1000)/2000.)*raw_cor
                surface=photo*0;surface_valid=0
                if surface_enabled and step>3000:
                    raw_surface,detail=concentration_loss(p,cam,p['moment_origin_camera'],surface_span)
                    surface_valid=int(detail['valid'].sum());surface=surface_weight*raw_surface
                shared=photo*0;shared_valid=0;shared_observations=0
                if shared_face:
                    raw_shared,detail=shared_face.loss(p,i)
                    if shared_face_mode=='finite_support':assert detail['available_count']==detail['observation_count'],('Lost named-feature support',step,i,detail['available_count'],detail['observation_count'])
                    shared=shared_face_weight*raw_shared;shared_valid=detail['available_count'];shared_observations=detail['observation_count']
                native_term=photo*0;native_valid=0;native_observations=0
                if native_person:
                    raw_native,detail=native_person.loss(m,i,PIPE,visit=int(counts[i]-1))
                    native_valid=detail['available_count'];native_observations=detail['observation_count'];native_term=.002*raw_native
                    native_window[0]+=native_valid;native_window[1]+=native_observations
                    for query in detail['queries']:
                        native_totals[i,query['region'],0]+=int(query['valid']);native_totals[i,query['region'],1]+=1
                loss=photo+.1*mask+flatten+normal+geo+ncc+correspondence+surface+shared
                loss=loss+native_term
                assert torch.isfinite(loss),(step,float(loss));loss.backward()
                with torch.no_grad():
                    if lineage is not None:lineage.before_update(m,step,i)
                    m.optimizer.step()
                    if lineage is not None:lineage.after_update(m)
                    m.optimizer.zero_grad(set_to_none=True)
                    if step<opt.densify_until_iter:
                        seen=(p['out_observe']>0)&visible
                        m.max_radii2D[seen]=torch.maximum(m.max_radii2D[seen],p['radii'][seen])
                        m.add_densification_stats(p['viewspace_points'],p['viewspace_points_abs'],visible)
                        if step>opt.densify_from_iter and step%opt.densification_interval==0:
                            if lineage is not None and step in [600,1000]:lineage.snapshot(m,step,render,PIPE,BG,phase='before_densify')
                            m.densify_and_prune(opt.densify_grad_threshold,opt.densify_abs_grad_threshold,opt.opacity_cull_threshold,extent,20 if step>opt.opacity_reset_interval else None)
                        if step%1000==0:
                            observed=torch.zeros(len(m._xyz),device='cuda',dtype=torch.int32)
                            for view in cams:
                                observed+=(render(view,m,PIPE,BG,return_plane=False,return_depth_normal=False)['out_observe']>0).int()
                            m.prune_points(observed<2)
                        if step%opt.opacity_reset_interval==0 or step==opt.densify_from_iter:
                            if lineage is not None:lineage.snapshot(m,step,render,PIPE,BG,phase='before_opacity_reset')
                            m.reset_opacity()
                    assert 10000<len(m._xyz)<650000,len(m._xyz)
                if step==1 or step%100==0:
                    rec=dict(step=step,steps=steps,seconds=time.monotonic()-start,points=len(m._xyz),photo=float(photo.detach()),alpha=float(mask.detach()),
                        flatten=float(flatten.detach()),normal=float(normal.detach()),geo=float(geo.detach()),ncc=float(ncc.detach()),valid_multi_pixels=valid,
                        correspondence=float(correspondence.detach()),correspondence_valid=correspondence_valid,
                        surface=float(surface.detach()),surface_valid=surface_valid,
                        shared_face=float(shared.detach()),shared_face_valid=shared_valid,shared_face_observations=shared_observations,
                        native_person=float(native_term.detach()),native_person_valid=native_valid,native_person_observations=native_observations,
                        frames_seen=int((counts>0).sum()),running=step<steps and step!=stop_after,peak_gpu_gb=torch.cuda.max_memory_allocated()/1e9,quality_accepted=False)
                    save(run/'progress.json',rec);log.write(json.dumps(rec)+'\n');log.flush();print(json.dumps(rec),flush=True)
                if native_person and step%500==0:
                    fraction=native_window[0]/max(native_window[1],1);native_history.append(dict(step=step,valid=native_window[0],queried=native_window[1],valid_fraction=fraction))
                    save(run/'native-coverage.json',dict(windows=native_history,per_frame_region_valid_total=native_totals.tolist(),regions=['face_image_bbox','upper_body','lower_body','feet'],quality_accepted=False))
                    if fraction<.9:
                        review(m,cams,run,step);m.save_ply(str(run/'point_cloud.ply'))
                        torch.save(dict(capture=m.capture(),step=step,counts=counts,config=config),run/'current.pt')
                        save(run/'failure.json',dict(reason='native_alignment_lost_support',step=step,valid_fraction=fraction,quality_accepted=False))
                        raise RuntimeError('Native alignment support fell below its fixed90percent window gate')
                    native_window=[0,0]
                if trace_surface and step in [1800,2999,6999,7001,7500,9001,18000]:
                    record_surface(m,cams,run,step,render,PIPE,BG)
                if lineage is not None and step in MILESTONES:lineage.snapshot(m,step,render,PIPE,BG)
                if step in [1800,6000,9000] or step==steps or step==stop_after or (run/'STOP').exists():
                    review(m,cams,run,step);m.save_ply(str(run/'point_cloud.ply'))
                    torch.save(dict(capture=m.capture(),step=step,counts=counts,config=config),run/'current.pt')
                    if step==stop_after or (run/'STOP').exists():break
        if lineage is not None:lineage.finish(m)
        checks=parity(m,cams)
        np.save(run/'sampling-schedule.npy',np.asarray(schedule,dtype=np.int16))
        save(run/'result.json',dict(state='completed_pending_export_and_review' if step==steps else 'stopped_pending_review',
            steps=step,nominal_schedule_steps=steps,diagnostic_stop=stop_after is not None,points=len(m._xyz),seconds=time.monotonic()-start,from_scratch=True,frames_used=int((counts>0).sum()),
            per_frame_updates=counts.tolist(),sampling_schedule_sha256=sha(run/'sampling-schedule.npy'),
            native_vs_standard=checks,quality_accepted=False,source_ply_sha256=sha(run/'point_cloud.ply')))
    finally:
        state=read(B/'work-status.json');state['local_training_running']=False;state['latest_local_execution']['running']=False;save(B/'work-status.json',state)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--steps',type=int,default=18000)
    p.add_argument('--height',type=int,default=1536);p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--scale-weight',type=float,default=None);p.add_argument('--multiview-factor',type=float,default=1.)
    p.add_argument('--alpha-composite-loss',action='store_true')
    p.add_argument('--mask-foreground-only',action='store_true')
    p.add_argument('--observed-correspondence',type=Path,default=None);p.add_argument('--observed-weight',type=float,default=.002)
    p.add_argument('--surface-weight',type=float,default=0.)
    p.add_argument('--pixel-centers-consistent',action='store_true');p.add_argument('--ncc-factor',type=float,default=1.)
    p.add_argument('--trace-surface',action='store_true')
    p.add_argument('--whole-body-trace',action='store_true');p.add_argument('--stop-after',type=int,default=None)
    p.add_argument('--shared-face-world',type=Path,default=None);p.add_argument('--shared-face-weight',type=float,default=.002)
    p.add_argument('--shared-face-mode',choices=['plane','finite_support'],default='plane')
    p.add_argument('--native-person-alignment',type=Path,default=None)
    a=p.parse_args();main(a.run,a.steps,a.height,a.preflight_only,a.scale_weight,a.multiview_factor,a.alpha_composite_loss,a.mask_foreground_only,a.observed_correspondence,a.observed_weight,a.surface_weight,a.pixel_centers_consistent,a.ncc_factor,a.trace_surface,a.whole_body_trace,a.stop_after,a.shared_face_world,a.shared_face_weight,a.shared_face_mode,a.native_person_alignment)
