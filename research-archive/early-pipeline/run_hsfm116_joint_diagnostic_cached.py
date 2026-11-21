"""Bounded headless native HSfM-SMPL diagnostic; two losses, one frozen input.

This is an explicitly adapted diagnostic, not a complete official reproduction.
Original sources and074 cameras are immutable. No Gaussian training or export.
"""
from pathlib import Path
import argparse,ast,collections,copy,datetime as dt,gc,hashlib,json,os,subprocess,sys,time,traceback
import numpy as np
from PIL import Image,ImageDraw
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
B=LAB/'outputs/mini-splatting2-mip-depth-001/shuolee';H=B/'current/portrait-alignment-open-source-116/hsfm'
J=H/'joint-diagnostic-reserved';O=J/'native-fit-cached';R=Path('/opt/gs-slam-lab/shuolee-hsfm116')
PRIVATE=B/'current/novel-view-face/head-structure/private-runtime/python-deps'
sys.path[:0]=[str(R/'dust3r'),str(R/'croco'),str(R/'python'),str(PRIVATE),str(PRIVATE/'smplx-official')]
from hsfm116_adapter import map_pixels,estimate_root_translation
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,v):
    p=O/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def literal(n):
    if isinstance(n,ast.Constant):return n.value
    if isinstance(n,(ast.Tuple,ast.List)):return tuple(literal(x) for x in n.elts) if isinstance(n,ast.Tuple) else [literal(x) for x in n.elts]
    if isinstance(n,ast.Name) and n.id=='inf':return float('inf')
    if isinstance(n,ast.UnaryOp) and isinstance(n.op,ast.USub):return -literal(n.operand)
    raise ValueError(ast.dump(n))
def main():
    if O.exists():
        assert (O/'before-scalar-failure.json').exists() and not (O/'protocol.json').exists() and not (O/'shared-initialization.pt').exists(), 'Do not repeat an initialized fit'
    else:O.mkdir()
    assert read(J/'frozen-validation.json')['coverage_passed']
    assert (J/'fresh-hmr/result.json').exists()
    t0=time.monotonic()
    def progress(stage,**k):save('progress.json',dict(updated=now(),state='running',stage=stage,pid=os.getpid(),seconds=time.monotonic()-t0,**k))
    gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True)
    processes=[]
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:cmd=p.read_bytes().replace(b'\0',b' ').decode(errors='replace')
        except OSError:continue
        if any(s in cmd for s in ('python','train','supervis')):processes.append(dict(pid=int(p.parent.name),command=cmd))
    save('process-check.json',dict(updated=now(),pid=os.getpid(),processes=processes,gpu=gpu.stdout));assert gpu.returncode==0 and not gpu.stdout.strip()
    rows=read(H/'input-pilot/inputs.json')['records'];ids=[r['raw_frame'] for r in rows]; names=[str(f) for f in ids]
    add=read(J/'implementation-addendum.json')
    save('protocol.json',dict(created=now(),script_sha256=sha(__file__),preregistration_sha256=sha(J/'preregistration.json'),addendum=add,
        additional_gauge_safeguard='Both arms fix firstcamera pose and geometric mean pair-scale at the SAME shared initialization. Native norm_pw_scale=False would allow a scene-only L1scale-collapse confound.',
        native_core='Pinned DUSt3R PointCloudOptimizer + AST extracted HSfM get_human_loss, bboxxyxy correction only',
        first_camera_raw_id=100,raw_frame_ids=ids,validation_sha256=sha(J/'frozen-validation.npz'),fresh_hmr_sha256=sha(J/'fresh-hmr/fresh-priors.npz'),
        local_adapter=True,official_complete_reproduction=False,budget_seconds=1450,gaussian_training=False))
    import torch,torch.nn.functional as F,cv2,smplx
    from scipy.spatial.transform import Rotation
    from dust3r.model import AsymmetricCroCo3DStereo
    from dust3r.inference import inference
    from dust3r.utils.image import load_images
    from dust3r.utils.device import collate_with_cat
    from dust3r.utils.geometry import geotrf
    from dust3r.cloud_opt.optimizer import PointCloudOptimizer
    torch.manual_seed(116);torch.set_num_threads(2);torch.cuda.reset_peak_memory_stats()
    ratio=float(rows[0]['K_processed_centers'][1][1]/rows[0]['K_processed_centers'][0][0])
    class PortraitOptimizer(PointCloudOptimizer):
        def get_intrinsics(self):
            K=super().get_intrinsics();K[:,1,1]=K[:,1,1]*ratio;return K
        def depth_to_pts3d(self):
            dep=self.get_depthmaps(raw=True)[...,None];f=self.get_focals();fxy=torch.cat([f,f*ratio],dim=1)
            rel=torch.cat([dep*(self._grid-self.get_principal_points()[:,None,:])/fxy[:,None,:],dep],dim=-1)
            return geotrf(self.get_im_poses(),rel)
    # One bounded repair after projector assertion; no network reinference or old fitting.
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    progress('loading_preserved_native_cache')
    output=torch.load(J/'native-fit/native-156-pair-output.pt',map_location='cpu',weights_only=False)
    forward_seconds=0.
    scene=PortraitOptimizer(*[output[k] for k in ('view1','view2','pred1','pred2')],verbose=False).cuda()
    scene.load_state_dict(torch.load(J/'projector-investigation/native-mst-state.pt',map_location='cpu',weights_only=True))
    save('repair-evidence.json',dict(updated=now(),previous_failure='../native-fit/failure.json',
        investigation='../projector-investigation/result.json',
        reason='Original first-run round trip was0.316px and stopped before fitting. Preserved-cache investigation passed9.16e-5px float32 and1.38e-13px float64. Original state was not saved, so its exact transient cause is not established.',
        repair='Use the preserved, unoptimized and numerically checked MST state with highest float32 precision. Same initialization for both arms. Do not relax projection tolerance. No network reinference, parameter sweep or prior subject fit.',
        state_sha256=sha(J/'projector-investigation/native-mst-state.pt'),network_reinference=False))
    # Native point-depth and camera projection MUST use the same portrait K.
    with torch.no_grad():
        xyz=scene.get_pts3d(raw=True);c2w=scene.get_im_poses();K=scene.get_intrinsics()
        idx=torch.arange(0,xyz.shape[1],max(1,xyz.shape[1]//64),device='cuda')
        X=geotrf(torch.linalg.inv(c2w),xyz[:,idx]);uv=torch.einsum('nij,nkj->nki',K,X);uv=uv[...,:2]/uv[...,2:]
        err=(uv-scene._grid[:,idx]).abs().max().item();assert err<.02,err
    save('actual-projector-check.json',dict(max_processed_px=err,contract='Native depth_to_pts3d -> inverse camera -> same K -> original integer grid',fy_fx_ratio=ratio))
    # Extract only upstream math functions, without demo/visualization side effects.
    ns={'torch':torch,'F':F,'defaultdict':collections.defaultdict}
    jt=ast.parse((H/'source/joint_names.py').read_text())
    for n in jt.body:
        if isinstance(n,ast.Assign) and isinstance(n.value,ast.List):
            for t in n.targets:
                if isinstance(t,ast.Name):ns[t.id]=ast.literal_eval(n.value)
    smpl45=next(n for n in jt.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SMPL_45_KEYPOINTS' for t in n.targets))
    assert isinstance(smpl45.value,ast.BinOp) and isinstance(smpl45.value.op,ast.Add) and isinstance(smpl45.value.left,ast.Name) and smpl45.value.left.id=='SMPL_KEYPOINTS'
    ns['SMPL_45_KEYPOINTS']=ns['SMPL_KEYPOINTS']+ast.literal_eval(smpl45.value.right)
    assert len(ns['SMPL_45_KEYPOINTS'])==45
    assert [ns['SMPL_45_KEYPOINTS'].index(x) for x in ns['COCO_WHOLEBODY_KEYPOINTS'][:23]]==[24,26,25,28,27,16,17,18,19,20,21,1,2,4,5,7,8,29,30,31,32,33,34]
    ns['coco_main_body_joint_idx']=list(range(23))
    source=(H/'source/align_world_env_and_smpl_hsfm_optim.py').read_text()
    assert source.count('bbox_area = bbox[2] * bbox[3]')==1
    source=source.replace('bbox_area = bbox[2] * bbox[3]','bbox_area = (bbox[2]-bbox[0]) * (bbox[3]-bbox[1])')
    tree=ast.parse(source);selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('get_human_loss','rotation_6d_to_matrix','matrix_to_rotation_6d')]
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(H/'source/align_world_env_and_smpl_hsfm_optim.py'),'exec'),ns)
    loss_fn=ns['get_human_loss'];to6=ns['matrix_to_rotation_6d']
    pri=np.load(J/'fresh-hmr/fresh-priors.npz');obs=np.load(H/'input-pilot/observations.npz');vis=np.load(J/'visibility.npz')
    assert np.array_equal(pri['raw_frame_ids'],ids)
    poseobs={};boxes={};roots=[];mapping=pri['smpl45_to_body23']
    Kinitial=scene.get_intrinsics().detach().cpu().numpy()
    for i,r in enumerate(rows):
        xy=map_pixels(obs['keypoints'][i],r['center_affine']);conf=vis['scores'][i]
        initconf=conf if (conf>=.5).sum()>=4 else obs['scores'][i,:23]
        root,_=estimate_root_translation(pri['root_relative_joints'][i,mapping],xy[:23],Kinitial[i],initconf,.5);roots.append(root)
        if (conf>0).any():
            data=np.c_[xy,np.zeros(133)];data[:23,2]=conf
            poseobs[str(r['raw_frame'])]=torch.tensor(data,device='cuda',dtype=torch.float32)
            box=map_pixels(np.array(r['bbox_xyxy']).reshape(2,2),r['edge_affine']).reshape(-1)
            boxes[str(r['raw_frame'])]=torch.tensor([*box,float(conf[conf>0].mean())],device='cuda',dtype=torch.float32)
    roots=np.asarray(roots);rots=Rotation.from_rotvec(pri['global_orient']).as_matrix();human_centers=roots[0]-np.einsum('nij,nj->ni',rots[0]@rots.transpose(0,2,1),roots)
    with torch.no_grad():
        c2w=scene.get_im_poses();G=torch.linalg.inv(c2w[0]);points=scene.get_pts3d();c2w=G@c2w;points=[geotrf(G,p) for p in points]
        cams=c2w[:,:3,3].cpu().numpy();signed_scale=float((cams*human_centers).sum()/(cams*cams).sum());scale=abs(signed_scale)
        assert np.isfinite(scale) and 1e-4<scale<1e4
        c2w[:,:3,3]*=scale;points=[p*scale for p in points];scene.norm_pw_scale=False
        scene.init_from_known_params_hongsuk(scene.get_focals().detach().cpu().flatten().tolist(),c2w,points)
    layer=smplx.SMPL(str(LAB/'repos/drivestudio/smpl_models/SMPL_NEUTRAL.pkl'),gender='neutral',num_betas=10,batch_size=1).cuda()
    def param(a):return torch.nn.Parameter(torch.as_tensor(a,device='cuda',dtype=torch.float32))
    human={'0':dict(betas=param(pri['betas'][0]),root_transl=param(roots[0]),
                    global_orient=param(to6(torch.tensor(rots[0],dtype=torch.float32)).numpy()),
                    body_pose=param(to6(torch.tensor(Rotation.from_rotvec(pri['body_pose'][0].reshape(23,3)).as_matrix(),dtype=torch.float32)).numpy()))}
    def h_loss(cam,KK,human):return loss_fn({1:layer},'smpl',human,names,torch.linalg.inv(cam),KK,{'0':poseobs},{'0':boxes},only_main_body_joints=True,shape_prior_weight=1.)
    initial_cam=scene.get_im_poses().detach();initial_K=scene.get_intrinsics().detach();log_res=param(0.)
    # Shared root/shape/scale registration is done once, then cloned into both arms.
    for k,p in human['0'].items():p.requires_grad_(k in ('root_transl','betas'))
    opt=torch.optim.Adam([human['0']['root_transl'],human['0']['betas'],log_res],lr=.015,betas=(.9,.9))
    for step in range(200):
        opt.zero_grad();cam=initial_cam.clone();cam[:,:3,3]=initial_cam[:,:3,3]*log_res.exp()
        loss,_=h_loss(cam,initial_K,human);assert torch.isfinite(loss);loss.backward();opt.step()
        if step%50==0:progress('shared_body_registration',step=step,loss=float(loss.detach()))
    with torch.no_grad():
        residual=float(log_res.exp());assert .01<residual<100
        c2w=scene.get_im_poses();c2w[:,:3,3]*=residual;points=[p*residual for p in scene.get_pts3d()]
        scene.init_from_known_params_hongsuk(scene.get_focals().detach().cpu().flatten().tolist(),c2w,points)
        scene.base_scale=float(scene.pw_poses[:,-1].mean().exp());scene.norm_pw_scale=True
    # Freeze validation pixels out of both native dense losses; no camera-based filtering.
    validation=np.load(J/'frozen-validation.npz');masks=np.ones((len(ids),512,288),np.float32)
    for frow,xy in zip(validation['frames'],validation['xy']):
        for f,p in zip(frow,xy):
            i=ids.index(int(f));q=np.rint(map_pixels(p,rows[i]['center_affine'])).astype(int);cv2.circle(masks[i],tuple(q),3,0,-1)
    masks=torch.tensor(masks.reshape(len(ids),-1),device='cuda')
    scene._weight_i*=masks[scene._ei];scene._weight_j*=masks[scene._ej]
    initial_state={k:v.detach().clone() for k,v in scene.state_dict().items()};human_state={k:v.detach().clone() for k,v in human['0'].items()}
    torch.save(dict(scene=initial_state,human=human_state,base_scale=scene.base_scale),O/'shared-initialization.pt')
    save('gauge.json',dict(initial_native_to_first_camera=G.cpu().numpy().tolist(),signed_HMR_scale=signed_scale,applied_positive_scale=scale,shared_residual_scale=residual,
        geometric_mean_pair_scale=scene.base_scale,first_frame=ids[0],fixed_first_camera=True,world_convention='Firstcamera OpenCV xright ydown zforward; diagnostic only, not finalworld/+YPLY',
        baseline074_untouched=True))
    for arm in ['scene_only','scene_human']:
        scene.load_state_dict(initial_state)
        for k,p in human['0'].items():p.data.copy_(human_state[k]);p.requires_grad_(arm=='scene_human' and k!='body_pose')
        sparams=[p for p in scene.parameters() if p.requires_grad];hparams=list(human['0'].values()) if arm=='scene_human' else []
        opt=torch.optim.Adam(sparams+hparams,lr=.015,betas=(.9,.9));first=scene.im_poses[0].detach().clone();trace=[]
        for step in range(500):
            if arm=='scene_human' and step==400:human['0']['body_pose'].requires_grad_(True)
            lr=.015+(.0001-.015)*step/499
            for g in opt.param_groups:g['lr']=lr
            opt.zero_grad();ls=scene();lh=torch.zeros((),device='cuda')
            if arm=='scene_human':lh,_=h_loss(scene.get_im_poses(),scene.get_intrinsics(),human)
            loss=ls+5*lh;assert torch.isfinite(loss),f'{arm} nonfinite at{step}'
            loss.backward();scene.im_poses.grad[0].zero_();opt.step()
            with torch.no_grad():scene.im_poses[0].copy_(first)
            if step%25==0 or step==499:
                row=dict(step=step+1,scene=float(ls.detach()),human=float(lh.detach()),total=float(loss.detach()));trace.append(row);progress(arm,**row)
            assert time.monotonic()-t0<1450,'P3 GPU diagnostic wall budget exceeded'
        K=scene.get_intrinsics().detach().cpu().numpy();poses=scene.get_im_poses().detach().cpu().numpy()
        Kraw=np.stack([np.linalg.inv(np.array(r['center_affine']))@k for r,k in zip(rows,K)])
        assert np.isfinite(Kraw).all() and np.isfinite(poses).all() and (Kraw[:,0,0]>0).all()
        np.savez_compressed(O/f'{arm}-cameras.npz',raw_frame_ids=ids,K_centers=Kraw,c2w=poses,K_processed=K,
                            **{f'body_{k}':v.detach().cpu().numpy() for k,v in human['0'].items()})
        save(f'{arm}-trace.json',trace)
        torch.save(scene.state_dict(),O/f'{arm}-native-state.pt')
        save(f'{arm}-result.json',dict(updated=now(),state='fit_complete_evaluation_pending',steps=500,frames=len(ids),loss=trace[-1],
            camera_convention='Kzero-based raster centers on rectified fullimage; c2w to common firstcam gauge',fresh_initializer=True,gaussian_training=False))
    save('result.json',dict(updated=now(),state='paired_native_fit_complete_evaluation_pending',frames=13,steps_per_arm=500,shared_registration_steps=200,
        seconds=time.monotonic()-t0,gpu_forward_seconds=forward_seconds,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
        quality_accepted=False,full175_validated=False,root_cause_resolved=False,gaussian_training=False,local_hsfm_adapter=True))
    save('progress.json',dict(updated=now(),state='completed',pid=os.getpid()));print('Paired native alignment diagnostic complete',flush=True)
try:main()
except Exception as e:
    if O.exists():save('failure.json',dict(updated=now(),error=repr(e),traceback=traceback.format_exc()));save('progress.json',dict(updated=now(),state='failed',pid=os.getpid(),error=repr(e)))
    raise
