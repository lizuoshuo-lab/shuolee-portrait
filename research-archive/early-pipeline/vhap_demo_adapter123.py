"""Small input/SMPL-X/output adapter around the unchanged VHAP_smplx tracker.

The upstream Adam, stage schedules, photometric loss, landmark loss and
regularizers are imported and executed. No SMPLify-X fitting is performed.
The canonical-offset bridge matches the original FLAME before-LBS contract.
"""
from pathlib import Path
import os,sys,json,time,traceback,hashlib,fcntl,logging,dataclasses
import numpy as np
from PIL import Image
from run_official_gaussians123 import save,read,sha,now,PROJECT,ASSETS
from intake_demo_vhap123 import ROOT
from preprocess_demo_fps3_123 import DATA,REPORT

REPO=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/repos/VHAP_smplx')
COMMIT='8980d4948d91b05ff96ae91bae29e3f15a238f5e'


def imports():
    import smplx,smplx.body_models
    sys.path.insert(0,str(REPO))
    # Upstream refers to an unpublished nested import. Resolve it to the
    # already installed official SMPL-X, not a reimplemented body model.
    sys.modules['smplx.smplx']=smplx
    sys.modules['smplx.smplx.body_models_modified']=smplx.body_models
    from vhap.model.tracker_smplx import GlobalTracker
    from vhap.config import base as c
    return smplx,GlobalTracker,c


class NativeSamples:
    def __init__(self,rows,obs,confidence,device='cuda'):
        import torch
        self.rows=rows;self.obs=obs;self.confidence=confidence;self.device=device
        self.num_timesteps=len(rows);self.batchify_all_views=False;self.cache={}
        self.COCO=np.array([0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11])
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        import torch
        r=self.rows[i]
        if i not in self.cache:
            raw=np.array(Image.open(r['image']).convert('RGB'),dtype=np.uint8)
            mask=np.array(Image.open(r['mask']).convert('L'),dtype=np.uint8)
            assert list(raw.shape[1::-1])==r['wh']==list(mask.shape[1::-1])
            self.cache[i]=(torch.from_numpy(raw.copy()).permute(2,0,1),torch.from_numpy(mask.copy())[None])
        raw,mask=self.cache[i];raw=raw.float()/255.;alpha=mask.float()/255.
        # Masked white target: background texture cannot be learned as clothing.
        rgb=raw*alpha+(1-alpha)
        pts=self.obs['keypoints'][i];score=self.confidence[i]
        body=np.zeros((25,3),np.float32)
        body[self.COCO,:2]=pts[:17];body[self.COCO,2]=score[:17]
        body[19:25,:2]=pts[17:23];body[19:25,2]=score[17:23]
        face=np.c_[pts[23:91],score[23:91]].astype(np.float32)
        left=np.c_[pts[91:112],score[91:112]].astype(np.float32)
        right=np.c_[pts[112:133],score[112:133]].astype(np.float32)
        # VHAP uses corner-based pixels in its NDC conversion.
        for a in (body,face,left,right):a[:,:2]+=.5
        # Upstream OpenGL also measures the vertical principal point from the
        # opposite edge. Reuse its conversion rather than flipping axes alone.
        from vhap.util.camera import convert_camera_convention
        c2w,glK=convert_camera_convention('opencv->opengl',
            np.linalg.inv(np.asarray(r['w2c']))[None],np.asarray(r['K'])[None].copy(),raw.shape[1],raw.shape[2])
        E=np.linalg.inv(c2w[0])
        return dict(rgb=rgb,alpha=alpha,alpha_map=alpha,timestep_index=i,
            intrinsic=torch.tensor(glK[0],dtype=torch.float32),extrinsic=torch.tensor(E,dtype=torch.float32),
            lmk2d=torch.from_numpy(face),lmk2d_body=torch.from_numpy(body),lmk2d_lhand=torch.from_numpy(left),lmk2d_rhand=torch.from_numpy(right))
    def one(self,i):
        import torch
        return {k:(torch.tensor([v]) if k=='timestep_index' else v[None]) for k,v in self[i].items()}


def config(c):
    pipeline=c.PipelineConfig(c.StageLmkInitRigidConfig(),c.StageLmkInitAllConfig(),c.StageLmkSequentialTrackingConfig(),
        c.StageLmkGlobalTrackingConfig(),c.StageRgbInitTextureConfig(),c.StageRgbInitAllConfig(),c.StageRgbInitOffsetConfig(),
        c.StageRgbSequentialTrackingConfig(),c.StageRgbGlobalTrackingConfig())
    return c.BaseTrackingConfig(data=c.DataConfig(root_folder=DATA,sequence='demo',calibrated=True,scaled=True,use_alpha_map=True),
        model=c.ModelConfig(n_shape=10,n_expr=10,use_hands=True,use_hand_pca=True,num_hand_pca_comps=12,
            use_vposer=True,vposer_ckpt=ASSETS/'vposer/V02_05'),render=c.RenderConfig(),log=c.LogConfig(interval_scalar=50,interval_media=1000000),
        exp=c.ExperimentConfig(output_folder=ROOT/'vhap-fit'),lr=c.LearningRateConfig(),
        w=c.LossWeightConfig(reg_offset_relax_for=(),reg_offset_lap_relax_for=(),reg_offset_rigid_for=()),pipeline=pipeline,batch_size=1,async_func=False)


def make_tracker(out):
    import torch, scipy.sparse as sp
    from torch.func import functional_call
    smplx,GlobalTracker,c=imports()
    from vhap.util.smplify_utils import JointMapper,smpl_to_openpose
    from vhap.util.render_nvdiffrast import NVDiffRenderer
    from smplx.lbs import vertices2landmarks,blend_shapes
    from smplify_score_contract123 import relative_confidence
    cfg=config(c)
    rows=read(DATA/'data/inputs.json')['records']
    with np.load(DATA/'data/observations.npz',allow_pickle=False) as z: obs=dict(z)
    assert list(obs['frame_ids'])==[r['frame'] for r in rows]
    confidence,q90=relative_confidence(obs['scores'],obs['boxes'],obs['flip_disagreement'])
    sys.path.insert(0,str(PROJECT/'src'))
    from shuolee_portrait import runtime as rt
    rt.configure(ASSETS)
    import shuolee_vposer as vp

    class Adapter(GlobalTracker):
        def __init__(self):
            self.cfg=cfg;self.device='cuda';self.out_dir=out;self.started=time.monotonic()
            self.calibrated=True;self.scaled=True;self.global_step=0;self.stage_counts={};self.last_log={}
            self.tb_writer=None;self.logger=logging.getLogger('vhap-demo');self.logger.setLevel(logging.INFO)
            self.log_interval_scalar=50;self.log_interval_media=1000000
            self.dataset=NativeSamples(rows,obs,confidence);self.n_timesteps=len(rows);self.image_size=(1520,2704)
            self.vposer,_,self.vposer_meta=vp.load_vposer('cuda');self.hand_prior_left=None;self.hand_prior_right=None
            mapping=smpl_to_openpose(model_type='smplx',use_hands=True,use_face=True,use_face_contour=False,openpose_format='coco25')
            assert len(mapping)==118
            self.smplx=smplx.SMPLX(str(ASSETS/'smplx/SMPLX_NEUTRAL.npz'),batch_size=1,num_betas=10,num_expression_coeffs=10,
                create_transl=False,create_body_pose=False,use_face_contour=False,use_pca=True,num_pca_comps=12,
                joint_mapper=JointMapper(mapping)).cuda().eval()
            self.smplx.requires_grad_(False);self.smplx.parents=self.smplx.parents.cpu()
            with np.load(ASSETS/'smplx/SMPLX_NEUTRAL.npz',allow_pickle=False) as z:
                self.smplx.register_buffer('vt',torch.tensor(z['vt'],dtype=torch.float32,device='cuda'))
                self.smplx.register_buffer('ft',torch.tensor(z['ft'],dtype=torch.int32,device='cuda'))
            self.lhand_idx=slice(25,46);self.rhand_idx=slice(46,67)
            faces=self.smplx.faces;n=len(self.smplx.v_template)
            ii=np.concatenate([faces[:,0],faces[:,1],faces[:,2],faces[:,1],faces[:,2],faces[:,0]])
            jj=np.concatenate([faces[:,1],faces[:,2],faces[:,0],faces[:,0],faces[:,1],faces[:,2]])
            A=(sp.csr_matrix((np.ones(len(ii)),(ii,jj)),shape=(n,n))>0).astype(np.float32)
            L=(sp.diags(np.asarray(A.sum(1)).ravel())-A).tocoo()
            self.smplx_laplacian=torch.sparse_coo_tensor(torch.tensor(np.array([L.row,L.col]),dtype=torch.long),torch.tensor(L.data),(n,n)).cuda()
            self.render=NVDiffRenderer(use_opengl=False,lighting_type=cfg.render.lighting_type,lighting_space=cfg.render.lighting_space,
                disturb_rate_fg=cfg.render.disturb_rate_fg,disturb_rate_bg=cfg.render.disturb_rate_bg,fid2cid=None)
            self.init_params() # unchanged upstream parameter definitions
            with torch.no_grad():
                self.global_orient[:,0]=np.pi  # SMPL +Y-up to OpenCV +Y-down.
                # Positive-depth seed from body bbox; optimized by upstream rigid stage.
                for i,r in enumerate(rows):
                    x0,y0,x1,y1=obs['boxes'][i];K=np.asarray(r['K_integer_centers'])
                    z=K[1,1]*1.7/(y1-y0)
                    self.transl[i]=torch.tensor([((x0+x1)/2-K[0,2])*z/K[0,0],((y0+y1)/2-K[1,2])*z/K[1,1],z],device='cuda')
            self.initial_template=self.smplx.v_template.detach().clone()
            self.input_confidence=confidence;self.score_q90=q90

        def forward_smplx(self,timesteps):
            body_pose=self.vposer.decode(self.pose_embedding[timesteps])['pose_body'].reshape(len(timesteps),-1)
            params=dict(betas=self.betas[None].expand(len(timesteps),-1),expression=self.expression[timesteps],body_pose=body_pose,
                jaw_pose=self.jaw_pose[timesteps],left_hand_pose=self.left_hand_pose[timesteps],right_hand_pose=self.right_hand_pose[timesteps],
                global_orient=self.global_orient[timesteps],transl=self.transl[timesteps],leye_pose=self.leye_pose[timesteps],reye_pose=self.reye_pose[timesteps])
            # Original FLAME semantics, compatible with the existing Gaussian bridge.
            ret=functional_call(self.smplx,{'v_template':self.smplx.v_template+self.static_offset[0]},(),params)
            verts=ret.vertices
            cano=self.smplx.v_template[None]+blend_shapes(self.betas[None],self.smplx.shapedirs)+self.static_offset
            lmks_face=vertices2landmarks(verts,self.smplx.faces_tensor,
                self.smplx.lmk_faces_idx[None].expand(len(timesteps),-1).contiguous(),
                self.smplx.lmk_bary_coords[None].expand(len(timesteps),-1,-1).contiguous())
            return verts,cano.expand(len(timesteps),-1,-1),ret.joints,lmks_face,self.get_albedo().expand(len(timesteps),-1,-1,-1)

        def optimize_iter(self,sample,optimizer,stage):
            if time.monotonic()-self.started>7200:raise RuntimeError('Bounded two-hour fitting budget reached; preserve evidence.')
            super().optimize_iter(sample,optimizer,stage)
            self.stage_counts[stage]=self.stage_counts.get(stage,0)+1
            if not torch.isfinite(self.static_offset).all():raise RuntimeError('Nonfinite shared offset.')
        def log_scalars(self,log_dict,timestep,**kw):
            self.last_log={k:float(v.detach()) if torch.is_tensor(v) else float(v) for k,v in log_dict.items()}
            msg=dict(updated=now(),state='vhap_fitting',pid=os.getpid(),step=self.global_step+1,frame=int(timestep),stage=kw.get('stage'),
                seconds=time.monotonic()-self.started,losses=self.last_log,smplifyx_fitting_used=False,quality_accepted=False)
            save(out/'progress.json',msg);print(json.dumps(msg),flush=True)
        def log_media(self,*a,**kw):pass
    tracker=Adapter()
    return tracker,rows


def preflight(t,rows,out):
    import torch
    t.get_train_parameters('lmk_init_all')
    i=0;s=t.dataset.one(i)
    with torch.no_grad():v,c,j,f,a=t.forward_smplx(s['timestep_index'])
    assert float(v[...,2].min())>.1
    K=torch.tensor(rows[i]['K_integer_centers'],device='cuda',dtype=torch.float32)
    uv=(j@K.T);uv=uv[...,:2]/uv[...,2:]
    ndc=t.render.world_to_ndc(j,s['extrinsic'].cuda(),s['intrinsic'].cuda(),t.image_size,flip_y=True)
    native=(ndc[...,:2]+1)*torch.tensor([2704,1520],device='cuda')/2-.5
    pixel=float((uv-native).abs().max());assert pixel<.001,pixel
    # All model, face and hand variables must receive finite landmark gradients.
    loss,logs,*_=t.compute_energy(s,stage='lmk_init_all')
    variables=[t.betas,t.pose_embedding,t.left_hand_pose,t.right_hand_pose,t.expression,t.jaw_pose,t.transl]
    grads=torch.autograd.grad(loss,variables,allow_unused=False)
    gradcheck={n:dict(finite=bool(torch.isfinite(g).all()),max_abs=float(g.abs().max())) for n,g in zip(
        ['shared_shape','body_pose','left_hand','right_hand','face_expression','jaw','translation'],grads)}
    assert all(x['finite'] and x['max_abs']>0 for x in gradcheck.values())
    t.get_train_parameters('rgb_init_offset')
    loss,logs,*_=t.compute_energy(s,stage='rgb_init_offset')
    grad,=torch.autograd.grad(loss,t.static_offset)
    assert torch.isfinite(grad).all() and grad.abs().max()>0
    result=dict(state='passed',updated=now(),native_projection_max_abs_px=pixel,landmark_gradient=gradcheck,
        offset_gradient_max_abs=float(grad.abs().max()),vertices=len(t.smplx.v_template),triangles=len(t.smplx.faces),
        UV_present=True,offset_domain='canonical before LBS',optimizer_updates=0,smplifyx_fitting_used=False,quality_accepted=False)
    save(out/'preflight.json',result);return result


def snapshot(t,rows,out,label,ids):
    import torch
    from review_full_sequence123 import mesh_renderer,joined,crop_box
    render_mesh,_=mesh_renderer();dest=out/label;dest.mkdir(exist_ok=False)
    arrays={k:getattr(t,k).detach().cpu().numpy() for k in ['betas','expression','global_orient','transl','pose_embedding','jaw_pose','leye_pose','reye_pose','left_hand_pose','right_hand_pose','static_offset','tex_extra','lights']}
    arrays['frame_ids']=np.asarray([r['frame'] for r in rows]);arrays['faces']=t.smplx.faces
    arrays['K_integer_centers']=np.array([r['K_integer_centers'] for r in rows]);arrays['w2c']=np.array([r['w2c'] for r in rows])
    np.savez_compressed(dest/'tracked-smplx-params.npz',**arrays)
    metrics=[];images=[]
    for i in ids:
        s=t.dataset.one(i);r=rows[i]
        with torch.no_grad():v,c,j,f,a=t.forward_smplx(s['timestep_index'])
        K=np.array(r['K_integer_centers']);v=v[0].cpu().numpy();j=j[0].cpu().numpy()
        raw=Image.open(r['image']).convert('RGB');overlay=render_mesh(raw,np.array([0,0,*raw.size]),dict(vertices_camera=v[None],faces=t.smplx.faces,K_integer_centers=K))
        points=t.dataset.obs['keypoints'][i]
        boxes=dict(body=r['initial_box_xyxy'],face=np.r_[points[23:91].min(0),points[23:91].max(0)],
            left_hand=np.r_[points[91:112].min(0),points[91:112].max(0)],right_hand=np.r_[points[112:133].min(0),points[112:133].max(0)])
        for region,box in boxes.items():
            crop=crop_box(box,raw.size,12);im=joined([raw.crop(crop),overlay.crop(crop)])
            name=f'{r["frame"]}_{region}.png';im.save(dest/name);images.append(dict(frame=r['frame'],region=region,path=str(dest/name),crop=crop))
        uv=j@K.T;uv=uv[:,:2]/uv[:,2:]
        obs=t.dataset.obs;C=t.input_confidence[i];target=np.zeros((118,3),np.float32)
        dst=np.r_[t.dataset.COCO,np.arange(19,25),np.arange(25,67),np.arange(67,118)]
        src=np.r_[np.arange(23),np.arange(91,133),np.arange(40,91)]
        target[dst,:2]=obs['keypoints'][i,src];target[dst,2]=C[src]
        err=np.linalg.norm(uv-target[:,:2],axis=1)
        groups={}
        for region,jj in dict(body=t.dataset.COCO,face=np.arange(67,118),left_hand=np.arange(25,46),right_hand=np.arange(46,67)).items():
            valid=jj[target[jj,2]>0];e=err[valid]
            groups[region]=dict(count=len(e),median=float(np.median(e)),p90=float(np.quantile(e,.9)),max=float(e.max()))
        metrics.append(dict(frame=r['frame'],groups=groups))
    d=np.linalg.norm(arrays['static_offset'][0],axis=1)*1000
    result=dict(updated=now(),state='saved_pending_visual_review',stage=label,images=images,metrics=metrics,
        offset_mm=dict(median=float(np.median(d)),p90=float(np.quantile(d,.9)),max=float(d.max())),
        all_errors_detector_fit_not_independent_ground_truth=True,quality_accepted=False)
    save(dest/'result.json',result);return result


def main():
    import argparse,subprocess,torch
    p=argparse.ArgumentParser();p.add_argument('--preflight-only',action='store_true');args=p.parse_args()
    assert read(REPORT/'result.json')['state']=='complete_pending_visual_review'
    out=ROOT/('vhap-preflight-camera-bridge' if args.preflight_only else 'vhap-fit')
    assert not out.exists(),'Do not repeat or overwrite a previous run.'
    assert subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()==COMMIT
    # A Windows checkout uses CRLF and mount permissions in WSL. Normalize
    # only those two git comparison settings; real content changes still fail.
    assert subprocess.check_output(['git','-C',str(REPO),'-c','core.autocrlf=true',
        '-c','core.filemode=false','status','--porcelain'],text=True)==''
    lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out.mkdir();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(6060)
    source_files=[REPO/'vhap/model/tracker_smplx.py',REPO/'vhap/config/base.py',REPO/'vhap/util/render_nvdiffrast.py',Path(__file__)]
    provenance=dict(updated=now(),upstream='https://github.com/jmaraval/VHAP_smplx',commit=COMMIT,
        files={str(p):sha(p) for p in source_files},inputs={str(DATA/p):sha(DATA/p) for p in ['data/inputs.json','data/observations.npz']},
        adaptations=['Native images/masks/RTMW133 to original landmark inputs; exact fullK pixel bridge',
            'Original convert_camera_convention for both c2w axes and vertical principal point; manifests stay OpenCV',
            'Official installed SMPLX and existing VPoser assets; UV from the same official model archive',
            'Shared offset before LBS as original FLAME and downstream Gaussian contract; fork post-pose offset not reused',
            'Positive-depth bbox seed in camera coordinates; upstream rigid initialization optimizes it',
            'Native input/output and progress callbacks; upstream stage optimizer/loss unchanged'],
        no_smplifyx_fitting=True,no_old_subject_mesh=True,no_new_K=True,no_new_losses=True,quality_accepted=False)
    save(out/'input-provenance.json',provenance)
    try:
        t,rows=make_tracker(out)
        save(out/'config.json',json.loads(json.dumps(dataclasses.asdict(t.cfg),default=str)))
        result=preflight(t,rows,out);print(json.dumps(result),flush=True)
        snapshot(t,rows,out,'zero-init',[0])
        if args.preflight_only:return
        s=t.dataset.one(0)
        for stage in ['lmk_init_rigid','lmk_init_all','rgb_init_texture','rgb_init_all','rgb_init_offset']:
            t.optimize_stage(stage,sample=s)
            snapshot(t,rows,out,stage,[0])
            save(out/'stage-completed.json',dict(stage=stage,step=t.global_step,updated=now()))
        save(out/'progress.json',dict(state='reference_mesh_review_pending',pid=None,updated=now(),step=t.global_step,
            full_sequence_complete=False,smplifyx_fitting_used=False,quality_accepted=False))
    except BaseException as e:
        save(out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),updated=now()));raise

if __name__=='__main__':main()
