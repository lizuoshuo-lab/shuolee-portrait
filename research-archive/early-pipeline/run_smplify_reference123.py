"""Bounded, separate-image reference using pinned SMPLify-X loss and priors.

This is a local full-K/RTMW/VPoser-v2 adapter, not an official full reproduction.
No old shared fit, background camera, or trained Gaussian enters initialization.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, ast, fcntl, hashlib, importlib.util, json, os, sys, time, traceback
import numpy as np
import torch
import torch.nn as nn
import smplx
from smplify_score_contract123 import relative_confidence, require_relative_confidence

T=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
O=T/'shuolee/current/local-clothed-surface-123/smplifyx-reference-control'
M=O.parent/'grass-exploratory/provenance-audit/mapping-audit'
R=Path('/opt/gs-slam-lab/shuolee-local-surface-123/grass-exploratory')
F=R.parent/'runtime'
A=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets')
DEVICE='cuda'
COCO=np.array([0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11])
NAMES=['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle']
BP=[404,404,57.4,4.78,4.78]
SHAPE=[100,50,10,5,5]
HAND=[0,0,0,.1,2]
FACE=[0,0,0,0,2]
JAW=[[4040,40400,40400],[4040,40400,40400],[574,5740,5740],[47.8,478,478],[47.8,478,478]]

def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def t(x):return torch.as_tensor(x,dtype=torch.float32,device=DEVICE)
def stats(x):
    x=np.asarray(x).reshape(-1)
    return {'n':len(x),'median':float(np.median(x)),'p90':float(np.quantile(x,.9)),'max':float(x.max())} if len(x) else {'n':0}
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def setup():
    torch.set_num_threads(4);torch.manual_seed(12301);np.random.seed(12301)
    assert torch.cuda.is_available()
    utils=module(M/'official/smplifyx/utils.py','official_utils123')
    prior=module(O/'official/smplifyx/prior.py','official_prior123')
    # Execute the exact two upstream loss classes, excluding the unused GUI imports.
    tree=ast.parse((O/'official/smplifyx/fitting.py').read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in ['SMPLifyLoss','SMPLifyCameraInitLoss']]
    assert len(nodes)==2
    scope={'torch':torch,'nn':nn,'np':np,'utils':utils}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'pinned-official-fitting-loss-classes','exec'),scope)
    mapping=utils.smpl_to_openpose(model_type='smplx',use_hands=True,use_face=True,use_face_contour=False,openpose_format='coco25')
    from smplx.joint_names import JOINT_NAMES
    assert len(mapping)==118 and [JOINT_NAMES[i] for i in mapping[COCO]]==NAMES
    assert np.array_equal(mapping[67:],np.arange(76,127))
    sys.path.insert(0,str(F/'src/avatar_core'))
    import shuolee_vposer as vp
    vp.MODEL=A/'vposer/V02_05';vp.CODE=F/'third_party/human_body_prior'
    vposer,_,meta=vp.load_vposer(DEVICE)
    return utils,prior,scope,mapping,vposer,meta

class FixedK(nn.Module):
    def __init__(self,K,body):
        super().__init__();self.register_buffer('K',t(K));self.body_ref=body
    @property
    def translation(self):return self.body_ref.transl
    def forward(self,joints):
        q=joints@self.K.T
        return q[...,:2]/q[...,2:]

class Ref:
    def __init__(self,parts,K,params):
        self.utils,self.prior,self.classes,self.mapping,self.vposer,_=parts
        self.body=smplx.SMPLX(str(A/'smplx/SMPLX_NEUTRAL.npz'),num_betas=10,num_expression_coeffs=10,use_pca=True,num_pca_comps=12,flat_hand_mean=False,use_face_contour=False,batch_size=1,joint_mapper=self.utils.JointMapper(self.mapping)).to(DEVICE)
        with torch.no_grad():
            for key in ['betas','global_orient','transl']:getattr(self.body,key).copy_(t(params[key]).reshape_as(getattr(self.body,key)))
            self.latent=nn.Parameter(self.vposer.encode(t(params['body_pose']).reshape(1,63)).mean.detach().clone())
        for p in self.body.parameters():p.requires_grad_(False)
        self.param={'global_orient':self.body.global_orient,'transl':self.body.transl,'latent':self.latent,'betas':self.body.betas,'left_hand_pca':self.body.left_hand_pose,'right_hand_pca':self.body.right_hand_pose,'jaw_pose':self.body.jaw_pose,'expression':self.body.expression}
        self.camera=FixedK(K,self.body)
        self.loss=self.classes['SMPLifyLoss'](rho=100,body_pose_prior=self.prior.L2Prior(),shape_prior=self.prior.L2Prior(),expr_prior=self.prior.L2Prior(),angle_prior=self.prior.SMPLifyAnglePrior(),jaw_prior=self.prior.L2Prior(),use_joints_conf=True,use_face=True,use_hands=True,left_hand_prior=self.prior.L2Prior(),right_hand_prior=self.prior.L2Prior(),interpenetration=False).to(DEVICE)
    def forward(self):return self.body(body_pose=self.vposer.decode(self.latent)['pose_body'].reshape(1,63),return_verts=True,return_full_pose=True)
    def params_on(self,names):
        for name,p in self.param.items():p.requires_grad_(name in names)
        return [self.param[n] for n in names]
    def snapshot(self):return {k:v.detach().clone() for k,v in self.param.items()}
    def restore(self,values):
        with torch.no_grad():
            for k,v in values.items():self.param[k].copy_(v)
    def weights(self,stage,height):
        self.loss.reset_loss_weights({'data_weight':1000/height,'body_pose_weight':BP[stage],'shape_weight':SHAPE[stage],'bending_prior_weight':3.17*BP[stage],'hand_prior_weight':BP[stage],'expr_prior_weight':SHAPE[stage],'jaw_prior_weight':JAW[stage]})
        jw=torch.ones((1,118),device=DEVICE);jw[:,[1,9,12]]=0 # official ignored joints; slot8 has no observed confidence
        jw[:,25:67]=HAND[stage];jw[:,67:]=FACE[stage]
        return jw
    def objective(self,out,gt,conf,jw):return self.loss(out,self.camera,gt,conf,self.body.faces_tensor,jw,use_vposer=True,pose_embedding=self.latent)

def inputs():
    plan=read(O/'plan.json')
    # Verify all recorded source bytes; failed fit is only hashed, never loaded.
    for path,h in plan['input_hashes'].items():assert sha(path)==h,path
    rows=read(R/'data/inputs.json')['records'];obs=dict(np.load(R/'data/observations.npz'));pred=dict(np.load(R/'inference/prediction-postprocessed.npz'))
    obs['fit_confidence'],_=relative_confidence(obs['scores'],obs['boxes'],obs['flip_disagreement'])
    return plan,rows,obs,pred
def parameters(pred,slot):return {k:pred['incam_'+k][slot] for k in ['body_pose','betas','global_orient','transl']}
def targets(obs,slot):
    gt=np.zeros((1,118,2),np.float32);conf=np.zeros((1,118),np.float32)
    ids=np.r_[COCO,np.arange(19,25),np.arange(25,67),np.arange(67,118)]
    src=np.r_[np.arange(23),np.arange(91,133),np.arange(40,91)]
    assert len(ids)==116 and len(np.unique(ids))==116 and len(src)==116
    gt[0,ids]=obs['keypoints'][slot,src];conf[0,ids]=require_relative_confidence(obs)[slot,src]
    assert np.isfinite(gt).all() and np.isfinite(conf).all()
    return t(gt),t(conf)
def errors(uv,gt,conf):
    d=np.linalg.norm(uv-gt,axis=-1)[0];ok=conf[0]>.25
    groups={'body17':COCO,'feet6':np.arange(19,25),'left_hand21':np.arange(25,46),'right_hand21':np.arange(46,67),'face51':np.arange(67,118)}
    return {name:stats(d[ids][ok[ids]]) for name,ids in groups.items()}

def controls():
    assert not (O/'controls.json').exists(),'Do not rerun completed controls'
    assert (O/'implementation-contract.json').exists() and (O/'input-visual-review.json').exists()
    plan,rows,obs,pred=inputs();parts=setup();ref=Ref(parts,rows[31]['K_integer_centers'],parameters(pred,31));checks={}
    # Independent numpy complete-K projection, with skew and noncentral principal point.
    K=np.array([[1120.,1.5,1301.25],[0,1119.,721.75],[0,0,1.]])
    xyz=np.array([[-.1,.2,2.3],[.3,-.4,4.1],[.6,.1,2.7]],np.float32)
    testcam=FixedK(K,ref.body);actual=testcam(t(xyz)[None]).detach().cpu().numpy()[0];q=xyz.astype(np.float64)@K.T
    checks['fullK_numpy_projection_max_px']=float(np.max(np.abs(actual-q[:,:2]/q[:,2:])))
    assert checks['fullK_numpy_projection_max_px']<.0003
    # All optimized variables must affect the actual projected objective, without relying on prior gradients.
    ref.params_on(list(ref.param));out=ref.forward();uv=ref.camera(out.joints)
    offset=torch.sin(torch.arange(236,device=DEVICE).float()).reshape(1,118,2)*2+.5
    target=uv.detach()+offset
    ((uv-target)**2).sum().backward()
    checks['projection_only_gradients']={k:float(v.grad.abs().max()) if v.grad is not None else None for k,v in ref.param.items()}
    assert all(x is not None and np.isfinite(x) and x>0 for x in checks['projection_only_gradients'].values())
    # Exact adapter synthetic root recovery, generated from known full SMPL-X geometry.
    ref.params_on(['global_orient','transl']);known=ref.snapshot()
    with torch.no_grad():ref.body.transl.add_(t([[.045,-.035,.12]]));ref.body.global_orient.add_(t([[.02,-.025,.015]]))
    opt=torch.optim.LBFGS([ref.body.global_orient,ref.body.transl],lr=1,max_iter=60,max_eval=90,tolerance_grad=1e-7,tolerance_change=1e-10,line_search_fn='strong_wolfe')
    gt=uv.detach();counter=0
    def closure():
        nonlocal counter
        counter+=1;opt.zero_grad();loss=((ref.camera(ref.forward().joints)-gt)**2).mean();loss.backward();return loss
    opt.step(closure)
    with torch.no_grad():e=torch.linalg.vector_norm(ref.camera(ref.forward().joints)-gt,dim=-1)
    checks['synthetic_root_recovery']={'evals':counter,'max_px':float(e.max()),'translation_max':float((ref.body.transl-known['transl']).abs().max()),'fixed_K_unchanged':True}
    assert float(e.max())<.05
    # Exercise the upstream objective and its vector jaw weights with all groups active.
    gt0,conf0=targets(obs,31);ref.params_on(list(ref.param));ref.body.zero_grad();ref.latent.grad=None
    jw=ref.weights(4,1520);loss=ref.objective(ref.forward(),gt0,conf0,jw);loss.backward()
    assert torch.isfinite(loss)
    checks['official_loss_all_parameters_finite']=all(v.grad is not None and bool(torch.isfinite(v.grad).all()) for v in ref.param.values())
    assert checks['official_loss_all_parameters_finite']
    save(O/'controls.json',{'created':now(),'state':'passed','checks':checks,'source_sha256':sha(__file__),'vposer':parts[-1],'real_fit':False,'physical_accuracy':False})
    print(json.dumps(checks),flush=True)

class BudgetEnd(RuntimeError):pass

def run():
    assert read(O/'controls.json')['state']=='passed' and not (O/'result.json').exists()
    assert not any(O.glob('slot*-result.json')),'Do not resume/cross-initialize/repeat a real reference view'
    plan,rows,obs,pred=inputs();parts=setup();results=[]
    save(O/'progress.json',{'state':'running','pid':os.getpid(),'created':now(),'training_steps':0})
    for slot in plan['views']:
        row=rows[slot];K=np.array(row['K_integer_centers']);ref=Ref(parts,K,parameters(pred,row['official_prediction_slot']));gt,conf=targets(obs,slot)
        stages=[];eval_count=0;view_start=time.monotonic();status='complete';initial=ref.snapshot()
        with torch.no_grad():initial_out=ref.forward();initial_uv=ref.camera(initial_out.joints).cpu().numpy();initial_vertices=initial_out.vertices.cpu().numpy()
        def one_stage(name,names,objective,max_iter,max_eval):
            nonlocal eval_count
            params=ref.params_on(names);opt=torch.optim.LBFGS(params,lr=1,max_iter=max_iter,max_eval=max_eval,tolerance_grad=1e-9,tolerance_change=1e-9,line_search_fn='strong_wolfe')
            count0=eval_count;best_loss=float('inf');best_state=ref.snapshot();initial_value=None
            def closure():
                nonlocal eval_count,best_loss,best_state,initial_value
                torch.cuda.synchronize()
                if eval_count>=300 or time.monotonic()-view_start>=180:raise BudgetEnd('Predeclared per-view budget')
                opt.zero_grad();out=ref.forward()
                if not bool(torch.isfinite(out.vertices).all()) or float(out.joints[...,2].min())<=.05:raise RuntimeError('Invalid geometry or depth')
                loss=objective(out)
                if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite loss')
                value=float(loss.detach());eval_count+=1
                if initial_value is None:initial_value=value
                if value<best_loss:best_loss=value;best_state=ref.snapshot()
                loss.backward()
                if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in params):raise RuntimeError('Nonfinite/missing optimized gradient')
                return loss
            try:opt.step(closure)
            except Exception:
                ref.restore(best_state)
                stages.append({'name':name,'closure_evaluations':eval_count-count0,'initial_loss':initial_value,'best_loss':best_loss if np.isfinite(best_loss) else None,'ended':'stopped'})
                raise
            # Keep the best evaluated state, not an unrecorded line-search trial.
            ref.restore(best_state)
            stages.append({'name':name,'closure_evaluations':eval_count-count0,'initial_loss':initial_value,'best_loss':best_loss,'ended':'normal'})
            save(O/'progress.json',{'state':'running','pid':os.getpid(),'slot':slot,'stages':stages,'evaluations':eval_count,'seconds':time.monotonic()-view_start,'training_steps':0})
        try:
            cam_loss=parts[2]['SMPLifyCameraInitLoss'](init_joints_idxs=torch.tensor([9,12,2,5]),trans_estimation=ref.body.transl.detach().clone(),data_weight=1000/row['height'] if 'height' in row else 1000/row['wh'][1],depth_loss_weight=100).to(DEVICE)
            one_stage('root_init',['global_orient','transl'],lambda out:cam_loss(out,ref.camera,gt),20,30)
            for stage in range(5):
                jw=ref.weights(stage,row['wh'][1])
                one_stage(f'official_weights_{stage}',list(ref.param),lambda out:ref.objective(out,gt,conf,jw),30,45)
        except BudgetEnd as exc:status='budget_ended';stop=str(exc)
        except Exception as exc:status='failed';stop=f'{type(exc).__name__}: {exc}';traceback.print_exc()
        torch.cuda.synchronize()
        with torch.no_grad():out=ref.forward();uv=ref.camera(out.joints).cpu().numpy();vertices=out.vertices.cpu().numpy();pose=ref.vposer.decode(ref.latent)['pose_body'].cpu().numpy()
        gt_np=gt.cpu().numpy();conf_np=conf.cpu().numpy()
        state={k:v.detach().cpu().numpy() for k,v in ref.param.items()}
        np.savez_compressed(O/f'slot{slot:03d}-fit.npz',**state,body_pose=pose,vertices_camera=vertices,joints_camera=out.joints.cpu().numpy(),projected=uv,initial_projected=initial_uv,initial_vertices_camera=initial_vertices,targets=gt_np,confidence=conf_np,K_integer_centers=K,faces=ref.body.faces)
        result={'slot':slot,'source_frame_index':row['source_frame_index'],'state':status,'seconds':time.monotonic()-view_start,'evaluations':eval_count,'stages':stages,'initial_vposer_projected_errors_px':errors(initial_uv,gt_np,conf_np),'final_detector_fit_errors_px':errors(uv,gt_np,conf_np),'parameter_delta_max':{k:float((v-initial[k]).abs().max()) for k,v in ref.param.items()},'all_K_values_fixed':bool(np.array_equal(K,ref.camera.K.detach().cpu().numpy().astype(float))) if K.dtype==np.float32 else bool(np.allclose(K,ref.camera.K.detach().cpu().numpy(),atol=1e-4,rtol=0)),'training_steps':0,'independent_physical_accuracy':False,'self_collision_tested':False,'unknown_visibility_not_gt':True}
        if status!='complete':result['stop']=stop
        save(O/f'slot{slot:03d}-result.json',result);results.append(result);print(json.dumps(result),flush=True)
        if status=='failed':break
    final={'created':now(),'state':'three_view_reference_complete' if len(results)==3 and all(x['state']=='complete' for x in results) else 'reference_stopped_or_budget_ended','views':results,'adapter':'Pinned SMPLify-X loss/priors/mapping with local full-K/RTMW/VPoser2/fresh-GVHMR wrapper','official_full_reproduction':False,'same_config_across_views':True,'joint_world_optimization':False,'source_sha256':sha(__file__),'gaussian_steps':0,'physical_quality_passed':False,'goal_complete':False}
    save(O/'result.json',final);save(O/'progress.json',{'state':final['state'],'pid':None,'finished':now(),'training_steps':0})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['controls','run']);args=ap.parse_args()
    lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        controls() if args.mode=='controls' else run()
    except Exception:
        save(O/f'{args.mode}-failure.json',{'created':now(),'traceback':traceback.format_exc(),'pid':os.getpid(),'gaussian_steps':0});raise
