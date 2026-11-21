"""Bounded pose-only native photometric fit; every Gaussian attribute stays fixed."""
from pathlib import Path
import argparse,json,hashlib,time,os,shutil
import numpy as np,torch
from gsplat.rendering import rasterization
from gsplat.cuda._wrapper import spherical_harmonics
from train_enerf_camera10_time import ss
from enerf_second_temporal_graph import GraphMotion
from enerf_hand_joint_model import HandJointRenderer
from enerf_mano_gaussian_surface import ManoGaussianSurface,render_hand_surface
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');S=L/'outputs/dynamic-human-enerf-second-interval-079';Q=L/'outputs/dynamic-human-enerf-contour-surface-103';O=L/'outputs/dynamic-human-hand-photo-pose-105';O.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();tt=lambda a:torch.as_tensor(a,dtype=torch.float32,device='cuda')
ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['prepare','fit','evaluate'],required=True);args=ap.parse_args();torch.set_num_threads(6);torch.manual_seed(10501)
source=Q/'contour_prior/resume.pt';before=sha(source);cp=torch.load(source,map_location='cpu',weights_only=False);qp=read(Q/'protocol.json');assert before=='ebae3e81c6bc511d7ab306939583e1ba67a0b2f65b5c1058edb08a139ae2e94c'
p={k:v.cuda() for k,v in cp['params'].items()};g=GraphMotion(S/'graph.npz',cp['scene_scale']);g.load_state_dict(cp['graph']);g.requires_grad_(False);active=torch.from_numpy(np.load(Q/'context-reset-mask.npy')).cuda();old=HandJointRenderer(S/'heldout/capture-affines-estimated.npz',g,g,active)
hand=ManoGaussianSurface(L/qp['priors']['contour_prior']['path'],S/'heldout/capture-affines-estimated.npz',cp['scene_scale'],qp['initial_color'],True);hand.load_state_dict(cp['hand_surface']);hand.requires_grad_(False)
for k in ['pose_delta','translation_delta']:getattr(hand,k).requires_grad_(True)
if args.mode=='evaluate':
    from PIL import Image
    R=O/'native-evaluation';R.mkdir(exist_ok=True);assert not (R/'metrics.json').exists();outpath=O/'resume.pt';digest=sha(outpath);q=torch.load(outpath,map_location='cpu',weights_only=False);assert q['renderer']=='mano_photo_pose105';assert all(torch.equal(v,q['params'][k]) for k,v in cp['params'].items());assert all(torch.equal(v,q['graph'][k]) for k,v in cp['graph'].items())
    for k,v in cp['hand_surface'].items():
        if k not in ['pose_delta','translation_delta']:assert torch.equal(v,q['hand_surface'][k]),k
    hand.load_state_dict(q['hand_surface']);hand.requires_grad_(False);records=read(L/'outputs/dynamic-human-enerf-hand-time-scope-088/evaluation-protocol.json')['records'];rows=[];U=L/'outputs/dynamic-human-enerf-hand-time-scope-088';u8=lambda a:np.rint(np.clip(a,0,1)*255).astype(np.uint8)
    with torch.no_grad():
        for r in records:
            c,f=r['camera'],r['frame'];stem=f'camera{c}-frame{f:06d}';path=(U if r.get('input_location')=='088' else S)/'native-input'/f'{stem}.npz';assert sha(path)==r['input_sha256'];z=np.load(path);raw,alpha,_=render_hand_surface(p,old,hand,c,f,tt(z['w2c']),tt(z['K']),1920,1080,emit_surface_weight=True);rgb=raw[0,:,:,:3].cpu().numpy();weight=raw[0,:,:,3].cpu().numpy();a=alpha[0,:,:,0].cpu().numpy();assert np.isfinite(rgb).all() and np.isfinite(a).all();pred=np.clip(rgb+.5*(1-a[...,None]),0,1);target=z['rgb'].astype(np.float32)/255*z['matte'][...,None]+.5*(1-z['matte'][...,None]);x0,y0,x1,y1=z['bbox'];bb=np.zeros_like(z['valid']);bb[y0:y1,x0:x1]=True;metrics={}
            for name,m in {'full_valid':z['valid'],'foreground':z['valid']&(z['matte']>.5),'foreground_bbox':z['valid']&bb}.items():
                d=pred.astype(float)[m]-target.astype(float)[m];metrics[name]=dict(pixels=int(m.sum()),psnr_db=float(-10*np.log10(np.mean(d*d))),mae=float(abs(d).mean()))
            np.savez_compressed(R/f'{stem}-prediction.npz',rgb=rgb,alpha=a,surface_weight=weight);Image.fromarray(u8(pred)).save(R/f'{stem}-trained.png');Image.fromarray(u8(a)).save(R/f'{stem}-alpha.png');rows.append(dict(camera=c,frame=f,split=r['split'],metrics=metrics,input_sha256=r['input_sha256'],bbox=r['bbox'],reused079parent_prediction=False));save(R/'progress.json',dict(images=len(rows)));print('POSE105_EVAL',c,f,flush=True)
    assert len(rows)==134 and sha(source)==before and sha(outpath)==digest;save(R/'metrics.json',dict(status='134native_evaluations_complete_comparison_required',records=rows,checkpoint_sha256=digest,all_Gaussian_attributes_and_old_motion_exact=True,goal_complete=False));exit()

rows=[r for r in read(L/'outputs/dynamic-human-hand-prior-096/input-manifest.json')['records']];assert len(rows)==105 and all(r['frame']%2==0 for r in rows) and not {r['camera'] for r in rows}&{'00','06','12'}
protocol=dict(status='pose_only_photometric_control',source_checkpoint_sha256=before,train_photos=105,context_pixels=[160,160],steps=60,optimizer='L-BFGS max_iter1 history10 strong_wolfe; full105-photo average at every closure',objective='0.8 native masked RGB L1 +0.2 DSSIM +0.1 matte L1, fixedgray0.5background; mean across105crops. Plus0.001 mean boundedpose_delta^2 +0.001 mean normalizedwrist_delta^2.',parameters='Only9x16x3 continuous pose corrections and9x3 wrist corrections. Original and new Gaussian offsets/scales/colors/opacities, shape/scale/time/cameras fixed.',bounds='Each axis-angle coefficient component +=0.25*tanh(raw) rad; each wrist coefficient component +=0.01*scene_scale*tanh(raw). These are coefficient bounds, not exact geodesic angle bounds.',comparison='103contour_prior is the unchanged starting control. Extra optimization and local objective differ; this is a pose-only mechanism test, not matched total training cost.',selection='Lowest average training objective only; no heldoutimage selection.',scope='Repeated same-scene development. No finger groundtruth; all105original photos unchanged. No opacity update. Not a newscene or fullbackground result.',goal_complete=False,rendering='Full1920x1080 rasterization followed by native160x160loss slice; avoids crop tile-layout changes.')
if (O/'protocol.json').exists():assert read(O/'protocol.json')==protocol
else:save(O/'protocol.json',protocol)
cache=[];start=time.monotonic();aold=p['opacities'].sigmoid()
with torch.no_grad():
    for r in ([r for r in rows if r['camera'] in ['03','04','13'] and r['frame']==732] if args.mode=='prepare' else rows):
        c,f=r['camera'],r['frame'];path=L/r['source'].replace('\\','/');assert sha(path)==r['source_sha256'];z=np.load(path);x0,y0,x1,y1=map(int,r['context_bbox']);K=z['K'].copy();W=tt(z['w2c']);mu,cov,linear,meta=old.geometry(p,c,f);center=-W[:3,:3].T@W[:3,3];directions=(old.base.inverse[meta[0],p['anchor']]@torch.linalg.solve(linear,(mu-center)[...,None]))[...,0];directions=torch.nn.functional.normalize(directions,dim=-1);color=(spherical_harmonics(2,directions,torch.cat([p['sh0'],p['shN']],1))+.5).clamp_min(0);matte=tt(z['matte'][y0:y1,x0:x1])[...,None];valid=tt(z['valid'][y0:y1,x0:x1])[...,None];rgb=tt(z['rgb'][y0:y1,x0:x1])/255;cache.append(dict(camera=c,frame=f,mu=mu,cov=cov,color=color,W=W,K=tt(K),matte=matte,valid=valid,target=(rgb*matte+.5*(1-matte))*valid,bbox=[x0,y0,x1,y1]))
        if len(cache)%21==0:print('POSE105_CACHE',len(cache),flush=True)
def render(o):
    hm,hc,ha,hcolor,_=hand.geometry(o['camera'],o['frame']);return rasterization(torch.cat([o['mu'],hm]),None,None,torch.cat([aold,ha]),torch.cat([o['color'],hcolor]),o['W'][None],o['K'][None],1920,1080,covars=torch.cat([o['cov'],hc]),packed=False,near_plane=.1,far_plane=20,eps2d=.3)
def photoloss(o):
    rgb,alpha,_=render(o);x0,y0,x1,y1=o['bbox'];rgb=rgb[:,y0:y1,x0:x1];alpha=alpha[:,y0:y1,x0:x1];pred=(rgb[0]+.5*(1-alpha[0]))*o['valid'];return .8*(pred-o['target']).abs().mean()+.2*(1-ss(pred.permute(2,0,1)[None],o['target'].permute(2,0,1)[None]))+.1*((alpha[0]-o['matte']).abs()*o['valid']).mean()
def regularizer():return .001*(.25*hand.pose_delta.tanh()).square().mean()+.001*hand.translation_delta.tanh().square().mean()
if args.mode=='prepare':
    assert not (O/'native-full-canvas-smoke.json').exists();checks=[]
    with torch.no_grad():
        for c in ['03','04','13']:
            o=next(o for o in cache if o['camera']==c and o['frame']==732);raw,alpha,_=render(o);z=np.load(Q/'contour_prior/native-evaluation'/f'camera{c}-frame000732-prediction.npz');x0,y0,x1,y1=o['bbox'];uncached=render_hand_surface(p,old,hand,c,732,o['W'],o['K'],1920,1080);cache_error=max(float((raw-uncached[0]).abs().max()),float((alpha-uncached[1]).abs().max()));assert cache_error<2e-6,cache_error
            differences=np.concatenate([abs(raw[0,y0:y1,x0:x1].cpu().numpy()-z['rgb'][y0:y1,x0:x1]).ravel(),abs(alpha[0,y0:y1,x0:x1,0].cpu().numpy()-z['alpha'][y0:y1,x0:x1]).ravel()]);checks.append(dict(camera=c,frame=732,cached_vs_uncached_max_difference=cache_error,full_vs_crop_max_difference=float(differences.max()),full_vs_crop_mean_difference=float(differences.mean()),full_vs_crop_p999_difference=float(np.quantile(differences,.999)),channels_above_one_uint8_level=int((differences>1/255).sum())))
    o=next(o for o in cache if o['camera']=='04' and o['frame']==732);loss=photoloss(o);grad=torch.autograd.grad(loss,[hand.pose_delta,hand.translation_delta]);assert all(torch.isfinite(v).all() and float(v.abs().max())>0 for v in grad)
    ready=all(r['full_vs_crop_max_difference']<2e-6 for r in checks)
    save(O/'native-full-canvas-smoke.json',dict(status='cached_uncached_full_canvas_and_source_checkpoint_reproduction_verified',checks=checks,gradient_norms=[float(v.norm()) for v in grad],trainable_parameters=[n for n,v in hand.named_parameters() if v.requires_grad],source_checkpoint_unchanged=sha(source)==before,no_optimizer_steps=True,ready_for_fit=ready,scope='Full canvas rendering then loss slicing. All3source checkpoint RGB/alpha reproductions must agree within2e-6. Earlier direct-crop differences preserved in crop-diagnostic.json; no optimizer steps used that path.',goal_complete=False));assert ready,checks;print('POSE105_PREFLIGHT_PASSED',flush=True);exit()
assert read(O/'native-full-canvas-smoke.json')['no_optimizer_steps'] and read(O/'native-full-canvas-smoke.json')['ready_for_fit'];assert not (O/'completion.json').exists()
params=[hand.pose_delta,hand.translation_delta];opt=torch.optim.LBFGS(params,lr=1,max_iter=1,max_eval=12,history_size=10,line_search_fn='strong_wolfe',tolerance_grad=1e-8,tolerance_change=1e-10);history=[];closures=0;begin_step=0
if (O/'resume.pt').exists():
    q=torch.load(O/'resume.pt',map_location='cpu',weights_only=False);assert q['renderer']=='mano_photo_pose105' and q['photo_pose_source_sha256']==before;hand.load_state_dict(q['hand_surface']);opt.load_state_dict(q['photo_pose_optimizer']);history=q['photo_pose_history'];closures=q['photo_pose_closures'];begin_step=q['photo_pose_updates']
def closure():
    global closures
    opt.zero_grad(set_to_none=True);total=0.
    for o in cache:
        value=photoloss(o)/len(cache);assert torch.isfinite(value);value.backward();total+=float(value.detach())
    reg=regularizer();reg.backward();assert all(torch.isfinite(v.grad).all() for v in params);closures+=1;return torch.tensor(total,device='cuda')+reg.detach()
@torch.no_grad()
def objective():return float(sum(float(photoloss(o))/len(cache) for o in cache)+regularizer())
if not history:history.append(dict(step=0,objective=objective(),closures=closures,elapsed_seconds=time.monotonic()-start))
def checkpoint(step):
    q=dict(cp);q.update(renderer='mano_photo_pose105',hand_surface={k:v.detach().cpu() for k,v in hand.state_dict().items()},photo_pose_updates=step,photo_pose_source_sha256=before,photo_pose_optimizer=opt.state_dict(),photo_pose_history=history,photo_pose_closures=closures);torch.save(q,O/'resume.partial');os.replace(O/'resume.partial',O/'resume.pt')
for step in range(begin_step+1,61):
    prev=history[-1]['objective'];opt.step(closure);value=objective();assert np.isfinite(value) and value<=prev+2e-6,(prev,value);row=dict(step=step,objective=value,closures=closures,max_pose_coefficient_component_delta_rad=float((.25*hand.pose_delta.tanh()).abs().max()),max_wrist_coefficient_component_delta_world=float((.01*cp['scene_scale']*hand.translation_delta.tanh()).abs().max()),elapsed_seconds=time.monotonic()-start);history.append(row);save(O/'progress.json',row);print('POSE105_STEP',json.dumps(row),flush=True)
    if step%10==0:checkpoint(step)
assert sha(source)==before
for k,v in cp['hand_surface'].items():
    if k not in ['pose_delta','translation_delta']:assert torch.equal(v,hand.state_dict()[k].cpu()),k
save(O/'completion.json',dict(status='60fullbatch_pose_only_updates_complete_native_evaluation_required',updates=60,full_training_observations_each=105,closures=closures,initial_objective=history[0]['objective'],final_objective=history[-1]['objective'],max_pose_coefficient_component_delta_rad=history[-1]['max_pose_coefficient_component_delta_rad'],max_wrist_coefficient_component_delta_world=history[-1]['max_wrist_coefficient_component_delta_world'],checkpoint_sha256=sha(O/'resume.pt'),source_checkpoint_unchanged=True,all_Gaussian_attributes_and_old_motion_exact=True,goal_complete=False));save(O/'history.json',history);shutil.copy2(__file__,O/Path(__file__).name);print('POSE105_TRAINING_COMPLETE',flush=True)
