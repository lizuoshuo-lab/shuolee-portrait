"""Compare existing full-person and hand-context photometric gradient scales.

Read-only at the trained parameters; no new objective is adopted or optimized.
"""
from pathlib import Path
import json,hashlib,shutil
import numpy as np,torch
from train_enerf_camera10_time import ss
from enerf_second_temporal_graph import GraphMotion
from enerf_hand_joint_model import HandJointRenderer
from enerf_mano_gaussian_surface import ManoGaussianSurface,render_hand_surface
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');S=L/'outputs/dynamic-human-enerf-second-interval-079';H=L/'outputs/dynamic-human-hand-consensus-097';P=L/'outputs/dynamic-human-enerf-hand-visibility-100';O=L/'outputs/dynamic-human-hand-body-alignment-101/gradient-scale';O.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();tt=lambda a:torch.as_tensor(a,dtype=torch.float32,device='cuda')
assert not (O/'summary.json').exists();torch.set_num_threads(4);path=P/'context_reset/resume.pt';before=sha(path);cp=torch.load(path,map_location='cpu',weights_only=False);p={k:v.cuda() for k,v in cp['params'].items()};act=torch.from_numpy(np.load(P/'context-reset-mask.npy')).cuda();g=GraphMotion(S/'graph.npz',cp['scene_scale']);g.load_state_dict(cp['graph']);g.requires_grad_(False);old=HandJointRenderer(S/'head/capture-affines.npz',g,g,act)
hand=ManoGaussianSurface(H/'all_views/parameters.npz',S/'heldout/capture-affines-estimated.npz',cp['scene_scale'],read(P/'protocol.json')['initial_color'],True);hand.load_state_dict(cp['hand_surface'])
obs={(r['camera'],r['frame']):r for r in read(L/'outputs/dynamic-human-enerf-hand-motion-082/observations.json')['records']};names=['translation_delta','pose_delta','color_logits','opacity_logits'];params=[getattr(hand,k) for k in names];records=[]
save(O/'protocol.json',dict(status='fixed_9training_context_gradient_scale_check_no_optimizer',cameras=['03','04','13'],frames=[720,726,732],background=[.5,.5,.5],source_checkpoint_sha256=before,loss='Same0.8L1+0.2DSSIM+0.1alpha loss as training. Full-person crop versus existing160x160hand-context crop from the same native render; unchanged regularization. Handcontext includes forearm andtoy, not semanticGT.',goal_complete=False))
def photometric(rgb,alpha,target,matte,valid):return .8*((rgb-target)*valid).abs().mean()+.2*(1-ss((rgb*valid).permute(2,0,1)[None],(target*valid).permute(2,0,1)[None]))+.1*((alpha-matte).abs()*valid).mean()
for c in ['03','04','13']:
 for f in [720,726,732]:
  z=np.load(S/'native-input'/f'camera{c}-frame{f:06d}.npz');x0,y0,x1,y1=z['bbox'];K=z['K'].copy();K[0,2]-=x0;K[1,2]-=y0;W=tt(z['w2c']);rgb,alpha,_=render_hand_surface(p,old,hand,c,f,W,tt(K),int(x1-x0),int(y1-y0));pred=rgb[0]+.5*(1-alpha[0]);mat=tt(z['matte'][y0:y1,x0:x1])[...,None];valid=tt(z['valid'][y0:y1,x0:x1])[...,None];target=tt(z['rgb'][y0:y1,x0:x1]/255)*mat+.5*(1-mat)
  a,b,d,e=obs[c,f]['crop'];a,b,d,e=int(a-x0),int(b-y0),int(d-x0),int(e-y0);assert a>=0 and b>=0 and d<=x1-x0 and e<=y1-y0 and (d-a,e-b)==(160,160)
  losses=dict(full_person=photometric(pred,alpha[0],target,mat,valid),hand_context=photometric(pred[b:e,a:d],alpha[0,b:e,a:d],target[b:e,a:d],mat[b:e,a:d],valid[b:e,a:d]),regularizer=hand.regularization());grad={}
  for name,loss in losses.items():
   gg=torch.autograd.grad(loss,params,retain_graph=name!='regularizer',allow_unused=True);grad[name]=[torch.zeros_like(p) if v is None else v for p,v in zip(params,gg)]
  metrics={}
  for k,n in enumerate(names):
   norm={a:float(grad[a][k].norm()) for a in grad};u=grad['full_person'][k].flatten();v=grad['regularizer'][k].flatten();metrics[n]=dict(norms=norm,context_to_full_ratio=norm['hand_context']/max(norm['full_person'],1e-20),regularizer_to_full_ratio=norm['regularizer']/max(norm['full_person'],1e-20),full_regularizer_cosine=float(torch.dot(u,v)/(u.norm()*v.norm()).clamp_min(1e-20)))
  row=dict(camera=c,frame=f,person_crop_pixels=int((x1-x0)*(y1-y0)),hand_crop_pixels=25600,losses={k:float(v.detach()) for k,v in losses.items()},gradient_metrics=metrics);records.append(row);print('GRADIENT_SCALE',json.dumps(row),flush=True)
  del rgb,alpha,pred,target,mat,valid,losses,grad,gg
assert sha(path)==before
summary=dict(status='9training_contexts_gradient_scales_measured_no_parameter_updates',records=records,
    median_ratios={n:{key:float(np.median([r['gradient_metrics'][n][key] for r in records])) for key in ['context_to_full_ratio','regularizer_to_full_ratio','full_regularizer_cosine']} for n in names},
    scope='At one trained checkpoint and graybackground, localcontext changes image normalization andSSIMwindow boundaries; no optimizer step or empirical improvement established. Not all105training photosorbackgrounds.',source_checkpoint_unchanged=True,goal_complete=False)
save(O/'summary.json',summary);shutil.copy2(__file__,O/Path(__file__).name);print('GRADIENT_AGGREGATE',json.dumps(summary['median_ratios']),flush=True)
