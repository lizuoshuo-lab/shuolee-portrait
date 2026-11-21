"""Read-only native actor-contribution/depth diagnosis of frozen graph model."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np,torch,cv2
from PIL import Image,ImageDraw
import enerf_graph_renderer as module
from enerf_graph_renderer import GraphMotion,ResidualGraphRenderer

L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
R=L/'outputs/dynamic-human-enerf-actor-layers-069'
I=L/'outputs/dynamic-human-enerf-heldout-cameras-068'
G=L/'outputs/dynamic-human-enerf-residual-graph-067'
read=lambda p:json.loads(p.read_text())
save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
u8=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8)
tensor=lambda x:torch.tensor(x,dtype=torch.float32,device='cuda')
R.mkdir(parents=True,exist_ok=True)
assert not (R/'render-summary.json').exists()
torch.set_num_threads(8)
checkpoint=G/'residual_graph/resume.pt';checkpoint_sha=sha(checkpoint)
assert checkpoint_sha==read(I/'protocol.json')['checkpoints']['graph']
contexts=[(c,t) for c in ['00','06','12'] for t in [745,751,755]]
save(R/'protocol.json',dict(status='fixed_before_layer_renders',checkpoint_sha256=checkpoint_sha,
 contexts=contexts,model_updates=0,actor_assignment='Frozen nearest-body-vertex anchor divided by6890, not instance ground truth.',
 diagnosis='Per-Gaussian compositing with original joint depth sorting; isolated actors are interventions, not observed ground truth.',
 thresholds=dict(core_matte=.95,core_erosion_pixels=5,low_alpha=.9,rgb_mae=.08,overlap_isolated_alpha=.5,mixed_contribution_fraction=[.1,.9],depth_display_world_units=.25),goal_complete=False))
cp=torch.load(checkpoint,map_location='cpu',weights_only=False);assert cp['step']==15000
p={k:v.cuda() for k,v in cp['params'].items()}
graph=GraphMotion(G/'graph.npz',cp['scene_scale']);graph.load_state_dict(cp['graph']);graph.eval()
motion=I/'capture-affines-estimated.npz'
assert sha(motion)==read(I/'motion-extension-audit.json')['branches']['estimated']['sha256']
model=ResidualGraphRenderer(motion,graph);del cp
actor=p['anchor']//6890;assert set(actor.unique().tolist())=={0,1}
onehot=torch.nn.functional.one_hot(actor,2).float()
original_raster=module.rasterization;capture={}
def capture_raster(*args,**kwargs):
 capture['args']=args;capture['kwargs']=kwargs
 return original_raster(*args,**kwargs)
module.rasterization=capture_raster

def panel(path,items,box):
 x0,y0,x1,y1=map(int,box);w=x1-x0;h=y1-y0
 im=Image.new('RGB',(w*len(items),h+36),'white');draw=ImageDraw.Draw(im)
 for i,(label,arr) in enumerate(items):
  if arr.ndim==2:arr=np.repeat(arr[...,None],3,axis=-1)
  im.paste(Image.fromarray(arr[y0:y1,x0:x1]),(i*w,36));draw.text((i*w+4,12),label,fill='black')
 im.save(path)

rows=[]
with torch.no_grad():
 for c,t in contexts:
  name=f'camera{c}-frame{t:06d}';z=np.load(I/'native-input'/f'{name}.npz')
  rgb,a,_=model.render(p,c,t,tensor(z['w2c']),tensor(z['K']),1920,1080,2)
  args=capture['args'];kw=capture['kwargs'];means=args[0];colors=args[4]
  depth=(means@args[5][0,:3,:3].T+args[5][0,:3,3])[:,2]
  features=torch.cat([colors,colors*onehot[:,0:1],colors*onehot[:,1:2],onehot,onehot*depth[:,None]],-1)
  f,joint_alpha,_=original_raster(args[0],args[1],args[2],args[3],features,*args[5:],**kw)
  f=f[0].cpu().numpy();alpha=joint_alpha[0,...,0].cpu().numpy()
  rgb=rgb[0].cpu().numpy();a=a[0,...,0].cpu().numpy()
  saved=np.load(I/'graph_estimated/native-evaluation'/f'{name}-prediction.npz')
  error=dict(saved_rgb=float(abs(rgb-saved['rgb']).max()),saved_alpha=float(abs(a-saved['alpha']).max()),feature_rgb=float(abs(rgb-f[...,:3]).max()),
   feature_alpha=float(abs(a-alpha).max()),rgb_sum=float(abs(f[...,:3]-f[...,3:6]-f[...,6:9]).max()),alpha_sum=float(abs(alpha-f[...,9]-f[...,10]).max()))
  assert max(error.values())<5e-6,error
  isolated=[]
  for person in [0,1]:
   mask=actor==person;kwi={**kw,'covars':kw['covars'][mask]}
   fi,ai,_=original_raster(means[mask],None,None,args[3][mask],torch.cat([colors[mask],depth[mask,None]],-1),*args[5:],**kwi)
   isolated.append((fi[0].cpu().numpy(),ai[0,...,0].cpu().numpy()))
  premult=f[...,:3];pred=np.clip(premult+.5*(1-alpha[...,None]),0,1)
  contribution=f[...,9:11];sum_depth=f[...,11:13]
  expected=sum_depth/np.maximum(contribution,1e-8)
  isolated_rgb=np.stack([v[...,:3] for v,_ in isolated],axis=2)
  isolated_alpha=np.stack([a for _,a in isolated],-1)
  isolated_depth=np.stack([v[...,3]/np.maximum(a,1e-8) for v,a in isolated],-1)
  ref=z['rgb'].astype(np.float32)/255*z['matte'][...,None]+.5*(1-z['matte'][...,None])
  core=cv2.erode((z['matte']>.95).astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)&z['valid']
  rgb_error=np.abs(pred-ref).mean(-1)
  overlap=(isolated_alpha>.5).all(-1)&z['valid']
  fraction=contribution[...,0]/np.maximum(alpha,1e-8)
  mixed=overlap&(fraction>.1)&(fraction<.9)
  low=core&(alpha<.9);bad=core&(rgb_error>.08)
  dz=isolated_depth[...,0]-isolated_depth[...,1]
  stats=dict(core_pixels=int(core.sum()),core_alpha_below_point9=int(low.sum()),core_alpha_below_point9_fraction=float(low.sum()/core.sum()),
   core_mae_above_point08=int(bad.sum()),bad_core_pixels_with_alpha_atleast_point9=int((bad&~low).sum()),
   isolated_overlap_pixels=int(overlap.sum()),mixed_actor_pixels=int(mixed.sum()),
   isolated_overlap_depth_difference_world_units_percentiles=np.percentile(dz[overlap],[5,50,95]).tolist())
  np.savez_compressed(R/f'{name}-layers.npz',rgb=premult,alpha=alpha,actor_rgb=f[...,3:9].reshape(1080,1920,2,3),actor_alpha=contribution,
   actor_depth=expected,isolated_rgb=isolated_rgb,isolated_alpha=isolated_alpha,isolated_depth=isolated_depth)
  identity=contribution[...,0:1]*np.array([1.,.45,.05])+contribution[...,1:2]*np.array([.1,.55,1.])+.5*(1-alpha[...,None])
  depthmap=np.zeros((1080,1920,3),np.float32)+.5
  scale=np.clip(abs(dz)/.25,0,1);color=np.where((dz<0)[...,None],np.array([1.,.45,.05]),np.array([.1,.55,1.]))
  depthmap[overlap]=(.5*(1-scale[...,None])+color*scale[...,None])[overlap]
  panel(R/f'{name}-actors.png',[('Original photograph',z['rgb']),('Joint render',u8(pred)),('Actor0 rendered alone',u8(isolated_rgb[:,:,0]+.5*(1-isolated_alpha[...,0:1]))),('Actor1 rendered alone',u8(isolated_rgb[:,:,1]+.5*(1-isolated_alpha[...,1:2])))],z['bbox'])
  panel(R/f'{name}-layers.png',[('Joint alpha: black0 white1',u8(alpha)),('Joint: orange actor0 / blue actor1',u8(identity)),('Isolated depth: nearer actor; max0.25',u8(depthmap))],z['bbox'])
  regions=[r for r in read(I/'review-protocol.json')['regions'] if r['camera']==c and r['frame']==t]
  region_stats=[]
  for r in regions:
   x0,y0,x1,y1=r['bbox'];sl=(slice(y0,y1),slice(x0,x1));s=core[sl]
   region_stats.append(dict(name=r['name'],bbox=r['bbox'],core_pixels=int(s.sum()),mean_joint_actor_fraction=(contribution[sl][s]/np.maximum(alpha[sl][s,None],1e-8)).mean(0).tolist(),mean_alpha=float(alpha[sl][s].mean())))
  rows.append(dict(camera=c,frame=t,consistency_max_abs_error=error,statistics=stats,projected_region_statistics=region_stats))
  print('ACTOR_LAYERS',c,t,json.dumps(stats),flush=True)
assert sha(checkpoint)==checkpoint_sha
save(R/'render-summary.json',dict(status='9native_frozen_graph_actor_layer_diagnostics_complete',checkpoint_sha256=checkpoint_sha,model_updates=0,records=rows,
 qualification='Alpha/depth/identity are model predictions. Expected depth is a weighted Gaussian center depth, not a mesh surface or ground truth. Isolated-actor images reveal model ownership but hidden surfaces need not be reconstructable from these cameras.',goal_complete=False))
shutil.copy2(__file__,R/Path(__file__).name)
