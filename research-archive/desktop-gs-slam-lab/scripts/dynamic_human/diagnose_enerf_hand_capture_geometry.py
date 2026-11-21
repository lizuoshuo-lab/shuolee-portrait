"""Read-only: motion-induced projection change across camera capture times at frame732."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np,torch
from gsplat.rendering import rasterization
from enerf_second_temporal_graph import GraphMotion
from enerf_hand_joint_model import HandJointRenderer
from enerf_hand_flow_model import project
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');S=L/'outputs/dynamic-human-enerf-second-interval-079';U=L/'outputs/dynamic-human-enerf-hand-time-scope-088';O=L/'outputs/dynamic-human-enerf-hand-capture-geometry-090';O.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'));save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda')
torch.set_num_threads(8);path=S/'local_bspline/resume.pt';before=sha(path);cp=torch.load(path,map_location='cpu',weights_only=False);p={k:v.cuda() for k,v in cp['params'].items()};active=torch.from_numpy(np.load(L/'outputs/dynamic-human-enerf-hand-joint-084/support/active-gaussians.npy')).cuda();g=GraphMotion(S/'graph.npz',cp['scene_scale']);g.load_state_dict(cp['graph']);g.requires_grad_(False);model=HandJointRenderer(S/'heldout/capture-affines-estimated.npz',g,g,active);ids=torch.where(active)[0];records=[r for r in read(U/'evaluation-protocol.json')['records'] if r['frame']==732];crops={r['camera']:r['crop'] for r in read(L/'outputs/dynamic-human-enerf-hand-motion-082/observations.json')['records'] if r['frame']==732};rows=[];arrays={};quant=lambda x:{str(q):float(np.quantile(x,q)) for q in [.05,.5,.95,.99]}
with torch.no_grad():
 ref=model.positions(p,'04',732)[ids];arrays['reference_positions']=ref.cpu().numpy()
 for r in sorted(records,key=lambda r:r['camera']):
  c=r['camera'];inp=(U if r.get('input_location')=='088' else S)/'native-input'/f'camera{c}-frame000732.npz';assert sha(inp)==r['input_sha256'];z=np.load(inp);W=tensor(z['w2c']);K=tensor(z['K']);m,C,_,_=model.geometry(p,c,732);uv,depth=project(m[ids],W,K);uvref,_=project(ref,W,K);delta=uvref-uv;norm=delta.norm(dim=-1);world=(m[ids]-ref).norm(dim=-1);values=torch.zeros(len(m),4,device='cuda');values[ids,:2]=delta;values[ids,2]=norm;values[ids,3]=1
  mapped,alpha,_=rasterization(m,None,None,p['opacities'].sigmoid(),values,W[None],K[None],1920,1080,covars=C,packed=False,near_plane=.1,far_plane=20,eps2d=.3);v=mapped[0].cpu().numpy();visible=None
  if c in crops:
   x0,y0,x1,y1=crops[c];vm=v[y0:y1,x0:x1];mask=z['valid'][y0:y1,x0:x1]&(z['matte'][y0:y1,x0:x1]>.5)&(vm[...,3]>.05);field=vm[...,:3]/np.maximum(vm[...,3:],1e-8);visible=dict(pixels=int(mask.sum()),mean_absolute_projected_shift_px=float(np.mean(field[...,2][mask])),quantiles_absolute_shift_px=quant(field[...,2][mask]),mean_signed_shift_px=field[...,:2][mask].mean(0).tolist(),active_contribution_alpha_mean=float(vm[...,3][mask].mean()))
   np.savez_compressed(O/f'camera{c}-visible-shift.npz',numerator=vm,mask=mask)
  arrays[f'positions_{c}']=m[ids].cpu().numpy();arrays[f'W_{c}']=z['w2c'];arrays[f'K_{c}']=z['K'];arrays[f'projected_delta_{c}']=delta.cpu().numpy()
  i=model.base.lookup[c,732];ir=model.base.lookup['04',732];rows.append(dict(camera=c,frame=732,effective_capture_index=float(model.times[i]),offset_vs_camera04_frames=float(model.times[i]-model.times[ir]),all_selected_world_shift_units=quant(world.cpu().numpy()),all_selected_projected_shift_px=quant(norm.cpu().numpy()),visible_hand_context=visible,input_sha256=r['input_sha256']));print('CAPTURE_GEOMETRY',json.dumps(rows[-1]),flush=True)
np.savez_compressed(O/'selected-geometry.npz',**arrays);assert sha(path)==before
save(O/'summary.json',dict(status='18capturetime_geometry_projections_complete_readonly',reference='camera04 at nominalframe732; identical targetW/K for each comparison',checkpoint_sha256=before,records=rows,scope='Model-implied displacement between camera capture times; not measured correspondence or error. Includes hidden/offsurface selected centers. Visible statistic is original-depth alpha-composited weighted norm; not norm of mean flow. Originalhandcontextincludesothermaterials. No clocks,cameras,motion ormodel changed.',goal_complete=False));shutil.copy2(__file__,O/Path(__file__).name)
