"""Read-only intervention: bypass the trained fine layer, retain all other trained state."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np,torch
from enerf_second_temporal_graph import GraphMotion
from enerf_hand_joint_model import HandJointRenderer
from enerf_hand_fine_model import HandFineGraph,HandFineRenderer
from enerf_hand_flow_model import project
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');S=L/'outputs/dynamic-human-enerf-second-interval-079';O=L/'outputs/dynamic-human-enerf-hand-fine-085';D=O/'fine-effect';D.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda')
assert not (D/'summary.json').exists();torch.set_num_threads(8)
contexts=[r for r in read(S/'input-audit.json')['records'] if r['split']=='train'];assert len(contexts)==105
inputs={}
for r in contexts:
    z=np.load(S/'native-input'/f"camera{r['camera']}-frame{r['frame']:06d}.npz");inputs[r['camera'],r['frame']]=dict(K=tensor(z['K']),W=tensor(z['w2c']))
def stats(x):
    a=np.asarray(x,dtype=np.float64).ravel();assert np.isfinite(a).all()
    return dict(count=len(a),mean=float(a.mean()),median=float(np.median(a)),p95=float(np.quantile(a,.95)),maximum=float(a.max()))
result={}
with torch.no_grad():
    for branch in ['fine_photo','fine_flow']:
        cpfile=O/branch/'resume.pt';before=sha(cpfile);cp=torch.load(cpfile,map_location='cpu',weights_only=False);p={k:v.cuda() for k,v in cp['params'].items()};active=cp['active_Gaussians'].cuda();g=GraphMotion(S/'graph.npz',cp['scene_scale']);g.load_state_dict(cp['graph']);ref=GraphMotion(S/'graph.npz',cp['scene_scale']);ref.load_state_dict(cp['reference_graph']);ref.requires_grad_(False);fine=HandFineGraph(O/'fine-graph.npz',cp['scene_scale']);fine.load_state_dict(cp['fine_graph']);model=HandFineRenderer(S/'head/capture-affines.npz',g,ref,active,fine);ids=model.ids;rows=[];allpix=[];allworld=[];angle=[];translation=[]
        for (c,t),o in inputs.items():
            coarse,C0,B0,_=HandJointRenderer.geometry(model,p,c,t);full,C1,B1,_=model.geometry(p,c,t);assert torch.equal(full[~active],coarse[~active]) and torch.equal(C1[~active],C0[~active]);uv0,z0=project(coarse[ids],o['W'],o['K']);uv1,z1=project(full[ids],o['W'],o['K']);ok=(z0>.1)&(z1>.1);pix=(uv1-uv0).norm(dim=-1)[ok].cpu().numpy();world=(full[ids]-coarse[ids]).norm(dim=-1).cpu().numpy();centers,R,d=model.fine_state(c,t);rot=torch.acos(((R.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1)).cpu().numpy()*180/np.pi;disp=d.norm(dim=-1).cpu().numpy();allpix.extend(pix);allworld.extend(world);angle.extend(rot);translation.extend(disp)
            rows.append(dict(camera=c,frame=t,projected_displacement_px=stats(pix),world_displacement=stats(world),node_rotation_degrees=stats(rot),node_translation=stats(disp),covariance_relative_change=float((C1[ids]-C0[ids]).norm(dim=(-1,-2)).div(C0[ids].norm(dim=(-1,-2)).clamp_min(1e-12)).mean())))
        arrays={k:np.asarray(v) for k,v in dict(projected_displacement=allpix,world_displacement=allworld,node_rotation=angle,node_translation=translation).items()};np.savez_compressed(D/f'{branch}-displacements.npz',**arrays);assert sha(cpfile)==before
        result[branch]=dict(checkpoint_sha256=before,contexts=105,selected_Gaussians=6330,inactive_geometry_exact=True,projected_displacement_px=stats(allpix),world_displacement=stats(allworld),node_rotation_degrees=stats(angle),node_translation=stats(translation),records=rows)
        del model,p,g,ref,fine,cp;torch.cuda.empty_cache()
save(D/'summary.json',dict(status='trained_fine_layer_effect_measured_in105original_training_contexts',branches=result,scope='Counterfactual bypass of fine layer in trained checkpoint; no retraining. All selected centers equally counted, including hidden/off-surface points. Not image error, surface accuracy or causal attribution. Same-scene development.',goal_complete=False));shutil.copy2(__file__,D/Path(__file__).name)
print(json.dumps({b:{k:v for k,v in r.items() if k!='records'} for b,r in result.items()}))
