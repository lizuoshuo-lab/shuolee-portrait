from pathlib import Path
import argparse,json,hashlib,shutil
import numpy as np,torch
from PIL import Image
from enerf_second_temporal_graph import GraphMotion
from enerf_hand_snapshot_model import HandSnapshotRenderer as HandJointRenderer
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');S=L/'outputs/dynamic-human-enerf-second-interval-079';J=L/'outputs/dynamic-human-enerf-hand-joint-084';U=L/'outputs/dynamic-human-enerf-hand-time-scope-088';O=L/'outputs/dynamic-human-enerf-hand-snapshot-091'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'));save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda');u8=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8)
ap=argparse.ArgumentParser();ap.add_argument('--branch',choices=['single_time'],required=True);args=ap.parse_args();b=args.branch;R=O/b;E=R/'native-evaluation';E.mkdir(parents=True,exist_ok=True);assert not (E/'metrics.json').exists();path=S/'local_bspline/resume.pt' if b=='parent' else R/'resume.pt';before=sha(path);cp=torch.load(path,map_location='cpu',weights_only=False);parent=torch.load(S/'local_bspline/resume.pt',map_location='cpu',weights_only=False);active=torch.from_numpy(np.load(J/'support/active-gaussians.npy'));changes={}
if b!='parent':
    assert read(R/'completion.json')['scope_steps']==3000 and cp['renderer']=='hand_shared_snapshot_091'
    for k,v in parent['params'].items():
        assert torch.equal(v[~active],cp['params'][k][~active])
        if k in ['anchor','node_indices','node_weights']:assert torch.equal(v,cp['params'][k])
        else:changes[k]=float((v[active]-cp['params'][k][active]).abs().max())
    assert all(torch.equal(v,cp['graph'][k]) for k,v in parent['graph'].items());assert all(int(v['step'])==3000 for o in cp['optim'].values() for v in o['state'].values());save(R/'frozen-state-audit.json',dict(checkpoint_sha256=before,nonhand_Gaussians=246654,all384motion_nodes_exact=True,allbindings_exact=True,all_optimizer_steps=3000,max_active_changes=changes,goal_complete=False))
torch.set_num_threads(8);p={k:v.cuda() for k,v in cp['params'].items()};g=GraphMotion(S/'graph.npz',cp['scene_scale']);g.load_state_dict(cp['graph']);g.requires_grad_(False);ref=GraphMotion(S/'graph.npz',cp['scene_scale']);ref.load_state_dict(parent['graph']);ref.requires_grad_(False);motion=S/'heldout/capture-affines-estimated.npz';assert sha(motion)==read(S/'heldout/motion-extension-audit.json')['branches']['estimated']['sha256'];model=HandJointRenderer(motion,g,ref,active.cuda());records=[r for r in read(U/'evaluation-protocol.json')['records'] if r['frame']==732];rows=[]
with torch.no_grad():
 for r in records:
    c,t=r['camera'],r['frame'];stem=f'camera{c}-frame{t:06d}';inp=(U if r.get('input_location')=='088' else S)/'native-input'/f'{stem}.npz';assert sha(inp)==r['input_sha256'];z=np.load(inp);old=S/'local_bspline/native-evaluation'/f'{stem}-prediction.npz';reuse=b=='parent' and old.exists()
    if reuse:
        q=np.load(old);rgb=q['rgb'];alpha=q['alpha']
    else:
        rgb,alpha,_=model.render(p,c,t,tensor(z['w2c']),tensor(z['K']),1920,1080,2);rgb=rgb[0].cpu().numpy();alpha=alpha[0,:,:,0].cpu().numpy()
    assert np.isfinite(rgb).all() and np.isfinite(alpha).all();pred=np.clip(rgb+.5*(1-alpha[...,None]),0,1);target=z['rgb'].astype(np.float32)/255*z['matte'][...,None]+.5*(1-z['matte'][...,None]);x0,y0,x1,y1=z['bbox'];bb=np.zeros_like(z['valid']);bb[y0:y1,x0:x1]=True;metrics={}
    for name,mask in {'full_valid':z['valid'],'foreground':z['valid']&(z['matte']>.5),'foreground_bbox':z['valid']&bb}.items():
        d=pred.astype(np.float64)[mask]-target.astype(np.float64)[mask];metrics[name]=dict(pixels=int(mask.sum()),psnr_db=float(-10*np.log10(np.mean(d*d))),mae=float(abs(d).mean()))
    if not reuse:np.savez_compressed(E/f'{stem}-prediction.npz',rgb=rgb,alpha=alpha);Image.fromarray(u8(pred)).save(E/f'{stem}-trained.png');Image.fromarray(u8(alpha)).save(E/f'{stem}-alpha.png')
    rows.append(dict(camera=c,frame=t,split=r['split'],metrics=metrics,input_sha256=r['input_sha256'],bbox=r['bbox'],reused079parent_prediction=reuse));save(E/'progress.json',dict(images=len(rows)));print('HAND_SCOPE_EVAL',b,c,t,flush=True)
assert sha(path)==before;save(E/'metrics.json',dict(status='18native_contexts_complete_comparison_pending',records=rows,checkpoint_sha256=before,goal_complete=False));shutil.copy2(__file__,R/Path(__file__).name)
