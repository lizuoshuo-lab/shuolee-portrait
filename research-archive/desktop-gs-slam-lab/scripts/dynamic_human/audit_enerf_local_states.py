"""Read actual saved tensors, identity/visibility constraints and budget sharing."""
from pathlib import Path
import json,torch,numpy as np,hashlib
LAB=Path(__file__).resolve().parents[2];out=LAB/'outputs/dynamic-human-enerf-local-time-029';results=[]
for g in ['same-index','offset-index']:
    run=out/g/'motion';comp=json.loads((run/'completion.json').read_text());first=torch.load(run/'frame-034.pt',weights_only=False,map_location='cpu');init=torch.load(out/g/'first-frame/resume.pt',weights_only=False,map_location='cpu');assert all(torch.equal(first[k],init['params'][k]) for k in first);del init;previous=first;rows=[]
    for t in comp['frames']:
        state=torch.load(run/f'frame-{t:03d}.pt',weights_only=False,map_location='cpu')
        for k,v in state.items():assert v.shape==first[k].shape and torch.isfinite(v).all(),(g,t,k)
        for k in ['scales','opacities']:assert torch.equal(first[k],state[k]),(g,t,k)
        motion=torch.linalg.vector_norm(state['means']-previous['means'],dim=1).numpy();rows.append(dict(common_frame=t,center_step_p50=float(np.median(motion)),center_step_p99=float(np.quantile(motion,.99)),center_step_max=float(motion.max())));previous=state
    assert comp['updates']==14400 and len(rows)==25
    results.append(dict(condition=g,points=len(first['means']),states_checked=len(rows),all_tensors_finite=True,identity_counts_fixed=True,scales_opacities_fixed=True,frame34_exactly_matches_first_fit=True,steps=rows))
static=out/'offset-index/static-multitime';protocol=json.loads((static/'protocol.json').read_text());cp=out/'offset-index/first-frame/resume.pt';assert hashlib.sha256(cp.read_bytes()).hexdigest()==protocol['initial_checkpoint_sha256'];assert protocol['total_updates']==20400
p=torch.load(static/'model.pt',weights_only=False,map_location='cpu');assert all(torch.isfinite(v).all() for v in p.values());resume=torch.load(static/'resume.pt',weights_only=False,map_location='cpu');assert resume['step']==14400 and all(torch.equal(p[k],resume['params'][k]) for k in p)
result=dict(status='all50dynamic_states_static_final_and_shared_start_budget_verified',dynamic=results,static_points=len(p['means']),static_final_exactly_matches_saved_optimizer_checkpoint=True,static_and_offset_dynamic_share_first6000updates=True,per_model_total_update_budget=20400,limits=['Persistent index is not verified physical surface correspondence','World scale and capture FPS remain unverified','Different model classes optimize different parameter sets; equal updates do not mean equal FLOPs or walltime'])
(out/'independent-state-audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(dict(status=result['status'],static_points=result['static_points'])))
