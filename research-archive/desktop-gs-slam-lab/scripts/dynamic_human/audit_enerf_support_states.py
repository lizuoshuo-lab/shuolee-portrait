"""Read every actual saved state and verify intended support-only interventions."""
from pathlib import Path
import json,hashlib,numpy as np,torch
torch.set_num_threads(8)
LAB=Path(__file__).resolve().parents[2];out=LAB/'outputs/dynamic-human-enerf-support-030';protocol=json.loads((out/'protocol.json').read_text());initial_path=out.parent/'dynamic-human-enerf-local-time-029/offset-index/first-frame/resume.pt';assert hashlib.sha256(initial_path.read_bytes()).hexdigest()==protocol['shared_initial_checkpoint_sha256'];initial=torch.load(initial_path,weights_only=False,map_location='cpu')['params'];reports=[]
for group in protocol['conditions']:
    run=out/group/'motion';done=json.loads((run/'completion.json').read_text());spec=json.loads((run/'protocol.json').read_text());assert done['frames']==protocol['frames'] and done['updates']==14400 and done['points']==150543;assert spec['initial_checkpoint_sha256']==protocol['shared_initial_checkpoint_sha256'];assert spec['source_manifest_sha256']==protocol['manifest_sha256'];rows=[];prev=initial
    for t in protocol['frames']:
        state=torch.load(run/f'frame-{t:03d}.pt',weights_only=False,map_location='cpu');assert set(state)==set(initial)
        for k,v in state.items():
            assert v.shape==initial[k].shape and torch.isfinite(v).all()
            if t==34:assert torch.equal(v,initial[k])
        changes={}
        for k,active in [('scales',group in ['scales','both']),('opacities',group in ['opacities','both'])]:
            delta=(state[k]-initial[k]).double();bound=np.log(2) if k=='scales' else 2.
            assert float(delta.abs().max())<=bound+1e-6
            if not active:assert torch.equal(state[k],initial[k])
            elif t>34:assert not torch.equal(state[k],initial[k])
            changes[k]=dict(active=active,mean_abs_initial=float(delta.abs().mean()),max_abs_initial=float(delta.abs().max()),fraction_at_bound=float((delta.abs()>=bound-1e-5).double().mean()),mean_abs_previous=float((state[k]-prev[k]).abs().mean()))
        ratio=(state['scales']-initial['scales']).double().exp().numpy();opacity=state['opacities'].double().sigmoid().numpy()
        rows.append(dict(frame=t,changes=changes,scale_ratio_q01_q50_q99=np.quantile(ratio,[.01,.5,.99]).tolist(),opacity_q01_q50_q99=np.quantile(opacity,[.01,.5,.99]).tolist(),center_step_p99=float(torch.quantile(torch.linalg.vector_norm(state['means']-prev['means'],dim=1),.99))));prev=state
    resume=torch.load(run/'resume.pt',weights_only=False,map_location='cpu');assert resume['frame']==58 and len(resume['history'])==24 and all(torch.equal(resume['params'][k],state[k]) for k in state)
    reports.append(dict(model=group,states=25,point_count=150543,first_state_exact=True,final_state_matches_resume=True,updates=14400,checks=rows))
    print('VERIFIED_MODEL',group,flush=True)
result=dict(status='100states_fixed_identity_common_initial_and_intended_support_bounds_verified',models=reports,first_frame_shared_updates=6000,total_updates_each=20400,limits=['Parameter index continuity is not proof of physical surface tracking','Bounds are safeguards for this local control, not physical validation of body geometry','No claims of capture seconds or independently verified metric world scale'])
(out/'independent-state-audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(result['status'])
