"""Cycle-consistent integer frame-offset hypothesis from training anchors only."""
from pathlib import Path
import argparse,json
import numpy as np
from audit_enerf_input import digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);src=out/'timing-diagnostic.json';data=json.loads(src.read_text());protocol=json.loads((out/'protocol.json').read_text());anchors=protocol['anchor_frames'];cams=sorted({r['source'] for r in data['pairs']}|{r['target'] for r in data['pairs']});ci={c:i for i,c in enumerate(cams)};anchor=ci['08'];edges=[]
    for pair in data['selected_candidates']:
        c,d=pair['source'],pair['target'];fit=[r for r in data['pairs'] if r['source']==c and r['target']==d and r['anchor'] in anchors[:2] and r['moving_count']>=12]
        if len(fit)!=2:continue
        cost=np.mean([r['moving']['scores_px'] for r in fit],axis=0);edges.append((ci[c],ci[d],cost))
    adjacency={c:set() for c in cams}
    for i,j,_ in edges:adjacency[cams[i]].add(cams[j]);adjacency[cams[j]].add(cams[i])
    components=[];remaining=set(cams)
    while remaining:
        pending=[min(remaining)];component=set()
        while pending:
            c=pending.pop()
            if c in component:continue
            component.add(c);pending.extend(adjacency[c]-component)
        components.append(sorted(component));remaining-=component
    if len(components)!=1:
        result=dict(status='rejected_global_time_offset_not_identifiable_from_disconnected_fit_graph',source_sha256=digest(src),fit_anchors=anchors[:2],independent_check_anchors=anchors[2:],reference_camera='08',components=components,accepted_fit_edges=len(edges),candidate_edges=len(data['selected_candidates']),offsets=None,dataset_role=protocol.get('dataset_role'),reason='Each fit edge requires at least12 moving mutually tracked matches in BOTH fit anchors. No relative offset can be established between disconnected components; a numeric minimum assigns arbitrary gauges and must not be applied.',retiming_performed=False)
        (out/'global-offset-hypothesis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result,indent=2));return
    def loss(v):
        return sum(cost[v[j]-v[i]+3] if abs(v[j]-v[i])<=3 else 50+10*abs(v[j]-v[i]) for i,j,cost in edges)
    rng=np.random.default_rng(73);solutions=[]
    for start in range(100):
        v=np.zeros(len(cams),dtype=int) if start==0 else rng.integers(-2,3,len(cams));v[anchor]=0
        for sweep in range(30):
            old=v.copy()
            for k in rng.permutation(len(cams)):
                if k==anchor:continue
                candidates=[]
                for n in range(-2,3):
                    v[k]=n;candidates.append(loss(v))
                v[k]=int(np.argmin(candidates))-2
            if np.array_equal(old,v):break
        solutions.append((loss(v),v.copy()))
    solutions.sort(key=lambda x:x[0]);local_value,local_best=solutions[0]
    # Exact dynamic program: the measured graph connects at most three
    # preceding camera positions. Retain their labels as the separator state.
    ordered=[]
    for pair in data['selected_candidates']:
        for c in [pair['source'],pair['target']]:
            if c not in ordered:ordered.append(c)
    position={c:i for i,c in enumerate(ordered)}
    edge_cost={(cams[i],cams[j]):cost for i,j,cost in edges}
    assert all(0<position[d]-position[c]<=3 for c,d in edge_cost)
    states={(): (0.0,[])}
    for k,c in enumerate(ordered):
        nxt={}
        for state,(cost,path) in states.items():
            for label in ([0] if c=='08' else range(-2,3)):
                added=0.0
                for j in range(max(0,k-3),k):
                    q=edge_cost.get((ordered[j],c))
                    if q is None:continue
                    lag=label-path[j]
                    added+=float(q[lag+3]) if abs(lag)<=3 else 50+10*abs(lag)
                key=tuple((path+[label])[-3:]);candidate=(cost+added,path+[label])
                if key not in nxt or candidate[0]<nxt[key][0]:nxt[key]=candidate
        states=nxt
    value,path=min(states.values(),key=lambda x:x[0]);best=np.array([path[position[c]] for c in cams]);assert abs(value-loss(best))<1e-8;checks=[]
    for r in data['pairs']:
        lag=int(best[ci[r['target']]]-best[ci[r['source']]])
        row=dict(source=r['source'],target=r['target'],anchor=r['anchor'],selected_lag=lag,moving_count=r['moving_count'],zero_px=r['moving']['zero_px'],selected_px=r['moving']['scores_px'][lag+3] if r['moving']['scores_px'] else None,still_count=r['low_motion_count'],still_zero_px=r['low_motion']['zero_px'],still_selected_px=r['low_motion']['scores_px'][lag+3] if r['low_motion']['scores_px'] else None)
        checks.append(row)
    aggregate={}
    for t in anchors:
        rows=[r for r in checks if r['anchor']==t and r['moving_count']>=12];aggregate[str(t)]=dict(pairs=len(rows),mean_pair_moving_zero_px=float(np.mean([r['zero_px'] for r in rows])),mean_pair_moving_selected_px=float(np.mean([r['selected_px'] for r in rows])),pairs_improved=sum(r['selected_px']<r['zero_px'] for r in rows),pairs_unchanged=sum(r['selected_lag']==0 for r in rows))
    result=dict(status='empirical_integer_camera_index_offset_hypothesis_not_hardware_timestamp_measurement',source_sha256=digest(src),reference_camera='08',source_index_for_common_frame='frame + offset[camera]',offsets=dict(zip(cams,map(int,best))),fit_anchors=[9,29],independent_check_anchors=[49],training_only=True,objective='Sum of pair costs: equal-weight mean of anchor9 and29 moving-point median absolute Sampson distance; all graph edges, global integer offsets [-2,+2], 08 fixed zero',multi_start=100,best_loss=float(value),best_solution_repeats=sum(np.array_equal(v,best) for _,v in solutions),zero_loss=float(loss(np.zeros(len(cams),dtype=int))),aggregate=aggregate,pairs=checks,limits=['Offsets are empirical image-index hypotheses, not acquisition-clock measurements','Heldout cameras not estimated; cannot retime their references from this result','Fixed calibration residual, LK drift, shutter blur and deformation remain potential confounders','No training or dataset mutation performed by this solver'])
    result['exact_graph_solver']='Dynamic programming over width-3 ordered camera graph; exact minimum within discrete [-2,+2] camera offsets and fixed gauge08=0, not a proof of physical timing'
    result['multi_start_best_loss']=float(local_value)
    result['fit_anchors']=anchors[:2];result['independent_check_anchors']=anchors[2:]
    result['objective']='Sum of pair costs: equal-weight mean of the two fit-anchor moving-point median absolute Sampson distances; all graph edges, integer offsets [-2,+2], 08 fixed zero'
    result['dataset_role']=protocol.get('dataset_role','original_appearance_training_observations')
    if result['dataset_role'].endswith('temporal_calibration_only'):
        result['training_only']=False
        result['limits'][1]=f'All18 cameras used only in designated calibration windows around{anchors}; cameras00,06,12 are calibration-visible and only later appearance images are held out. Follow the source-manifest split_scope.'
    (out/'global-offset-hypothesis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8');print(json.dumps({k:v for k,v in result.items() if k!='pairs'},indent=2))
if __name__=='__main__':main()
