from pathlib import Path
import json,hashlib,numpy as np,argparse
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current')
REF=B/'person-static-pgsr-fresh-064'
names={'065':'person-static-no-scale-fresh-065','066':'person-static-no-multiview-fresh-066'}
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(n):
    R=B/names[n];ga=R/'gaussian-run';a=read(REF/'gaussian-run/config.json');b=read(ga/'config.json')
    for rel in ['clothing-init/points.npz','clothing-init/motion.npz','data/training-inputs.json','data/observations.npz']:
        if rel.endswith('.npz'):
            x=np.load(REF/rel);y=np.load(R/rel);assert x.files==y.files and all(np.array_equal(x[k],y[k]) for k in x.files),rel
        else:assert sha(REF/rel)==sha(R/rel),rel
    assert a['max_height']==b['max_height']==1536 and a['steps']==b['steps']==18000
    diffs={k:[a['options'][k],v] for k,v in b['options'].items() if a['options'].get(k)!=v}
    expected={'scale_loss_weight':[100.,0.]} if n=='065' else {'multi_view_geo_weight':[.03,0.],'multi_view_ncc_weight':[.15,0.]}
    assert diffs==expected,diffs
    assert read(REF/'gaussian-run/training-cameras.json')==read(ga/'training-cameras.json')
    cams=read(ga/'training-cameras.json')['records'];rng=np.random.RandomState(6060);stack=[];sched=[]
    for step in range(1,18001):
        if not stack:stack=rng.permutation(175).tolist()
        i=stack.pop();j=int(rng.choice(cams[i]['neighbor_ids'])) if step>7000 else -1;sched.append([i,j])
    counts=np.bincount(np.array(sched)[:,0],minlength=175)
    ref_counts=read(REF/'gaussian-run/result.json')['per_frame_updates'];assert counts.tolist()==ref_counts
    final=(ga/'result.json').exists()
    if final:assert np.array_equal(np.load(ga/'sampling-schedule.npy'),np.array(sched))
    state=dict(reference='064',candidate=n,only_option_changes=diffs,raw_inputs_equal=True,
        freshly_regenerated_initial_arrays_equal=True,cameras_equal=True,height=1536,
        initial_points=200000,steps=18000,sequence_reconstructed_from_reference_code=True,
        reference_counts_exact=True,actual_candidate_sequence_exact=final,
        code_reference_sha256=sha(REF/'source-final/shuolee_static.py'),candidate_source_sha256=sha(R/'source-final/shuolee_static.py'),
        caveat='One seed on one sequence. CUDA accumulation and densification can amplify differences; no independent geometry GT.',
        quality_accepted=False)
    (R/'controlled-protocol.json').write_text(json.dumps(state,indent=2)+'\n',encoding='utf8');print(json.dumps(state))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('round',choices=names);run(p.parse_args().round)
