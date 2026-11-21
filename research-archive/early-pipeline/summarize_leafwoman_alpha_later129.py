"""Thin CPU adapter of frozen1000 alpha audit for existing10k/18k snapshots.

Never imports a trainer, reads live state as completion, renders or modifies
training. Compare same derived masks/cameras; no opacity-reset causal claim.
"""
import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
import textwrap

import numpy as np
import summarize_leafwoman_alpha129 as core

CORE_SHA='278cade800558f19c307ecaf84ad51fa0539606a12cbaa8b75e334eea1715dea'
EARLY=core.G/'gaussian-run/step-001000/alpha-summary129.json'
EARLY_SHA='c59d34cb4fabfa6657afae2611df526a23535c17d671637246216e05a5737a7d'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def adapted_audit(step):
    assert sha(core.__file__)==CORE_SHA and step in (10000,18000)
    source=textwrap.dedent(inspect.getsource(core.main))
    edits=[
        ("G/'process-exit.json',SNAP/'result.json'","SNAP/'result.json'"),
        ("assert result['step']==result['optimizer_updates']==1000","assert result['step']==STEP"),
        ("assert result['points']==33767 and result['frames_rendered']==12","assert result['points']>0 and result['frames_rendered']==12"),
        ("assert read(G/'process-exit.json')['exit_code']==0","assert STEP in (10000,18000) # Fixed snapshot audit permits ongoing same-candidate training."),
        ("subject='leafwoman',step=1000,","subject='leafwoman',step=STEP,"),
        ("optimizer_updates=1000,points=33767","optimizer_updates=STEP-1,points=result['points']"),
        ("snapshot_global_iteration=1000","snapshot_global_iteration=STEP"),
        ("periodic_reset_multiples_reached=0","periodic_reset_multiples_reached=(STEP-1)//interval"),
        ("'No1800-period reset has occurred before this1000 snapshot; these artifacts cannot establish periodic opacity-reset causation.'",
         "'Clock-implied original periodic resets before this pre-update snapshot are recorded, but additional optimization, geometry, density/pruning and other changes co-occur. This is not a reset intervention or causal isolation.'"),
    ]
    for old,new in edits:
        assert source.count(old)==1,(old,source.count(old));source=source.replace(old,new)
    tree=ast.parse(source);fn=tree.body[0]
    assert isinstance(fn.body[-2],ast.With) and isinstance(fn.body[-1],ast.Expr)
    # Reuse all immutable file/hash/float32/PNG/mask/quantile checks. Return the
    # summary in memory so this new adapter writes one final receipt only.
    fn.body[-2:]=[ast.Return(value=ast.Name(id='summary',ctx=ast.Load()))]
    tree=ast.fix_missing_locations(tree)
    snapshot=core.G/f'gaussian-run/step-{step:06d}'
    ns=dict(core.__dict__,STEP=step,SNAP=snapshot,OUT=snapshot/'alpha-summary129.json',__file__=__file__)
    exec(compile(tree,str(__file__)+'::same_raw_alpha_audit','exec'),ns)
    return ns['main'],snapshot,hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()


def point_opacity(snapshot):
    from plyfile import PlyData
    path=snapshot/'world.ply';result=core.read(snapshot/'result.json');digest=sha(path)
    assert digest==result['world_ply_sha256']
    vertex=PlyData.read(path)['vertex'].data
    assert len(vertex)==result['points'] and 'opacity' in vertex.dtype.names
    logits=np.asarray(vertex['opacity'],dtype=np.float64);assert np.isfinite(logits).all()
    # Stable sigmoid, no alteration or lower/upper logit clamp.
    values=np.empty_like(logits);pos=logits>=0
    values[pos]=1/(1+np.exp(-logits[pos]));exp=np.exp(logits[~pos]);values[~pos]=exp/(1+exp)
    probs=[0.,.01,.1,.25,.5,.75,.9,.99,1.]
    assert sha(path)==digest
    return dict(points=len(vertex),world_ply_sha256=digest,logit_dtype=str(vertex['opacity'].dtype),
        sigmoid_mean=float(values.mean()),sigmoid_median=float(np.median(values)),
        quantile_probabilities=probs,sigmoid_quantiles=[float(x) for x in np.quantile(values,probs)],
        scope='Unweighted distribution over current Gaussian points; not ray alpha, visible front-surface opacity or matched point lineage. Densification/pruning changes the population.')


def main(step,actual_frame18_reviewed):
    assert sha(core.__file__)==CORE_SHA and sha(EARLY)==EARLY_SHA
    fn,snapshot,generated_sha=adapted_audit(step);summary=fn()
    earlier=core.read(EARLY)
    previous={r['frame']:r for r in earlier['records']}
    assert set(previous)=={r['frame'] for r in summary['records']}
    for key in (core.G/'data/training-inputs.json',core.G/'pixel-mapping.json'):
        assert summary['source_hashes'][str(key)]==earlier['source_hashes'][str(key)]
    comparison={};perframe=[]
    for row in summary['records']:
        old=previous[row['frame']]
        assert row['derived_mask_sha256']==old['derived_mask_sha256'] and row['derived_wh']==old['derived_wh']
        for region in core.REGIONS:assert row['regions'][region]['pixels']==old['regions'][region]['pixels']
        perframe.append(dict(frame=row['frame'],regions={r:dict(
            delta_mean=row['regions'][r]['mean']-old['regions'][r]['mean'],
            delta_fraction_below_0_5=row['regions'][r]['fraction_below_0_5']-old['regions'][r]['fraction_below_0_5'],
            delta_fraction_below_0_95=row['regions'][r]['fraction_below_0_95']-old['regions'][r]['fraction_below_0_95']) for r in core.REGIONS}))
    for region in core.REGIONS:
        a=earlier['area_weighted_aggregate'][region];b=summary['area_weighted_aggregate'][region]
        assert a['pixels']==b['pixels']
        comparison[region]=dict(pixels=a['pixels'],step1000=a,later=b,
            delta_mean=b['mean']-a['mean'],delta_median=b['median']-a['median'],
            delta_fraction_below_0_5=b['fraction_below_0_5']-a['fraction_below_0_5'],
            delta_fraction_below_0_95=b['fraction_below_0_95']-a['fraction_below_0_95'])
    if not actual_frame18_reviewed:
        summary['actual_viewed_images']=[]
        summary['frame18_visual_note']='No actual image review declared by this invocation; numeric hash audit only.'
    else:
        summary['frame18_visual_note']='Actual current frame18 Gaussian+alpha PNG inspected: lower skirt has stretched strips, holes and a lateral pale fragment near the bottom. High accumulated alpha there does not prove correct skirt/feet geometry. No result-driven pixel/point deletion.'
    summary.update(helper_sha256=sha(__file__),frozen_1000_audit_helper_sha256=CORE_SHA,
        generated_audit_AST_sha256=generated_sha,earlier_summary=str(EARLY),earlier_summary_sha256=EARLY_SHA,
        snapshot_phase='Original Scene.save before this iteration densification and optimizer.step; step label is not optimizer-update count.',
        optimizer_update_count_source='Original train.py saving branch precedes density/Adam. Earlier1000 was special post-update reviewed stop; later10000/18000 use original pre-update saving branch.',
        comparison_to_step1000=comparison,per_frame_comparison_to_step1000=perframe,
        point_opacity_distributions=dict(step1000=point_opacity(EARLY.parent),later=point_opacity(snapshot)),
        causal_limit='Same candidate/masks/cameras but many co-occurring updates, geometry/scale/opacity learning, density/pruning and clock-implied periodic resets. Descriptive progression only; cannot attribute a change to reset or any unique cause.',
        training_liveness_claim='This CPU helper never polls, signals, waits on or changes the training PID; mutable live progress/exit files are deliberately not completion gates.',
        quality_accepted=False)
    assert sha(core.__file__)==CORE_SHA and sha(EARLY)==EARLY_SHA
    out=snapshot/'alpha-summary129.json'
    with out.open('x',encoding='utf-8') as stream:stream.write(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(path=str(out),sha256=sha(out),helper_sha256=sha(__file__),step=step,
        points=summary['points'],optimizer_updates=step-1,aggregate=summary['area_weighted_aggregate'],
        comparison={k:{n:v for n,v in r.items() if n.startswith('delta_')} for k,r in comparison.items()},
        opacity=summary['point_opacity_distributions'],quality_accepted=False),ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--step',type=int,choices=[10000,18000],required=True)
    p.add_argument('--actual-frame18-reviewed',action='store_true');a=p.parse_args()
    main(a.step,a.actual_frame18_reviewed)
