from pathlib import Path
import json, hashlib, numpy as np

B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
R=B/'current/whole-body-early-trace-fresh-095';O=R/'gaussian-run/lineage'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def q(a):
    a=np.asarray(a,float);a=a[np.isfinite(a)]
    return dict(n=len(a),median=float(np.median(a)),p90=float(np.quantile(a,.9)),max=float(a.max())) if len(a) else dict(n=0)
def sigmoid(a):return 1/(1+np.exp(-np.clip(a,-80,80)))

def main():
    assert read(R/'gaussian-run/result.json')['steps']==1800
    roots=np.load(O/'initial-roots.npz');groups=roots['groups'];extent=np.linalg.norm(np.ptp(roots['xyz'],axis=0))
    pairs=np.load(B/'current/whole-body-alignment-audit-094/queries.npz')['pairs']
    labels=['head_height','shoulder_arms_height','trunk_hands_height','legs_shoes_height']
    events=[];ids=np.arange(len(groups));born_steps=np.zeros(len(groups),int);kinds=np.zeros(len(groups),np.int8)
    for path in sorted(O.glob('event-*.npz')):
        e=np.load(path);kind=str(e['kind']);record=dict(file=path.name,kind=kind,step=int(e['step']),frame=int(e['frame'])+1)
        if kind=='prune':
            record['removed']=len(e['removed']);assert np.isin(e['removed'],ids).all();ids=ids[~np.isin(ids,e['removed'])]
        else:
            assert np.isin(e['parent'],ids).all();assert np.array_equal(e['child'],np.arange(len(born_steps),len(born_steps)+len(e['child'])))
            record.update(created=len(e['child']),root_bands={lab:int((groups[e['root']]==i).sum()) for i,lab in enumerate(labels)})
            ids=np.r_[ids,e['child']];born_steps=np.r_[born_steps,np.full(len(e['child']),int(e['step']))];kinds=np.r_[kinds,np.full(len(e['child']),1 if kind=='clone' else 2,np.int8)]
        events.append(record)
    final=np.load(O/'step-001800-post_step/gaussians.npz');assert np.array_equal(ids,final['ids'])
    sampled=[]
    for path in sorted(O.glob('updates-*.npz')):
        z=np.load(path);d=z['after'][:,:3]-z['before'][:,:3];grp=groups[z['roots']]
        rows={label:q(np.linalg.norm(d[grp==i],axis=1)/extent) for i,label in enumerate(labels)}
        sampled.append(dict(start=int(z['steps'][0]),end=int(z['steps'][-1]),records=len(d),actual_adam_position_update_fraction_extent=rows))
    snapshots=[]
    for folder in sorted(O.glob('step-*')):
        if not (folder/'metrics.json').exists():continue
        z=np.load(folder/'gaussians.npz');m=read(folder/'metrics.json');transfer=np.load(folder/'transfers.npz')
        assert all(np.isfinite(z[k]).all() for k in ['xyz','scaling','rotation','opacity','features_dc','features_rest'])
        scales=np.exp(z['scaling']);displacement=np.linalg.norm(z['xyz']-roots['xyz'][z['roots']],axis=1)/extent
        bound=[]
        for lab in labels:
            rows=[r for r in m['boundary_four_fixed_views'] if r['region']==lab]
            bound.append(dict(region=lab,outside_alpha_mean=float(sum(r['boundary_outside_alpha_mass'] for r in rows)/sum(r['boundary_outside_pixels'] for r in rows)),
                outward_contour_pixels=sum(r['outward_contour_pixels'] for r in rows),inward_contour_pixels=sum(r['inward_contour_pixels'] for r in rows)))
        snapshots.append(dict(folder=folder.name,step=m['step'],points=len(z['ids']),
            descendant_points=int((z['ids']>=len(groups)).sum()),opacity=q(sigmoid(z['opacity'])),
            min_scale_fraction_extent=q(scales.min(1)/extent),max_scale_fraction_extent=q(scales.max(1)/extent),
            final_root_displacement_fraction_extent={lab:q(displacement[groups[z['roots']]==i]) for i,lab in enumerate(labels)},
            transfers=m['transfers'],boundary_four_views=bound))
    interventions=[]
    for step,event in [(500,'before_opacity_reset'),(600,'before_densify'),(1000,'before_densify')]:
        before=O/f'step-{step:06d}-{event}';after=O/f'step-{step:06d}-post_step';a=np.load(before/'gaussians.npz');z=np.load(after/'gaussians.npz')
        common,ia,iz=np.intersect1d(a['ids'],z['ids'],return_indices=True)
        detail=dict(step=step,event=event,surviving_preexisting=len(common),new_after=int((~np.isin(z['ids'],a['ids'])).sum()),removed=int((~np.isin(a['ids'],z['ids'])).sum()),
            surviving_xyz_max_abs=float(np.abs(a['xyz'][ia]-z['xyz'][iz]).max()),surviving_logscale_max_abs=float(np.abs(a['scaling'][ia]-z['scaling'][iz]).max()),
            surviving_rotation_max_abs=float(np.abs(a['rotation'][ia]-z['rotation'][iz]).max()),surviving_dc_max_abs=float(np.abs(a['features_dc'][ia]-z['features_dc'][iz]).max()),
            alpha_before=q(sigmoid(a['opacity'][ia])),alpha_after=q(sigmoid(z['opacity'][iz])))
        t0=np.load(before/'transfers.npz');t1=np.load(after/'transfers.npz');rec=[]
        for g,name in enumerate(['face_lower','face_upper','torso','legs_shoes'],1):
            finite=(pairs[:,1]==g)&t0['finite']&t1['finite'];visible=finite&t0['visible']&t1['visible']
            er0=np.linalg.norm(t0['uv']-pairs[:,6:8],axis=1);er1=np.linalg.norm(t1['uv']-pairs[:,6:8],axis=1)
            rec.append(dict(region=name,same_finite_queries_before=q(er0[finite]),same_finite_queries_after=q(er1[finite]),
                same_opaque_queries_before=q(er0[visible]),same_opaque_queries_after=q(er1[visible]),
                visible_before=int(((pairs[:,1]==g)&t0['visible']).sum()),visible_after=int(((pairs[:,1]==g)&t1['visible']).sum())))
        detail['transfers']=rec;interventions.append(detail)
    save(R/'early-timeline.json',dict(steps=1800,exact_lineage_replay_passed=True,extent_world=float(extent),events=events,snapshots=snapshots,events_without_optimizer_updates_between=interventions,
        sampled_actual_updates=sampled,limitations=['Body bands are initial projected image-height bands, not semantic anatomy.',
        'Only sampled roots have every-step Adam updates. All surviving points have milestone snapshots.',
        'Plane-depth transfer can be ambiguous at low alpha or mixed normals; retain alpha and opaque-query counts.',
        'The500/600 snapshot pairs isolate their scheduled events; long-interval changes do not prove one loss caused them.',
        'No camera ground truth, historical white-fringe causal attribution or quality acceptance.'],quality_accepted=False))
    np.savez_compressed(R/'point-lineage-index.npz',born_step=born_steps,kind=kinds,final_ids=final['ids'],final_roots=final['roots'])
    for x in snapshots:
        print(json.dumps(dict(state=x['folder'],points=x['points'],opacity=x['opacity']['median'],torso=x['transfers']['torso']['all_finite_error_px']['median'],legs=x['transfers']['legs_shoes']['all_finite_error_px']['median'],outside={r['region']:r['outside_alpha_mean'] for r in x['boundary_four_views']})))
    print(json.dumps(dict(events=interventions)))

if __name__=='__main__':main()
