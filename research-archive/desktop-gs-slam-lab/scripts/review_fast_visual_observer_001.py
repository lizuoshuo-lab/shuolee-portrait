"""Validate new VIO residual observations against branch semantics and state logs."""
from pathlib import Path
from collections import OrderedDict
import csv,hashlib,json,re
import numpy as np
LAB=Path(__file__).resolve().parents[1]
NAME='exp14_basement_2_livo_20261013T085415Z_641c515c'
RUN=LAB/'outputs/fast-livo2-hilti-gt-003/runs'/NAME
OUT=LAB/'outputs/fast-livo2-visual-observer-001'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
status=read(RUN/'status.json');assert all(status['runtime_evidence_checks'].values())
assert status['bag_playback_completed'] and not status['error']
text=(RUN/'roslaunch.log').read_text(encoding='utf-8',errors='replace')
frames={};steps={};iters=OrderedDict()
for line in text.splitlines():
    for marker,target,size in (('[GS_VIO_FRAME]',frames,2),('[GS_VIO_ITER]',iters,11),('[GS_VIO_STEP]',steps,8)):
        if marker not in line:continue
        values=line.split(marker,1)[1].split()[:size];assert len(values)==size
        a=list(map(float,values));assert np.isfinite(a).all()
        key=int(a[0]) if size==2 else tuple(int(x) for x in a[:3])
        assert key not in target;target[key]=a
assert iters.keys()==steps.keys()
assert list(frames)==list(range(737))
groups=OrderedDict()
for key,a in iters.items():groups.setdefault(key[:2],[]).append(a)
rows=[];boundaries={};rejected=0
for (frame,level),group in groups.items():
    assert [int(a[2]) for a in group]==list(range(len(group))) and 1<=len(group)<=5
    old_error=np.finfo(np.float32).max;old_position=None
    for i,a in enumerate(group):
        f,l,it,points,n,error,last,accept,*pos=a;step=steps[(frame,level,int(it))]
        assert error>=0 and n==points*64 and n>0
        assert last==old_error and bool(accept)==(error<=last) and step[3]==accept
        if i:assert np.array_equal(pos,steps[(frame,level,i-1)][5:8])
        if accept:old_error=error;old_position=pos
        else:
            rejected+=1;assert np.array_equal(step[5:8],old_position) and step[4]==1 and i==len(group)-1
        rows.append(dict(frame=frame,level=level,iteration=i,points=int(points),scalar_measurements=int(n),prestep_photometric_MSE=error,previous_accepted_MSE=last,accepted_branch=bool(accept),termination_flag=bool(step[4]),position_step_m=float(np.linalg.norm(np.array(step[5:8])-pos))))
    final=steps[(frame,level,len(group)-1)]
    assert final[4]==1 or len(group)==5
    boundaries[(frame,level)]=dict(frame=frame,level=level,iterations=len(group),initial_MSE=group[0][5],last_accepted_tested_MSE=old_error,
        tested_ratio=old_error/group[0][5],termination='rollback' if not final[3] else ('small_update' if final[4] else 'iteration_budget'),
        returned_state_has_same_level_residual=not bool(final[3]))
# Bind observed positions to independent mixed LIO/VIO state text streams.
def state_groups(path):
    groups=OrderedDict()
    for line in path.read_text().splitlines():
        a=list(map(float,line.split()));groups.setdefault(a[0],[]).append(a)
    return groups
pre=state_groups(RUN/'official-log-after/mat_pre.txt');post=state_groups(RUN/'official-log-after/mat_out.txt')
def rounding(v):
    a=np.abs(v);return np.where(a>0,.5*10.**(np.floor(np.log10(np.maximum(a,1e-300)))-5),0)+2e-12
matched=0
for frame,fa in frames.items():
    if frame==0:assert not any(k[0]==frame for k in groups);continue
    timestamp=fa[1];keys=np.array(list(pre));idx=int(np.argmin(np.abs(keys-timestamp)));key=float(keys[idx])
    assert abs(key-timestamp)<=rounding(np.array(timestamp))
    assert len(pre[key])==len(post[key])==2
    assert np.array_equal(pre[key][1][:19],post[key][0][:19])
    first=iters[(frame,3,0)][8:11];lastlevel=groups[(frame,0)]
    final=steps[(frame,0,len(lastlevel)-1)][5:8]
    for observed,rounded in ((first,pre[key][1][4:7]),(final,post[key][1][4:7])):
        assert np.all(np.abs(np.array(observed)-rounded)<=rounding(np.array(rounded)))
    matched+=1
assert matched==736 and len(groups)==736*4
for filename,records in (('visual-iterations.csv',rows),('visual-levels.csv',list(boundaries.values()))):
    with (OUT/filename).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
summaries=[]
for level in range(4):
    a=[r for r in boundaries.values() if r['level']==level]
    summaries.append(dict(level=level,frame_count=len(a),median_tested_MSE_ratio=float(np.median([r['tested_ratio'] for r in a])),strict_tested_decreases=sum(r['tested_ratio']<1 for r in a),rollback_exits=sum(r['termination']=='rollback' for r in a),small_update_exits=sum(r['termination']=='small_update' for r in a),iteration_budget_exits=sum(r['termination']=='iteration_budget' for r in a)))
result=dict(status='completed_new_observed_full_replay',run=NAME,processed_visual_frames=len(frames),frames_with_visual_constraints=matched,frames_initializing_visual_map=1,
    observed_iterations=len(rows),observed_pyramid_level_paths=len(groups),accepted_branch_iterations=sum(r['accepted_branch'] for r in rows),rejected_and_rolled_back_iterations=rejected,
    exact_rejected_position_restoration_checks=rejected,frame_boundary_matches=matched,
    final_level_returns_without_residual_at_final_pose=sum(not v['returned_state_has_same_level_residual'] for v in boundaries.values() if v['level']==0),per_level=summaries,
    settings=dict(inverse_composition_enabled=False,patch_size_px=8,pyramid_levels=4,maximum_iterations_per_level=5,time_offsets_s=0),
    limitations=['New isolated replay; not historical run residuals','Accepted branch checks current pre-step error then takes another step','Small-update/budget exit does not reevaluate the newly returned pose at that level','Gray-value squared residual is not position error','No new calibration or time-shift tuning; no new accuracy improvement claim','Printf observations may perturb scheduling; not runtime benchmarking'])
save(OUT/'metrics.json',result)
save(OUT/'validation.json',dict(status='passed_branches_states_and_runtime_evidence',all_logged_branches_match_MSE_comparison=True,all_next_iteration_positions_match_previous_step=True,all_rollback_positions_match_saved_accepted_state=True,all_constrained_frame_positions_match_independent_state_logs=True,
    runtime_checks=status['runtime_evidence_checks'],inputs_sha256={str(p):sha(p) for p in (RUN/'roslaunch.log',RUN/'status.json',RUN/'official-log-after/mat_pre.txt',RUN/'official-log-after/mat_out.txt',OUT/'observer.patch',OUT/'source-override-receipt.json')},scope='Logged state coordinate matches account for six-significant-digit text rounding; iterator observations use seventeen digits.'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(2,1,figsize=(11,6.5),layout='constrained',sharex=True)
a=[r for r in boundaries.values() if r['level']==0];x=[frames[r['frame']][1] for r in a]
ax[0].plot(x,[r['initial_MSE'] for r in a],color='#999999',lw=.9,label='First evaluated state, finest level')
ax[0].plot(x,[r['last_accepted_tested_MSE'] for r in a],color='#007f88',lw=.9,label='Last accepted evaluated state')
ax[0].set(yscale='log',ylabel='Photometric MSE / gray value squared');ax[0].legend(fontsize=9)
ax[1].plot(x,[r['tested_ratio'] for r in a],color='#007f88',lw=.8);ax[1].axhline(1,color='gray',lw=.7)
ax[1].set(xlabel='Estimator elapsed image time / s',ylabel='Last accepted / first MSE')
for axis in ax:axis.grid(alpha=.2)
fig.suptitle('FAST-LIVO2: observed residual tests, not a residual evaluation of every returned pose')
fig.savefig(OUT/'visual-residual-observations.png',dpi=180);plt.close(fig)
print(json.dumps(result))
