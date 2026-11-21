"""Independent CPU review of the explicitly nonhistorical room continuations."""
from pathlib import Path
import csv, hashlib, json
import numpy as np
LAB=Path(__file__).resolve().parents[1]
OUT=LAB/'outputs/splatam-room-window-continuation-001'
TRAJ=LAB/'outputs/splatam-repeat-evaluations-001/freiburg1_room_full_320x240_rgbd_s0_20261010T125931Z_0d3dc918/trajectory'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
rows=read(OUT/'frames.json');metrics=read(OUT/'metrics.json')
iterations=list(csv.DictReader((OUT/'iterations.csv').open()))
assert [r['frame'] for r in rows]==list(range(801,843))+list(range(1101,1138))
assert len(iterations)==12*200
assert metrics['tracking_optimizer_steps']==79*200 and metrics['mapping_iterations']==79*30
T=np.array(read(TRAJ/'metrics.json')['alignment']['transform_estimated_to_reference'])
ref=np.loadtxt(TRAJ/'reference-independent.tum')
records=[]
for row in rows:
    f=row['frame'];before=np.linalg.inv(row['initial_w2c']);after=np.linalg.inv(row['selected_w2c'])
    eb=float(np.linalg.norm((T@before)[:3,3]-ref[f,1:4]));ea=float(np.linalg.norm((T@after)[:3,3]-ref[f,1:4]))
    assert abs(ea-row['fixed_alignment_position_error_m'])<1e-12
    a,b=row['initial'],row['selected']
    assert a['fixed_support_pixels']==b['fixed_support_pixels']==a['support_pixels']
    assert np.isclose(b['loss'],b['support_pixels']*(b['depth_mean_m']+b['weighted_color_mean']),rtol=2e-6)
    assert np.isclose(b['fixed_support_loss'],b['fixed_support_pixels']*(b['fixed_support_depth_mean_m']+b['fixed_support_weighted_color_mean']),rtol=2e-6)
    records.append(dict(frame=f,position_error_before_m=eb,position_error_after_m=ea,position_error_change_m=ea-eb,
        loss_ratio=b['loss']/a['loss'],fixed_support_loss_ratio=b['fixed_support_loss']/a['fixed_support_loss'],support_ratio=b['support_pixels']/a['support_pixels'],
        depth_before_m=a['depth_mean_m'],depth_after_m=b['depth_mean_m'],fixed_depth_after_m=b['fixed_support_depth_mean_m'],
        historical_position_error_m=row['historical_position_error_m'],continuation_vs_history_position_m=row['continuation_vs_historical_position_m']))
with (OUT/'review-per-frame.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
summary=dict(status='passed_independent_CPU_review',frames=79,focal_iteration_rows=len(iterations),fixed_alignment_reproduced=True,initial_support_count_preserved_all_frames=True,
    loss_decreased_frames=sum(r['loss_ratio']<1 for r in records),fixed_support_loss_decreased_frames=sum(r['fixed_support_loss_ratio']<1 for r in records),
    loss_decreased_but_position_error_increased_frames=sum(r['loss_ratio']<1 and r['position_error_change_m']>0 for r in records),
    fixed_support_loss_decreased_but_position_error_increased_frames=sum(r['fixed_support_loss_ratio']<1 and r['position_error_change_m']>0 for r in records),
    focal=[r for r in records if r['frame'] in (838,1133,1136,1137)],
    interpretation='These new continuations do not recover historical map/RNG state. Compare initial vs selected pose only within the same continuation frame/map; no drift-repair claim.',
    input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (OUT/'frames.json',OUT/'iterations.csv',TRAJ/'metrics.json',TRAJ/'reference-independent.tum')})
save(OUT/'validation.json',summary)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(3,2,figsize=(11.5,8.5),layout='constrained')
for col,(lo,hi) in enumerate(((801,842),(1101,1137))):
    a=[r for r in records if lo<=r['frame']<=hi];x=[r['frame'] for r in a]
    axes[0,col].plot(x,[100*r['position_error_after_m'] for r in a],color='#007f88',label='New continuation')
    axes[0,col].plot(x,[100*r['historical_position_error_m'] for r in a],color='#888888',ls='--',label='Historical trajectory')
    axes[0,col].set_title(f'Checkpoint {lo-1}: frames {lo}-{hi}');axes[0,col].legend(fontsize=9)
    axes[1,col].plot(x,[r['loss_ratio'] for r in a],label='Changing support',color='#b66025')
    axes[1,col].plot(x,[r['fixed_support_loss_ratio'] for r in a],label='Fixed initial support',color='#007f88')
    axes[1,col].axhline(1,color='gray',lw=.7);axes[1,col].legend(fontsize=9)
    axes[2,col].plot(x,[1000*r['position_error_change_m'] for r in a],color='#7c55a2')
    axes[2,col].axhline(0,color='gray',lw=.7);axes[2,col].set_xlabel('Frame index')
    for ax in axes[:,col]:ax.grid(alpha=.2)
axes[0,0].set_ylabel('Aligned position error / cm')
axes[1,0].set_ylabel('Selected / initial loss')
axes[2,0].set_ylabel('Selected - initial error / mm')
fig.suptitle('New checkpoint continuations: matching loss and reference error are different quantities')
fig.savefig(OUT/'window-loss-and-error.png',dpi=170);plt.close(fig)
print(json.dumps({k:v for k,v in summary.items() if k!='input_sha256'},ensure_ascii=False))
