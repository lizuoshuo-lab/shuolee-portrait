"""Independently replay cache membership on the CPU; render evidence stays bound to raw outputs."""
from pathlib import Path
import hashlib,json
import numpy as np
LAB=Path(__file__).resolve().parents[1]
BASE=LAB/'outputs/large-scene-gs-001'
RAW=BASE/'research/city-parameter-paging-023'
OLD=BASE/'independent-evaluation/small-city006-heldout-1404-001'
OUT=LAB/'outputs/city-parameter-paging-001'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
OUT.mkdir(exist_ok=False)
m=read(RAW/'metrics.json');rows=read(RAW/'measurements.json');bound=read(RAW/'input-bindings.json')
source=read(OLD/'merged-post/tau3/per-frame-audit.json')
capacity=m['tree_capacity'];tree_count=m['total_model_rows']-m['skybox_rows']
slot=np.full(capacity,-1,np.int32);where=np.full(tree_count,-1,np.int32);age=np.full(capacity,-1,np.int64)
assert len(rows)==len(source)==117
for j,r in enumerate(rows):
    assert j==r['index'] and r['rgb_bitwise_equal'] and r['rgb_max_abs_difference']==0
    with np.load(source[j]['selected']['path']) as a:needed=np.union1d(a['indices'],a['parent_gaussian_indices'])
    new=needed[where[needed]<0];active=where[needed];protect=np.zeros(capacity,bool);protect[active[active>=0]]=True
    candidates=np.flatnonzero(~protect);replace=candidates[np.argsort(age[candidates],kind='stable')[:len(new)]]
    evicted=slot[replace];evicted=evicted[evicted>=0]
    assert len(new)==r['cache_misses'] and len(needed)-len(new)==r['cache_hits'] and len(evicted)==r['evicted_rows']
    assert 236*len(new)==r['raw_parameter_transfer_bytes']
    where[evicted]=-1;where[new]=replace;slot[replace]=new;age[where[needed]]=j
    assert np.array_equal(slot[where[needed]],needed)
    assert np.count_nonzero(slot>=0)==r['occupied_tree_rows']<=capacity
assert m['resident_parameter_bytes']==(capacity+100000)*268
assert m['whole_model_parameter_bytes']==m['total_model_rows']*268
assert m['total_raw_parameter_transfer_bytes']==sum(r['raw_parameter_transfer_bytes'] for r in rows)+236*100000
m['resident_parameter_reduction_fraction']=1-m['resident_parameter_bytes']/m['whole_model_parameter_bytes']
m['required_row_cache_hit_fraction']=m['total_cache_hits']/(m['total_cache_hits']+m['total_cache_misses'])
save(OUT/'metrics.json',m)
save(OUT/'validation.json',dict(status='passed_independent_CPU_cache_replay',views=117,all_cache_membership_transitions_match=True,all_counts_and_transfer_bytes_match=True,raw_RGB_checks_passed=all(r['rgb_bitwise_equal'] for r in rows),
    source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (RAW/'metrics.json',RAW/'measurements.json',RAW/'input-bindings.json',RAW/'source.py')},
    limits='Independent CPU cache validation; GPU floating images were checked by the actual execution, not rerendered by this review. Timing excludes cut selection and archive reading. Camera order is sparse heldout sampling, not a continuous interactive path.'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(2,1,figsize=(10.5,6),layout='constrained',sharex=True)
x=np.arange(117)
ax[0].plot(x,[r['raw_parameter_transfer_bytes']/2**20 for r in rows],color='#007f88')
ax[0].set_ylabel('Missing raw parameters / MiB')
ax[1].plot(x,[r['paging_host_ms'] for r in rows],color='#b66025',label='CPU bookkeeping + upload + decode')
ax[1].plot(x,[r['render_host_ms'] for r in rows],color='#007f88',label='Unchanged renderer')
ax[1].set(xlabel='Heldout view index (sparse original order)',ylabel='Synchronized time / ms');ax[1].legend(fontsize=9)
for a in ax:a.grid(alpha=.2)
fig.suptitle('Bounded GPU cache: 117 / 117 images exactly equal to archived tau=3 renders')
fig.savefig(OUT/'paging-transfer-time.png',dpi=180);plt.close(fig)
print(json.dumps(m))
