from pathlib import Path
import json,numpy as np,hashlib,shutil
LAB=Path(__file__).resolve().parents[2];OUT=LAB/'outputs/dynamic-human-enerf-time-graph-051';SRC=LAB/'outputs/dynamic-human-enerf-actor56-time-049'
read=lambda p:json.loads(p.read_text())
save=lambda p,x:p.write_text(json.dumps(x,indent=2)+'\n')
s=read(OUT/'summary.json');offset=s['offsets'];rows=[]
for r in s['records']:
 c,d=r['source'],r['target'];lag=offset[d]-offset[c];assert lag==r['lag']
 scores=[]
 for t in [740,750,760]:
  z=np.load(SRC/f'tracks-{c}-{d}-t{t:03d}.npz');m=z['source_speed']>1
  if t<760:assert m.sum()>=12
  values=np.median(np.abs(z['signed_sampson'][m]),axis=0)
  scores.append(values)
 assert abs(np.mean([v[3] for v in scores[:2]])-r['fit_zero_px'])<1e-10
 assert abs(np.mean([v[lag+3] for v in scores[:2]])-r['fit_selected_px'])<1e-10
 if r['check_covered']:
  assert np.allclose([scores[2][3],scores[2][lag+3]],[r['check_zero_px'],r['check_selected_px']],atol=1e-10,rtol=0)
  if lag:rows.append(dict(source=c,target=d,lag=lag,before=r['check_zero_px'],after=r['check_selected_px']))
extra=[r for r in rows if r['source'] in ['02','05'] or r['target'] in ['02','05']]
assert len(extra)==4 and sum(r['after']>r['before'] for r in extra)==3
save(OUT/'independent-audit.json',dict(status='87saved_track_arrays_recomputed_fit_and_check_scores_exact',edge_count=len(s['records']),
    source_hash=hashlib.sha256((SRC/'timing-diagnostic.json').read_bytes()).hexdigest(),extra02and05check=extra))
save(OUT/'final-summary.json',dict(status='additional_global02and05shifts_fail_independent_anchor_no_global_retiming',
    offsets=offset,fit_edges=s['fit_edges'],check_edges=s['check_edges'],check_mean_before=s['check_zero_mean_px'],
    check_mean_after=s['check_selected_mean_px'],extra02and05check=extra,
    decision='Most aggregate improvement comes from camera10 already tested050. Of four covered check edges involving additional02/05shifts,02to05improves but13to02,15to02,15to05allworsen. Do not launch global shifted training or modify original data.',
    limitations='Not a physical clock calibration; offset02hits selected labelbound, but failed independent checks give no reason to widen search and tune against checkframe.',goal_complete=False))
shutil.copy2(__file__,OUT/Path(__file__).name)
print('051fixed-track scores verified; additional global shifts rejected before training')
