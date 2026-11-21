from pathlib import Path
import json,numpy as np,hashlib
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-enerf-hand-capture-geometry-090';read=lambda p:json.loads(p.read_text(encoding='utf8'));s=read(O/'summary.json');z=np.load(O/'selected-geometry.npz');ref=z['reference_positions'].astype(np.float64);maxerr=0;vmaxerr=0
def project(m,W,K):
 x=m@W[:3,:3].T+W[:3,3];q=x@K.T;return q[:,:2]/q[:,2:]
for r in s['records']:
 c=r['camera'];W=z['W_'+c].astype(np.float64);K=z['K_'+c].astype(np.float64);m=z['positions_'+c].astype(np.float64);delta=project(ref,W,K)-project(m,W,K);err=float(abs(delta-z['projected_delta_'+c]).max());maxerr=max(maxerr,err);assert err<.002
 for q,v in r['all_selected_projected_shift_px'].items():assert abs(float(np.quantile(np.linalg.norm(delta,axis=-1),float(q)))-v)<.002
 if r['visible_hand_context']:
  p=np.load(O/f'camera{c}-visible-shift.npz');v=p['numerator'];mask=p['mask'];field=v[...,:3]/np.maximum(v[...,3:],1e-8);value=float(field[...,2][mask].mean());e=abs(value-r['visible_hand_context']['mean_absolute_projected_shift_px']);vmaxerr=max(vmaxerr,e);assert e<1e-6 and int(mask.sum())==r['visible_hand_context']['pixels']
rr=s['records'];out=dict(status='18camera_projections_and15visiblecontext_arrays_independently_verified_readonly',checkpoint_sha256=s['checkpoint_sha256'],max_projection_recompute_error_px=maxerr,max_visible_mean_recompute_error_px=vmaxerr,effective_capture_index_range=[min(r['effective_capture_index'] for r in rr),max(r['effective_capture_index'] for r in rr)],selected_examples={r['camera']:r for r in rr if r['camera'] in ['02','04','13','17']},interpretation='Inheritedcapturetimes atnominal732 spanmorethan3frames; modelimpliedhandprojectiondifference isnotnegligible. Itisnotobservedreprojectionerror. Camera17 near04time also failsin089, so timingaloneisnotproven. Nextsharedworldsnapshot fitisacounterfactual tocheckspatialcompatibility,notpermissiontooverwriteclocks.',goal_complete=False);(O/'final-summary.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8');print(json.dumps({k:v for k,v in out.items() if k!='selected_examples'}))
