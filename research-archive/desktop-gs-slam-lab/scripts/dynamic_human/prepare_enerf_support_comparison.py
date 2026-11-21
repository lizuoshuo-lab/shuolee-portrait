"""Reuse the audited native comparison with explicit original/corrected roots."""
from pathlib import Path
import json
L=Path(__file__).resolve().parents[2];D=L/'scripts/dynamic_human';O=L/'outputs/dynamic-human-enerf-support-correction-081'
s=(D/'compare_enerf_second_interval.py').read_text()
s=s.replace("O=R/'comparison';O.mkdir(exist_ok=True)","T=L/'outputs/dynamic-human-enerf-support-correction-081';O=T/'comparison';O.mkdir(exist_ok=True)")
s=s.replace("branches=['global_cubic','local_bspline'];labels=['Original photograph','Global cubic','Cubic + local B-splines']","branches=['original_support','corrected_support'];roots={branches[0]:R/'local_bspline',branches[1]:T/'local_bspline'};labels=['Original photograph','Original training support','Conservative corrected support']")
s=s.replace("R/b/'", "roots[b]/'").replace("R/br/name", "roots[br]/name")
s=s.replace('rows=[];panels=[];transitions=[];previous=None;', 'rows=[];panels=[];transitions=[];detail_scores=[];previous=None;')
old="for r in detail.get((c,f),[]):panel(c,f,r['name'],r['bbox'],arrays)"
new="""for r in detail.get((c,f),[]):
  panel(c,f,r['name'],r['bbox'],arrays)
  xx0,yy0,xx1,yy1=r['bbox'];roi=np.zeros_like(valid);roi[yy0:yy1,xx0:xx1]=True;m=roi&valid&(matte>.5);vals={}
  for b in branches:
   d=predictions[b].astype(np.float64)[m]-ref.astype(np.float64)[m];vals[b]=dict(psnr_db=float(-10*np.log10(np.mean(d*d))),mae=float(abs(d).mean()))
  detail_scores.append(dict(camera=c,frame=f,name=r['name'],bbox=r['bbox'],split=inputs[c,f]['split'],pixels=int(m.sum()),metrics=vals,change_db=vals[branches[1]]['psnr_db']-vals[branches[0]]['psnr_db']))"""
assert old in s;s=s.replace(old,new)
s=s.replace("save(O/'records.json',dict(records=rows));", "save(O/'detail-metrics.json',dict(records=detail_scores,scope='Original079 frozen source-defined windows and original matte; context windows, not exact hand/foot semantic masks.'));save(O/'records.json',dict(records=rows));")
s=s.replace("Same-scene new motion interval, one seed; all120pairs evaluated, no upscaling/sharpening. PSNR or temporal residual improvement is not clearance of hands/shoes/occlusion failures. Fullbackground integration still incomplete.","Same079 interval, local B-spline representation and one seed. Only training support changed in63images;42 retain original masks. Same initial Gaussian/graph/body/cameras/crops/update budget/schedule. Evaluation RGB/mattes/windows unchanged. Partial background correction, not semantic truth. Clarity must be visually assessed; no fullbackground integration.")
assert "R/b/" not in s and "R/br/" not in s
(D/'compare_enerf_second_support.py').write_text(s,encoding='utf8')
save=lambda p,x:p.write_text(json.dumps(x,indent=2)+'\n')
save(O/'evaluation-plan.json',dict(status='frozen_while_training_before_candidate_render',reference='079/local_bspline',candidate='081/local_bspline',full_and_detail_windows='079/review-plan/protocol.json, reused unchanged',evaluation='All120 native images,90knownodd+18otherodd+12evendiagnostic; original RGB/mattes and metric code unchanged',additional_detail_scores='Original foreground intersect existing24hand/object/foot windows; no new mask or test-based tuning',goal_complete=False))
p=L/'manifests/dynamic-human-work.json';m=json.loads(p.read_text(encoding='utf8'));m.update(status='conservative_training_support_matched_intervention_running',current_runs=['outputs/dynamic-human-enerf-support-correction-081/local_bspline'],outdoor_support_correction='outputs/dynamic-human-enerf-support-correction-081/evaluation-plan.json',next_action='Complete15000 matched updates and native120-image evaluation. Compare original fixed metrics plus existing24detail windows; inspect real images before accepting change.',goal_turn_classification='Progress: generated105training-only segmentation candidates, visually inspected all105; rejected6occluded cameras, protected original hands/toys/feet, approved63partial corrections; actual matched training started.')
save(p,m);print('Prepared matched comparison and running-state manifest')
