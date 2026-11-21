from pathlib import Path
L=Path(__file__).resolve().parents[2];S=L/'scripts/dynamic_human'
text=(S/'compare_enerf_second_support.py').read_text(encoding='utf8')
text=text.replace("dynamic-human-enerf-support-correction-081","dynamic-human-enerf-open-hand-flow-083")
text=text.replace("branches=['original_support','corrected_support'];roots={branches[0]:R/'local_bspline',branches[1]:T/'local_bspline'};labels=['Original photograph','Original training support','Conservative corrected support']","branches=['photo_control','normal_flow'];roots={b:T/b for b in branches};labels=['Original photograph','Photo-only continuation','Hand normal-flow continuation']")
text=text.replace("[('trained',u8(pred)),('source',rgb),('reference',u8(ref)),('alpha',u8(p['alpha']))]","[('trained',u8(pred)),('alpha',u8(p['alpha']))]")
text=text.replace('imagechecks==960','imagechecks==480').replace('120paired_native_predictions_960PNGs','120paired_native_predictions_480PNGs')
old="for name in ['initial-and-sampling-audit.json']:\n a,b=[read(roots[br]/name) for br in branches];assert a==b"
new="a,b=[dict(np.load(roots[br]/'schedule.npz')) for br in branches];assert a.keys()==b.keys() and all(np.array_equal(a[k],b[k]) for k in a)\nfa,fb=[read(roots[br]/'frozen-state-audit.json') for br in branches];assert fa['parent_sha256']==fb['parent_sha256'] and fa['active_nodes']==fb['active_nodes']==27\nparent_report=read(R/'local_bspline/native-evaluation/metrics.json')['records'];parent_scores={split:float(np.mean([r['metrics']['foreground']['psnr_db'] for r in parent_report if r['split']==split])) for split in ['train','validation_time','validation_camera']}"
assert old in text;text=text.replace(old,new)
text=text.replace('identical_initialization_and_sampling=True,max_recomputed_metric_difference','identical_initialization_and_sampling=True,parent_foreground_psnr_db=parent_scores,parent_metrics_sha256=sha(R/\'local_bspline/native-evaluation/metrics.json\'),max_recomputed_metric_difference')
old='Same079 interval, local B-spline representation and one seed. Only training support changed in63images;42 retain original masks. Same initial Gaussian/graph/body/cameras/crops/update budget/schedule. Evaluation RGB/mattes/windows unchanged. Partial background correction, not semantic truth. Clarity must be visually assessed; no fullbackground integration.'
new='Same079parent,1500additionalgraph-onlyupdates,27activegraphnodes. Candidate addsqualifiedhandnormalflow; controlhaszeroflowweight. AllGaussianparameters/body/cameras/clocks/targets/samplingfrozen andmatched. Original120evaluationcontextsanddetailwindowsunchanged. Groundtruthflow unavailable; actualRGBclaritymustbevisuallyassessed. Nojointfullbackground.'
assert old in text;text=text.replace(old,new)
path=S/'compare_enerf_hand_flow_control.py';assert not path.exists();path.write_text(text,encoding='utf8');print('Generated paired CPU checker with480nativePNG checks andparentreference')
