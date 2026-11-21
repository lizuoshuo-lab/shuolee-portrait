"""Prepare a fixed-prior paired GS trial with persistent old-context alpha caps."""
from pathlib import Path
import json,hashlib,shutil
L=Path(__file__).resolve().parents[2];D=Path(__file__).parent;Q=L/'outputs/dynamic-human-hand-contour-prior-102';O=L/'outputs/dynamic-human-enerf-contour-surface-103';P=L/'outputs/dynamic-human-enerf-hand-visibility-100';O.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'));save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (O/'protocol.json').exists();evaluation=read(Q/'evaluation/summary.json');assert evaluation['source_parameters_unchanged']
files=[Q/'evaluation'/r['file'] for r in evaluation['panels']];assert len(files)==11
save(Q/'visual-review.json',dict(status='all22native_source_contour_comparisons_inspected',files=[dict(file=str(p.relative_to(Q)),sha256=sha(p)) for p in files],observations=['Candidate coverage better contains sourcehandparts on someviews including04/13maskdiagnostics; differencesaremodest.','Forearmcutisnot supervised;originalproximalwristcontoursremainimperfect.','Surrogate contours stillmerge narrowfingersanddonotprove correctdepth, texture orclearGSreconstruction.'],qualified_for_paired_native_RGB_trial=True,geometry_accepted_as_GT=False,clarity_accepted=False,goal_complete=False))
save(Q/'final-summary.json',dict(status='paired1500step_contour_prior_qualification_complete_RGBtrial_required',training={b:read(Q/b/'summary.json') for b in ['keypoint_control','contour_candidate']},aggregate=evaluation['aggregate'],candidate_keypoint_residual=evaluation['candidate_keypoint_residual'],mask_fit_contexts=18,mask_diagnostic_contexts=4,actual_source_contexts_reviewed=22,Gaussian_updates=False,licensed_geometry_kept_local=True,clarity_accepted=False,goal_complete=False,next='Matched nativeGSappearance trial with fixedcontrol/candidatecontinuouspriors andidenticalpersistentoldcontextalphacap; preserve original079model.'))
proto=read(P/'protocol.json');proto.update(status='fixed_before_contour_surface_trial',stage='103',branches=['prior_control','contour_prior'],seed=10301,steps_each=6000,initial_new_opacity=.2,persistent_old_opacity_cap=.02,old_opacity_cap_count=6330,
    priors={b:dict(path='outputs/'+Q.name+'/'+src+'/parameters.npz',sha256=sha(Q/src/'parameters.npz')) for b,src in [('prior_control','keypoint_control'),('contour_prior','contour_candidate')]},
    difference='Onlythe102continuousMANOprior differs. Both1500steppriorsusedidentical105softkeypointinputs;candidate additionallyused18weaktrainingmasks. No newmasklossinGSappearancefit.',
    fixed='Alloriginalgeometry,color,motion,camera,time and246654noncontextopacities. MANOpose,translation,shapeandscale fixedtoselectedprior throughoutRGBtraining.',
    learned='New1538GaussianRGB,opacity,trianglelocaloffsetandscale;6330oldcontextalphalearnedwithpersistentceiling0.02aftereachstep.',
    common_intervention='Capall6330oldcontextalphasat0.02forbothbranches. Thismayremovearm/toycontributions;mustchecknativeRGBandheldoutcontexts. Itisaboundedcloneexperiment,notdeletionoforiginalGS.',
    comparison_scope='Pairisolatesfixedpriorchoiceundercommonpersistentcap. Versus100historicalmodelalsodiffersinposeoptimizationandoldalphacap,sonotapurepriorablationagainst100.',
    evaluation='Original134native1920x1080cases; requirebody,hand,toyandcontinuousframechecks beforeadoption.',goal_complete=False)
for k in ['reset_counts','initialization','initial_reset_count','reset_count','selection','base_099_comparison']:
 proto.pop(k,None)
save(O/'protocol.json',proto);shutil.copy2(P/'context-reset-mask.npy',O/'context-reset-mask.npy')
def change(s,old,new,count=None):
 if count is not None:assert s.count(old)==count,(old,s.count(old))
 else:assert old in s,old
 return s.replace(old,new)
s=(D/'train_enerf_hand_visibility.py').read_text(encoding='utf8')
s=change(s,'dynamic-human-enerf-hand-visibility-100','dynamic-human-enerf-contour-surface-103');s=change(s,"choices=['narrow_reset','context_reset']","choices=['prior_control','contour_prior']",1);s=change(s,"torch.manual_seed(10001)","torch.manual_seed(10301)",1)
s=change(s,"ManoGaussianSurface(H/'all_views/parameters.npz',S/'heldout/capture-affines-estimated.npz',cp['scene_scale'],protocol['initial_color'],True)","ManoGaussianSurface(L/protocol['priors'][args.branch]['path'],S/'heldout/capture-affines-estimated.npz',cp['scene_scale'],protocol['initial_color'],False)",1)
s=change(s,"reset=torch.from_numpy(np.load(O/('reset-mask.npy' if args.branch=='narrow_reset' else 'context-reset-mask.npy'))).cuda();assert read(O/'selection/visual-review.json')['approved_for_bounded_trial']","reset=active;assert read(O/'smoke-visual-review.json')['ready_for_training'];assert sha(L/protocol['priors'][args.branch]['path'])==protocol['priors'][args.branch]['sha256']",1)
s=change(s,'mano_surface100','mano_surface103');s=change(s,"opt.step();opt.zero_grad(set_to_none=True);oldopt.step();oldopt.zero_grad(set_to_none=True);factor=(1/3)**((i+1)/6000)","opt.step();opt.zero_grad(set_to_none=True);oldopt.step();oldopt.zero_grad(set_to_none=True)\n with torch.no_grad():p['opacities'][active]=p['opacities'][active].clamp_max(float(np.log(.02/.98)))\n factor=(1/3)**((i+1)/6000)",1)
s=change(s,'"""Two matched native-RGB trials: articulated surface with fixed/refined hand pose."""','"""Matched fixed-prior nativeRGB trials with persistent oldcontextalphacaps."""',1)
(D/'train_enerf_contour_surface.py').write_text(s,encoding='utf8')
s=(D/'evaluate_enerf_hand_visibility.py').read_text(encoding='utf8');s=change(s,'dynamic-human-enerf-hand-visibility-100','dynamic-human-enerf-contour-surface-103');s=change(s,"choices=['narrow_reset','context_reset']","choices=['prior_control','contour_prior']",1);s=change(s,'mano_surface100','mano_surface103');s=change(s,"ManoGaussianSurface(H/'all_views/parameters.npz',S/'heldout/capture-affines-estimated.npz',cp['scene_scale'],protocol['initial_color'],True)","ManoGaussianSurface(L/protocol['priors'][args.branch]['path'],S/'heldout/capture-affines-estimated.npz',cp['scene_scale'],protocol['initial_color'],False)",1);s=change(s,"+(['pose_delta','translation_delta'] if True else [])","",1)
s=s.replace("assert all(torch.equal(v,cp['graph'][k])", "assert float(cp['params']['opacities'][active].sigmoid().max())<=.020001\nassert all(torch.equal(v,cp['graph'][k])")
(D/'evaluate_enerf_contour_surface.py').write_text(s,encoding='utf8')
shutil.copy2(__file__,O/Path(__file__).name);print('Prepared103pairedtrainingandevaluationprograms; originalmodels unchanged.')
