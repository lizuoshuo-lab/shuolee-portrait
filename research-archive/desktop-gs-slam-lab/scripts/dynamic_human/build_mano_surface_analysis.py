from pathlib import Path
P=Path(__file__).parent
s=(P/'compare_enerf_hand_prior_control.py').read_text(encoding='utf8')
s=s.replace('dynamic-human-enerf-hand-prior-control-098','dynamic-human-enerf-mano-surface-099')
s=s.replace("branches=['parent','frozen','free_motion','soft_prior'];roots=dict(parent=T/'parent',frozen=T/'multi_time',free_motion=L/'outputs/dynamic-human-enerf-hand-trajectories-095/points',soft_prior=N/'soft_prior')", "branches=['parent','soft_prior','fixed_pose','photo_pose'];roots=dict(parent=T/'parent',soft_prior=L/'outputs/dynamic-human-enerf-hand-prior-control-098/soft_prior',fixed_pose=N/'fixed_pose',photo_pose=N/'photo_pose')")
s=s.replace("labels=['Original','Parent','Fixed motion','Free motion','Soft hand prior']", "labels=['Original','Parent','Soft prior (3k)','Fixed pose (6k)','Photo pose (6k)']")
s=s.replace('soft_prior_vs_free_motion_db','photo_pose_vs_fixed_pose_db')
s=s.replace("metrics['soft_prior']['foreground']['psnr_db']-metrics['free_motion']['foreground']['psnr_db']", "metrics['photo_pose']['foreground']['psnr_db']-metrics['fixed_pose']['foreground']['psnr_db']")
s=s.replace("vals['soft_prior']['psnr_db']-vals['free_motion']['psnr_db']", "vals['photo_pose']['psnr_db']-vals['fixed_pose']['psnr_db']")
old="schedules=[dict(np.load(roots[br]/'schedule.npz')) for br in branches[1:]];assert all(a.keys()==schedules[0].keys() and all(np.array_equal(a[k],schedules[0][k]) for k in ['camera','frame','background']) for a in schedules)"
new="schedules={b:dict(np.load(roots[b]/'schedule.npz')) for b in branches[1:]};assert all(np.array_equal(schedules['fixed_pose'][k],schedules['photo_pose'][k]) for k in ['camera','frame','background']);assert all(np.array_equal(schedules['fixed_pose'][k][j:j+3000],schedules['soft_prior'][k]) for j in [0,3000] for k in ['camera','frame','background'])"
assert old in s;s=s.replace(old,new)
s=s.replace("identical_initialization_cameras_background=True,identical_training_times=True", "identical_parent_cameras_training_times=True,matched_pair=['fixed_pose','photo_pose'],steps={'parent':0,'soft_prior':3000,'fixed_pose':6000,'photo_pose':6000}")
s=s.replace('Same-scene105trainphoto3000updates soft MANO control; matched095freepoint motion and088frozenmotion.1937soft-associatedGaussians, nohardtemplate substitution. Actual134native evaluation;nojointbackground ornewsceneacceptance.', 'Same-scene105trainphotos. Fixed/refined pose are the matched6000update pair, each1538articulated surfaceGaussians plus6330oldalpha. Parent and098softprior3000 are references with different optimization budgets. Actual134native evaluation;notnewsceneorjointbackgroundacceptance.')
assert 'free_motion' not in s
(P/'compare_enerf_mano_surface.py').write_text(s,encoding='utf8')
s=(P/'audit_enerf_hand_prior_temporal.py').read_text(encoding='utf8').replace('dynamic-human-enerf-hand-prior-control-098','dynamic-human-enerf-mano-surface-099').replace("branches=['parent','frozen','free_motion','soft_prior']", "branches=['parent','soft_prior','fixed_pose','photo_pose']").replace("labels=['Original','Parent','Fixed motion','Free motion','Soft hand prior']", "labels=['Original','Parent','Soft prior (3k)','Fixed pose (6k)','Photo pose (6k)']")
(P/'audit_enerf_mano_surface_temporal.py').write_text(s,encoding='utf8')
s=(P/'review_enerf_hand_prior_control.py').read_text(encoding='utf8')
s=s.replace("r=next(r for r in p if r['camera']==c and r['frame']==f and r['name'] in ['full-person','worst-full-person']);path=", "r=next((r for r in p if r['camera']==c and r['frame']==f and r['name'] in ['full-person','worst-full-person']),None)\n if r is None:continue\n path=")
(P/'review_enerf_hand_prior_control.py').write_text(s,encoding='utf8')
(P/'review_enerf_mano_surface.py').write_text(s.replace('dynamic-human-enerf-hand-prior-control-098','dynamic-human-enerf-mano-surface-099'),encoding='utf8')
print('Wrote 099 comparison, temporal audit, and native review inventory')
