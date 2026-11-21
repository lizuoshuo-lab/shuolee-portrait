from pathlib import Path
L=Path(__file__).resolve().parents[2];D=L/'scripts/dynamic_human'
s=(D/'render_enerf_hand_trajectory_temporal.py').read_text()
def once(a,b):
 global s
 assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
s=s.replace('dynamic-human-enerf-hand-trajectories-095','dynamic-human-enerf-hand-prior-control-098')
once("for branch,old in [('parent','parent'),('frozen','multi_time')]:","for branch,old in [('parent','parent'),('frozen','multi_time'),('free_motion','points')]:")
once("src=U/'temporal'/old/f'{stem}.{suffix}'","src=(L/'outputs/dynamic-human-enerf-hand-trajectories-095/temporal' if branch=='free_motion' else U/'temporal')/old/f'{stem}.{suffix}'")
once("for branch in ['nodes','points']:","for branch in ['soft_prior']:")
once("HandTrajectory(branch,","HandTrajectory('points',")
s=s.replace('52new_native_ROIs_plus52exact_reused088','26new_native_ROIs_plus78exact_reused088_095')
(D/'render_enerf_hand_prior_temporal.py').write_text(s,encoding='utf8')
s=(D/'audit_enerf_hand_trajectory_temporal.py').read_text().replace('dynamic-human-enerf-hand-trajectories-095','dynamic-human-enerf-hand-prior-control-098').replace("'nodes'","'free_motion'").replace("'points'","'soft_prior'").replace("'Node motion'","'Free motion'").replace("'Point motion'","'Soft hand prior'")
(D/'audit_enerf_hand_prior_temporal.py').write_text(s,encoding='utf8')
s=(D/'audit_enerf_hand_trajectory_motion.py').read_text().replace('dynamic-human-enerf-hand-trajectories-095','dynamic-human-enerf-hand-prior-control-098')
once("for branch in ['nodes','points']:","for branch in ['soft_prior']:")
once("HandTrajectory(branch,","HandTrajectory('points',")
s=s.replace("cp['scene_scale'],branch)","cp['scene_scale'],'points')").replace('both234context_fields','one234context_field')
(D/'audit_enerf_hand_prior_motion.py').write_text(s,encoding='utf8');print('098temporal and motion audit scripts ready')
