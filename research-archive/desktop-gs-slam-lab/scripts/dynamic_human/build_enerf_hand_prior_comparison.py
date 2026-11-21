from pathlib import Path
L=Path(__file__).resolve().parents[2];D=L/'scripts/dynamic_human'
s=(D/'compare_enerf_hand_trajectory.py').read_text()
def once(a,b):
 global s
 assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
once("N=L/'outputs/dynamic-human-enerf-hand-trajectories-095'","N=L/'outputs/dynamic-human-enerf-hand-prior-control-098'")
once("roots=dict(parent=T/'parent',frozen=T/'multi_time',nodes=N/'nodes',points=N/'points')","roots=dict(parent=T/'parent',frozen=T/'multi_time',nodes=L/'outputs/dynamic-human-enerf-hand-trajectories-095/points',points=N/'soft_prior')")
s=s.replace("'nodes'","'free_motion'").replace("'points'","'soft_prior'").replace('nodes=','free_motion=').replace('points=N/','soft_prior=N/').replace('points_vs_nodes','soft_prior_vs_free_motion')
once("labels=['Original','Parent','Fixed motion','Node motion','Point motion']","labels=['Original','Parent','Fixed motion','Free motion','Soft hand prior']")
once("scope='Same-scene continuous localtranslation diagnostic. All105trainimages, original134evaluationcontexts; no newview/timedependent calibration. Frozencontrol088, matched nodeversuspoint095. Nojointstaticbackground ornewsceneacceptance.'","scope='Same-scene105trainphoto3000updates soft MANO control; matched095freepoint motion and088frozenmotion.1937soft-associatedGaussians, nohardtemplate substitution. Actual134native evaluation;nojointbackground ornewsceneacceptance.'")
(D/'compare_enerf_hand_prior_control.py').write_text(s,encoding='utf8');print('098fourmodel native comparison script ready')
