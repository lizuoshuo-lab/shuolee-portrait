from pathlib import Path
D=Path(__file__).resolve().parent
s=(D/'compare_enerf_hand_view_scope.py').read_text(encoding='utf8').replace('dynamic-human-enerf-hand-view-scope-089','dynamic-human-enerf-hand-snapshot-091').replace('single_camera','shared_snapshot')
s=s.replace("shared_snapshot=O/'shared_snapshot'","shared_snapshot=O/'single_time'").replace("O/'shared_snapshot'","O/'single_time'")
s=s.replace("O/'shared_snapshot/","O/'single_time/")
s=s.replace("'15-view fit','1-view fit'","'Capture-time motion','Shared snapshot'").replace("fitted_in089=c=='04'","training_camera=c not in ['00','06','12']")
s=s.replace("set(b['camera'])=={'04'}","np.array_equal(a['camera'],b['camera'])").replace('only_camera_sampling_changes=True','only_selected_capture_geometry_changes=True')
s=s.replace("'fitted_camera'","'camera04_reference'").replace("'other14_known_cameras'","'other14_training_cameras'")
(D/'compare_enerf_hand_snapshot.py').write_text(s,encoding='utf8',newline='\n')
print('091 comparison prepared against088matched15viewfit')
