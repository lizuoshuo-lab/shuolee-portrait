from pathlib import Path
import json
D=Path(__file__).parent
def renamed(s):
 return s.replace('dynamic-human-enerf-hand-visibility-100','dynamic-human-enerf-contour-surface-103').replace('narrow_reset','prior_control').replace('context_reset','contour_prior').replace('photo_pose','previous')
labels="['Original','Parent','Previous model','Prior control','Contour prior']"
s=renamed((D/'compare_enerf_hand_visibility.py').read_text(encoding='utf8')).replace('dynamic-human-enerf-mano-surface-099/previous','dynamic-human-enerf-hand-visibility-100/context_reset').replace("['Original','Parent','Prior surface','Narrow reset','Context reset']",labels)
old='Same105photos and6000updates. Narrow1937 vs context6330 reset differ only in4393additional original opacity initializations. Prior099surface is the earliernew-opacity0.02reference, bothnewbranchesstart0.2. No hard deletion; all6330oldopacities can relearn. Selection geometry/support is heuristic; unchanged nativeheldout protocol;not independentnewscene acceptance.'
assert old in s;s=s.replace(old,'Same105photos/6000updates. Both103branchesmaintain0.02oldcontextalphaceilingandfixed102MANOpriors;pairdiffersonly102keypoint/contourprior. 100context_resetisanearlierreferencewithfreeoldalphasandphotoposecorrections;itsdifferenceisnotapurepriorablation. Original079andheldouttargetsunchanged;developmentonly,notindependentnewsceneacceptance.')
(D/'compare_enerf_contour_surface.py').write_text(s,encoding='utf8')
s=renamed((D/'render_enerf_hand_visibility_temporal.py').read_text(encoding='utf8')).replace("Q=L/'outputs/dynamic-human-enerf-mano-surface-099'","Q=L/'outputs/dynamic-human-enerf-hand-visibility-100'")
s=s.replace("src=Q/'temporal'/branch/","src=Q/'temporal'/('context_reset' if branch=='previous' else branch)/")
old="ManoGaussianSurface(H/'all_views/parameters.npz',S/'heldout/capture-affines-estimated.npz',cp['scene_scale'],read(O/'protocol.json')['initial_color'],True)"
assert old in s;s=s.replace(old,"ManoGaussianSurface(L/read(O/'protocol.json')['priors'][branch]['path'],S/'heldout/capture-affines-estimated.npz',cp['scene_scale'],read(O/'protocol.json')['initial_color'],False)").replace('reused099','reused100')
(D/'render_enerf_contour_surface_temporal.py').write_text(s,encoding='utf8')
s=renamed((D/'audit_enerf_hand_visibility_temporal.py').read_text(encoding='utf8')).replace("['Original','Parent','Prior surface','Narrow reset','Context reset']",labels)
(D/'audit_enerf_contour_surface_temporal.py').write_text(s,encoding='utf8')
s=(D/'review_enerf_hand_visibility.py').read_text(encoding='utf8').replace('dynamic-human-enerf-hand-visibility-100','dynamic-human-enerf-contour-surface-103')
(D/'review_enerf_contour_surface.py').write_text(s,encoding='utf8')
print('Prepared independent103comparison and26contexttemporalchecks; no executionyet.')
