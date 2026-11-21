from pathlib import Path
import json
L=Path(__file__).resolve().parents[2];D=L/'scripts/dynamic_human';O=L/'outputs/dynamic-human-enerf-hand-snapshot-091';O.mkdir(exist_ok=True)
src=(D/'train_enerf_hand_time_scope.py').read_text(encoding='utf8')
src=src.replace('Matched frozen-motion canonical-surface diagnostic: one time vs all seven.','Shared local world-snapshot control; not a recovered capture-time model.')
src=src.replace('from enerf_hand_joint_model import HandJointRenderer','from enerf_hand_snapshot_model import HandSnapshotRenderer as HandJointRenderer')
src=src.replace('dynamic-human-enerf-hand-time-scope-088','dynamic-human-enerf-hand-snapshot-091').replace('hand_frozen_scope_088','hand_shared_snapshot_091')
src=src.replace("choices=['single_time','multi_time']","choices=['single_time']")
src=src.replace("Allbody,capturetime,coarsegraphmotion andbindingsfixedto079; nofinegraph.","Selected6330material uses04frame732geometry, covariance andSHframe inallviews. Othermaterial retainsoriginalcapturetime. Allmotionparametersfrozen.")
src=src.replace("Canonicalshape/appearance diagnostic, notcontinuoussolution. LargerGaussianLRthan084/085,matchedbetween088branches; cannotcompareto085asone-factorablation.","Counterfactual single-snapshot diagnostic. Samecanonicalinitialparams,schedule,loss,LR,steps as088single_time. Renderedinitialgeometry changesoutside04bydesign. Not a timingcalibration orcontinuoussolution.")
src=src.replace("18nativeviewsat732with00/06/12unseen; original120casesforcross-timecost ifanyfitgain. Sourcecropsfixedbeforetraining, noheldoutselection.","All18nativeviewsat732;15train3unseen;originalsourcehandwindows. No clockfit ornewgeometrytruth. Sharedposeassumption may bewrong.")
src=src.replace('start=0;history=[];elapsed=0', '''from enerf_hand_joint_model import HandJointRenderer as OriginalRenderer
original=OriginalRenderer(S/'head/capture-affines.npz',g,ref,active)
with torch.no_grad():
    o=obs['04',732];aa,ab,_=model.render(p,'04',732,o['W'],o['K'],o['width'],o['height'],2);ba,bb,_=original.render(p,'04',732,o['W'],o['K'],o['width'],o['height'],2);assert torch.equal(aa,ba) and torch.equal(ab,bb)
    reference_geometry=model.geometry(p,'04',732)
    for camera_check in cams:
        new=model.geometry(p,camera_check,732);old=original.geometry(p,camera_check,732)
        for k in range(3):assert torch.equal(new[k][active],reference_geometry[k][active]) and torch.equal(new[k][~active],old[k][~active])
save(R/'renderer-preflight.json',dict(camera04_RGB_andalpha_exact=True,all15views_selectedgeometry_identical=True,nonselectedgeometry_exact=True,scope='No sourcecalibration orcapturetimevalues changed. Activegeometry snapshotonly.'))
start=0;history=[];elapsed=0''')
(D/'train_enerf_hand_snapshot.py').write_text(src,encoding='utf8',newline='\n')
src=(D/'evaluate_enerf_hand_view_scope.py').read_text(encoding='utf8').replace('dynamic-human-enerf-hand-view-scope-089','dynamic-human-enerf-hand-snapshot-091').replace("choices=['single_camera']","choices=['single_time']").replace('hand_frozen_view_089','hand_shared_snapshot_091').replace('from enerf_hand_joint_model import HandJointRenderer','from enerf_hand_snapshot_model import HandSnapshotRenderer as HandJointRenderer')
(D/'evaluate_enerf_hand_snapshot.py').write_text(src,encoding='utf8',newline='\n')
(D/'run_enerf_hand_snapshot.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/surface-gs/env.sh
L=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
python "$L/scripts/dynamic_human/train_enerf_hand_snapshot.py" --branch single_time
python "$L/scripts/dynamic_human/evaluate_enerf_hand_snapshot.py" --branch single_time
''',encoding='utf8',newline='\n')
(O/'protocol.json').write_text(json.dumps(dict(question='Does forcing one common hand-world snapshot improve15-view fit versus088camera-specific inheritedmotion? Counterfactual, notclocktruth.',parent='079',baseline='088single_time',train_cameras=15,frame=732,heldout=['00','06','12'],steps=3000,active=6330,onlychange='selectedgeometry/SHframe uses04frame732for everycamera',sampling_loss_LR_points_same_as088=True,initial_rendered_geometry_changes=True,independent_capture_timing_not_recovered=True,goal_complete=False),indent=2)+'\n',encoding='utf8')
print('091 shared hand snapshot prepared with reference and frozen-geometry preflight')
