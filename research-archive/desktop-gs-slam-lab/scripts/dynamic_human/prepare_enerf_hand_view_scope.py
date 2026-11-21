from pathlib import Path
import json
L=Path(__file__).resolve().parents[2];D=L/'scripts/dynamic_human';O=L/'outputs/dynamic-human-enerf-hand-view-scope-089';O.mkdir(exist_ok=True)
src=(D/'train_enerf_hand_time_scope.py').read_text(encoding='utf8')
src=src.replace('Matched frozen-motion canonical-surface diagnostic: one time vs all seven.','Single training image fit diagnostic; never a multiview reconstruction claim.')
src=src.replace('dynamic-human-enerf-hand-time-scope-088','dynamic-human-enerf-hand-view-scope-089').replace('hand_frozen_scope_088','hand_frozen_view_089').replace('single_time','single_camera')
src=src.replace("choices=['single_camera','multi_time']","choices=['single_camera']")
src=src.replace("if args.branch=='single_camera' and r['frame']!=732:continue","if r['frame']!=732 or r['camera']!='04':continue")
src=src.replace("camera=[cams[cr.randrange(15)] for _ in range(steps)]","camera=['04']*steps")
src=src.replace("Same3000cameras andbackgrounddraws. Single_timeonly732;multi_time7evenframes from720..732.","Onefixedtrainingimage camera04frame732;3000updates.Samebackgrounddraws as088.")
src=src.replace("Canonicalshape/appearance diagnostic, notcontinuoussolution. LargerGaussianLRthan084/085,matchedbetween088branches; cannotcompareto085asone-factorablation.","Capacity/optimization diagnostic only. Same6330points,parent,LR,loss,steps,backgrounds as088; only camera sampling changed to04. No heldout learning or model promotion.")
src=src.replace("18nativeviewsat732with00/06/12unseen; original120casesforcross-timecost ifanyfitgain. Sourcecropsfixedbeforetraining, noheldoutselection.","18nativeviews at732, source-defined160px handcrops. Cross-view costs explicit. Singleimage cannot establish depth or dynamic surface correctness.")
(D/'train_enerf_hand_view_scope.py').write_text(src,encoding='utf8',newline='\n')
src=(D/'evaluate_enerf_hand_time_scope.py').read_text(encoding='utf8')
src=src.replace("O=L/'outputs/dynamic-human-enerf-hand-time-scope-088'","U=L/'outputs/dynamic-human-enerf-hand-time-scope-088';O=L/'outputs/dynamic-human-enerf-hand-view-scope-089'")
src=src.replace("choices=['parent','single_time','multi_time']","choices=['single_camera']").replace('hand_frozen_scope_088','hand_frozen_view_089')
src=src.replace("records=read(O/'evaluation-protocol.json')['records']","records=[r for r in read(U/'evaluation-protocol.json')['records'] if r['frame']==732]")
src=src.replace("(O if r.get('input_location')=='088' else S)","(U if r.get('input_location')=='088' else S)").replace('134native_contexts_complete_comparison_pending','18native_contexts_complete_comparison_pending')
(D/'evaluate_enerf_hand_view_scope.py').write_text(src,encoding='utf8',newline='\n')
(D/'run_enerf_hand_view_scope.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/surface-gs/env.sh
L=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
python "$L/scripts/dynamic_human/train_enerf_hand_view_scope.py" --branch single_camera
python "$L/scripts/dynamic_human/evaluate_enerf_hand_view_scope.py" --branch single_camera
''',encoding='utf8',newline='\n')
(O/'protocol.json').write_text(json.dumps(dict(question='Does existing fixed-capacity hand material fit a single sharp training image? Separate within-view fitting from cross-view compromise before new surface initialization.',training_camera='04',frame=732,steps=3000,active_Gaussians=6330,all_motion_frozen=True,all_other_Gaussians_frozen=246654,comparison='079parent and088single_time15cameras; same initialization/loss/LR/background draw/updates; onlycamera sampling changes',evaluation='18native1920x1080at732 and15source160pxhandwindows; heldout00/06/12 nevertrained',not_a_final_solution=True),indent=2)+'\n',encoding='utf8')
print('089 single-image diagnostic prepared; other motion/material preserved')
