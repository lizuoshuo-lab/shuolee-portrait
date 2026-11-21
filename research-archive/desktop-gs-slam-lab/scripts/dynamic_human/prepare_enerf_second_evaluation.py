"""Prepare fixed-clock new-camera motion and complete native evaluation tools."""
from pathlib import Path
import json,hashlib
L=Path(__file__).resolve().parents[2];S=L/'scripts/dynamic_human';R=L/'outputs/dynamic-human-enerf-second-interval-079';O=R/'heldout';O.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
clocks=L/'outputs/dynamic-human-enerf-rgb-pose-correction-074/heldout/timing-summary.json'
save(O/'protocol.json',dict(status='fixed_before_motion_extension_and_rendering',body_motion_sha256=sha(R/'head/motion-fit.npz'),
 clock_source_sha256=sha(clocks),clock_source='Freeze074camera00/06/12clocks estimated on744..756even images; no720..732other-camera keypoint or RGB fit.',
 interpretation='Same previously calibrated cameras, new motion interval; appearance withheld on00/06/12 throughout079. Not unseen-camera calibration or independent scene test.',
 offsets_frames=json.loads(clocks.read_text())['offsets_frames'],goal_complete=False))
text=(S/'extend_enerf_rgb_camera_motion.py').read_text();text=text[:text.index('co=np.load(')]
text=text.replace("R=L/'outputs/dynamic-human-enerf-rgb-pose-correction-074/heldout'","R=L/'outputs/dynamic-human-enerf-second-interval-079/heldout';R.mkdir(exist_ok=True)")
text=text.replace('dynamic-human-enerf-rgb-pose-correction-074/head','dynamic-human-enerf-second-interval-079/head').replace("B=L/'outputs/dynamic-human-enerf-rgb-pose-correction-074/body'","B=H")
text=text.replace("read(R/'timing-summary.json')['offsets_frames']","protocol['offsets_frames']").replace('ts-750','ts-726').replace('[744,750,756]','[720,726,732]').replace("['zero','estimated']","['estimated']").replace('744,757','720,733')
text+="\nshutil.copy2(__file__,R/Path(__file__).name);print(json.dumps(reports),flush=True)\n"
(S/'extend_enerf_second_motion.py').write_text(text)
text=(S/'evaluate_enerf_graph_control.py').read_text()
def once(a,b):
 global text
 assert text.count(a)==1,(a,text.count(a));text=text.replace(a,b,1)
once('from enerf_graph_renderer import GraphMotion,ResidualGraphRenderer','')
once("OUT=LAB/'outputs/dynamic-human-enerf-articulated-control-064'","OUT=LAB/'outputs/dynamic-human-enerf-second-interval-079'")
once("choices=['residual_graph']","choices=['global_cubic','local_bspline']")
once('torch.set_num_threads(8);cp=',"if a.branch=='global_cubic':\n from enerf_second_renderer import GraphMotion,ResidualGraphRenderer\nelse:\n from enerf_second_temporal_graph import GraphMotion,ResidualGraphRenderer\ntorch.set_num_threads(8);cp=")
once("assert sha(motion_file)==read(ROOT/'protocol.json')['motion_sha256'];", "source_motion=OUT/'head/capture-affines.npz';assert sha(source_motion)==read(ROOT/'protocol.json')['motion_sha256'];assert sha(motion_file)==read(OUT/'heldout/motion-extension-audit.json')['branches']['estimated']['sha256'];")
once("cp['motion_sha256']==sha(motion_file)","cp['motion_sha256']==sha(source_motion)")
once("['rotation_raw','translation_raw']","['rotation_raw','translation_raw','local_rotation_raw','local_translation_raw']")
once('assert gsteps==[15000,15000]','assert gsteps==([15000,15000] if a.branch==\'global_cubic\' else [15000,15000,15000,15000])')
text=text.replace('physicalframe750','physicalframe726')
once("r['split']=='validation_time' or (r['camera'] in ['01','04','08','17'] and r['frame'] in [744,750,756])","r['frame']%2==1 or (r['camera'] in ['01','04','08','17'] and r['frame'] in [720,726,732])")
once('assert len(records)==102','assert len(records)==120')
once("selection='All90odd-timevalidationimages plus12even-time diagnosticimages; 15knowntrainingcameras, no00/06/12.',images=102","selection='All90knowncamera odd-timevalidation,18othercamera odd-timevalidation,12even-time trainingdiagnostics.',images=120")
text=text.replace('Original062RGB','Original079RGB').replace('102nativeimages_complete','120nativeimages_complete').replace('Residual graph reconstruction',"Reconstruction: '+a.branch+'").replace("'ARTICULATED_EVAL'","'SECOND_INTERVAL_EVAL'")
(S/'evaluate_enerf_second_interval.py').write_text(text)
(S/'run_enerf_second_motion_extension.sh').write_text('#!/bin/bash\nset -euo pipefail\nsource /opt/gs-slam-lab/surface-gs/env.sh\npython /mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/scripts/dynamic_human/extend_enerf_second_motion.py\n',newline='\n')
(S/'run_enerf_second_evaluation.sh').write_text('''#!/bin/bash
set -euo pipefail
source /opt/gs-slam-lab/surface-gs/env.sh
L=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
for branch in global_cubic local_bspline; do
 python "$L/scripts/dynamic_human/evaluate_enerf_second_interval.py" --branch "$branch" --result-root "$L/outputs/dynamic-human-enerf-second-interval-079" --motion-file "$L/outputs/dynamic-human-enerf-second-interval-079/heldout/capture-affines-estimated.npz" 2>&1 | tee -a "$L/logs/enerf-second-interval-079-$branch-evaluation.log"
done
''',newline='\n')
compile(text,str(S/'evaluate_enerf_second_interval.py'),'exec')
print('120native views perbranch evaluator prepared; no new appearance rendered yet.')
