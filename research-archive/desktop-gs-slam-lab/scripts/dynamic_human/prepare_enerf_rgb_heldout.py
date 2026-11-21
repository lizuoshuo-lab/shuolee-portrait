"""Prepare RGB-corrected clock-only calibration on the existing development views."""
from pathlib import Path
import json,shutil,hashlib
L=Path(__file__).resolve().parents[2];S=L/'scripts/dynamic_human';I=L/'outputs/dynamic-human-enerf-heldout-cameras-068';P=L/'outputs/dynamic-human-enerf-rgb-pose-correction-074';R=P/'heldout';R.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name in ['input-audit.json','person-boxes.json']:shutil.copy2(I/name,R/name)
p=read(I/'protocol.json');p.update(status='fixed_before_corrected_RGB_newcamera_observations',trajectory_sha256=sha(P/'estimated_offsets-fit.npz'),body_motion_sha256=sha(P/'head/motion-fit.npz'),
 clock='RGB-corrected21even-timeimages fitthreeclockoffsetsagainstfrozen074bodyfoottrajectories;39imagesobservedbutoddframesnotfit; noappearanceupdates.',
 branches=['074_corrected_RGB_pose_graph'],checkpoints={},scope='Same068developmentcameras00/06/12;notfreshunseentest.',goal_complete=False)
(R/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
for source,dest,replaces in [
 ('observe_enerf_heldout_cameras.py','observe_enerf_rgb_heldout.py',{
  "R=L/'outputs/dynamic-human-enerf-heldout-cameras-068'":"R=L/'outputs/dynamic-human-enerf-rgb-pose-correction-074/heldout';I=L/'outputs/dynamic-human-enerf-heldout-cameras-068'",
  "f=R/'native-input'":"f=I/'native-input'",
  'xy,s=pose(bgr,bboxes=':'xy,s=pose(np.ascontiguousarray(rgb),bboxes=',
 }),
 ('fit_enerf_heldout_clocks.py','fit_enerf_rgb_heldout_clocks.py',{
  "R=L/'outputs/dynamic-human-enerf-heldout-cameras-068'":"R=L/'outputs/dynamic-human-enerf-rgb-pose-correction-074/heldout'",
  "outputs/dynamic-human-enerf-pose-sequence-062/estimated_offsets-fit.npz":"outputs/dynamic-human-enerf-rgb-pose-correction-074/estimated_offsets-fit.npz",
 }),
 ('extend_enerf_frozen_camera_motion.py','extend_enerf_rgb_camera_motion.py',{
  "R=L/'outputs/dynamic-human-enerf-heldout-cameras-068'":"R=L/'outputs/dynamic-human-enerf-rgb-pose-correction-074/heldout';I=L/'outputs/dynamic-human-enerf-heldout-cameras-068'",
  "H=L/'outputs/dynamic-human-enerf-head-motion-066'":"H=L/'outputs/dynamic-human-enerf-rgb-pose-correction-074/head'",
  "B=L/'outputs/dynamic-human-enerf-continuous-body-063'":"B=L/'outputs/dynamic-human-enerf-rgb-pose-correction-074/body'",
  "outputs/dynamic-human-enerf-pose-sequence-062/estimated_offsets-fit.npz":"outputs/dynamic-human-enerf-rgb-pose-correction-074/estimated_offsets-fit.npz",
  "np.load(R/'native-input'":"np.load(I/'native-input'",
  "Frozen062body trajectories":"Frozen074RGBbody trajectories",
 }),
 ('render_enerf_forearm_heldout.py','render_enerf_rgb_heldout.py',{
  'dynamic-human-enerf-forearm-matte-repair-072':'dynamic-human-enerf-rgb-pose-correction-074',
  "E=R/'heldout/native-evaluation'":"M=R/'heldout';E=R/'heldout/native-evaluation'",
  "I/'capture-affines-estimated.npz'":"M/'capture-affines-estimated.npz'",
  "I/'motion-extension-audit.json'":"M/'motion-extension-audit.json'",
  'Frozen068same21even-image estimates, no refitting for072':'074RGB-corrected21even-image clockfit; noappearancefitting',
  'Evaluate072on the same18new-camera odd frames using frozen068clocks.':'Evaluate074on the same18development-view frames withRGB-corrected clocks.',
 })]:
 s=(S/source).read_text()
 for before,after in replaces.items():assert before in s,(source,before);s=s.replace(before,after)
 assert not (S/dest).exists();(S/dest).write_text(s)
print('Three heldout-view preprocessing/clock/extension/render scripts prepared; native inputs reused byte-for-byte.')
