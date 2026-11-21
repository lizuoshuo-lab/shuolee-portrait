"""Prepare an isolated replay of the existing body/head fit with corrected RGB observations."""
from pathlib import Path
import json,hashlib
L=Path(__file__).resolve().parents[2];S=L/'scripts/dynamic_human';O=L/'outputs/dynamic-human-enerf-rgb-pose-correction-074'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
assert read(O/'preprocessing-audit.json')['corrected_normalized_tensor_max_abs_difference_from_official']==0
assert read(O/'summary.json')['source_unchanged']
assert (O/'estimated_offsets-fit.npz').is_file()
for source,dest,replacements in [
 ('fit_enerf_continuous_body.py','fit_enerf_rgb_body.py',{
  "SOURCE=LAB/'outputs/dynamic-human-enerf-pose-sequence-062'":"SOURCE=LAB/'outputs/dynamic-human-enerf-rgb-pose-correction-074'",
  "OUT=LAB/'outputs/dynamic-human-enerf-continuous-body-063'":"OUT=LAB/'outputs/dynamic-human-enerf-rgb-pose-correction-074/body'",
  "branches=['fixed_offsets','estimated_offsets']":"branches=['estimated_offsets']",
  "for label in ['fixed_offsets','estimated_offsets']:":"for label in ['estimated_offsets']:",
  "062 cubic3Djoint trajectories":"074 RGB-corrected cubic3Djoint trajectories",
  "both_continuous_body_motion_hypotheses_fitted":"RGB_corrected_estimated_offset_body_fitted",
 }),
 ('fit_enerf_head_motion.py','fit_enerf_rgb_head.py',{
  "SRC=LAB/'outputs/dynamic-human-enerf-continuous-body-063'":"SRC=LAB/'outputs/dynamic-human-enerf-rgb-pose-correction-074/body'",
  "SEQ=LAB/'outputs/dynamic-human-enerf-pose-sequence-062'":"SEQ=LAB/'outputs/dynamic-human-enerf-rgb-pose-correction-074'",
  "OUT=LAB/'outputs/dynamic-human-enerf-head-motion-066'":"OUT=LAB/'outputs/dynamic-human-enerf-rgb-pose-correction-074/head'",
  "exactly063estimated branch used by064 baseline":"exactly074 RGB-corrected estimated-offset body branch",
  "Same546training-only supported head observations as065":"Same546training camera/time/head indices as065; correctedRGB predictedcoordinates",
 })]:
 text=(S/source).read_text()
 for old,new in replacements.items():
  assert old in text,(source,old);text=text.replace(old,new)
 assert not (S/dest).exists();(S/dest).write_text(text)
receipt=dict(status='RGB_observation_body_head_refit_prepared',change='Correctmodelinputchannels change body/headpredictions and fittedcameraoffsets; treated as one preprocessing repair, not independentlyisolatedtimingandpose gains.',
 preserved='Original059camera/jointsupport,060shape/scale/canonicalmesh,fitframes,solverbudgets,body2500steps,head3000steps. Handandfingerobservationsnotadded.',
 limits='Old059supportand060canonicalreference derivedfromhistoricalBGRpredictions remainfrozenforthiscontrol. Repairisnotclaimofcompleteallupstreamrecalibration.',
 files={f:sha(S/f) for f in ['fit_enerf_rgb_pose_timing.py','fit_enerf_rgb_body.py','fit_enerf_rgb_head.py']},goal_complete=False)
(O/'motion-refit-protocol.json').write_text(json.dumps(receipt,indent=2)+'\n');print('Isolated motion refit prepared; no historical files changed.')
