"""Record explicit observed image decisions; does not decide by metrics alone."""
import argparse
from capture119_clothed import ROOT,REC,rt
p=argparse.ArgumentParser();p.add_argument('--subject',required=True);p.add_argument('--phase',choices=['fit','surface','train'],required=True);p.add_argument('--step',type=int,default=1000);p.add_argument('--observation',required=True);p.add_argument('--stop',action='store_true');a=p.parse_args()
root=ROOT/a.subject;folder=REC/f'{a.subject}-{a.phase}-{a.step:06d}-review';index=rt.read(folder/'index.json')
d=dict(created=rt.now(),subject=a.subject,step=a.step,reviewed_frames=[x['frame'] for x in index['images'] if x['region']=='body'],
    review_image_sha256=rt.sha(folder/'body-contact.png'),observations=a.observation,summary=a.observation,
    quality_accepted=False,physical_calibration_accepted=False)
if a.phase=='fit':
    gate=rt.read(root/'fit/geometry-gate.json');assert gate['coarse_initialization_gate_passed'] or a.stop
    d.update(fit_sha256=rt.sha(root/'fit/fitted.npz'),actual_same_camera_fit_reviewed=True,allow_surface_initialization=not a.stop,formal_geometry_accepted=False,coarse_gate=gate)
    path=root/'reviews/fit-decision.json'
elif a.phase=='surface':
    d.update(points_sha256=rt.sha(root/'clothing-init/points.npz'),actual_initialization_review=True,actual_center_distribution_review=True,allow_finite_research_training=not a.stop)
    path=root/'reviews/surface-decision.json'
else:
    d.update(snapshot_sha256=rt.sha(root/f'gaussian-run/step-{a.step:06d}/world.ply'),actual_same_camera_review=True,actual_center_distribution_review=True,continue_same_run=not a.stop)
    path=root/f'reviews/gaussian-step-{a.step:06d}-decision.json'
assert not path.exists(),'Never replace an already consumed training decision'
rt.save(path,d);rt.save(folder/'actual-review.json',d)
print('ACTUAL_IMAGE_DECISION_RECORDED',a.subject,a.phase,'stop' if a.stop else 'continue_finite_same_run')
