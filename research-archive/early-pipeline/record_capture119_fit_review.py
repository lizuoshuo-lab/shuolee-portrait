"""Record the six fresh same-camera body overlays actually inspected this turn."""
from capture119_clothed import ROOT,REC,rt
root=ROOT/'portrait';folder=REC/'portrait-fit-001000-review'
gate=rt.read(root/'fit/geometry-gate.json');index=rt.read(folder/'index.json')
rt.save(root/'reviews/fit-decision.json',dict(created=rt.now(),fit_sha256=rt.sha(root/'fit/fitted.npz'),
    actual_same_camera_fit_reviewed=True,reviewed_frames=[x['frame'] for x in index['images'] if x['region']=='body'],
    review_image_sha256=rt.sha(folder/'body-contact.png'),coarse_gate=gate,
    allow_surface_initialization=True,formal_geometry_accepted=False,physical_calibration_accepted=False,
    observations='Fresh bare template stays approximately within the body in six views; skirt, sweater, hat and red bag extend clearly beyond it. Some foot/face alignment remains imperfect. Use this fit only for a broad proposal box and scale, not a skin or garment target.'))
print('FRESH_FIT_REVIEW_RECORDED')
