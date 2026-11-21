"""Freeze semantic support review before the first global pose solve."""
from global_fixedk121 import *
target=R/'global-fixedk-plan.json'
old=R/'global-fixedk-plan.initial-background-proposal.json'
if old.exists() or any((ROOT/s/'global-fixedk').exists() for s in SUBJECTS):
    raise RuntimeError('Preflight already committed or solver started')
old.write_bytes(target.read_bytes());plan=read(target)
plan['material_preflight']={
 'grass':dict(decision='stop_before_pose',reason='Sampled long background tracks are foliage. Of376 retained fit tracks outside the person,0 remain wholly in the lower half image; no supported near-ground static control across the24frames. Do not estimate a camera with moving leaves and call it static calibration.'),
 'sofa':dict(decision='run_one_fixedK_diagnostic',min_y_fraction=.5,reason='Exclude upper-half curtain/ceiling repetition. Reuse lower-half furniture/floor observations only, frozen before geometry. Cushion deformation/glass remain material risks to audit; no assumption that every remaining pixel is static.'),
 'portrait':dict(decision='run_one_fixedK_diagnostic',min_y_fraction=.6,reason='Reviewed ground-crack tracks are plausible material identities. Restrict all observations to the lower60% cutoff below distant foliage/water. Ground is mostly planar; do not equate conditional-K consistency with physical calibration.'),
 'indoor':dict(decision='stop_before_pose',reason='Sampled long background tracks dominated by bright repeated/specular chandelier details. Only45 fit tracks remain wholly in the lower half across24frames; no tested adjacent/one-gap pair has20 such tracks. Valid static multi-depth coverage not established; no candidate trained from this unsupported graph.')}
plan['preflight_changes_not_geometry_tuning']='One semantic eligibility correction after native patch review and coordinate-distribution inspection, before any121global camera solve. No K/geometry/residual threshold scan. Original proposal is retained.'
plan['camera_fit']='Retained FIT tracks outside body+12px for every observation AND declared per-person material-support region. No heldout conversion. Grass/indoor stop before camera fitting.'
plan['worker_sha256']=sha(Path(__file__).with_name('global_fixedk121.py'));plan['material_reviewed']=now()
save(target,plan)
protocol=read(R/'protocol.json');protocol['phase_C_global_fixedK']='global-fixedk-plan.json';save(R/'protocol.json',protocol)
save(R/'global-material-review.json',dict(updated=now(),reviewed=['four native first/middle/last overview sheets','32 temporal-stratified longest fit/heldout background three-view patches'],findings=plan['material_preflight'],sampled_tracks_are_not_all_material_truth=True,training_started=False))
print('GLOBAL_MATERIAL_PREFLIGHT_FROZEN')
