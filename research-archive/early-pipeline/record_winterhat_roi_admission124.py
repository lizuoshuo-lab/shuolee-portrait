"""Record actual source/ROI/mask inspection before zero-update GPU checks."""
from train_winterhat_half8_sh0124 import OUT
from run_official_gaussians123 import read, save, sha, now

assert read(OUT/'progress.json')['state']=='prepared_input_review_pending'
rows=read(OUT/'source-native-inputs.json')['records']
assert len(rows)==8
for row in rows:
    for key,digest in [('image','source_sha256'),('mask','mask_sha256')]:assert sha(row[key])==row[digest]
review=read(OUT/'validity-review.json');assert not review['actual_visual_review']
review.update(updated=now(),actual_visual_review=True,training_admitted=True,
    finding='All8 cropped RGB/mask/masked targets actually inspected. Hat, cardigan, long skirt, bag and boots included. Some bag interior holes are present in the source masks; their physical correctness is unverified. No pixels were filled or hand-edited.',
    initial_admission_scope='1000-step exploratory diagnostic only; raised forearm/wrist-depth mismatch remains. No automatic18k continuation.',
    record_contact_sha256=sha(OUT/'input-contact.png'),source_mask_unchanged=True,
    next_review_regions=['raised_arm_depth16/96/109','duplicate_arm_or_floating_hand','long_sleeve_stretch','hat/skirt/bag coverage','face/boots'],
    quality_accepted=False)
save(OUT/'validity-review.json',review)
plan=read(OUT/'plan.json');plan.update(initial_admission_max_step=1000,automatic_continue_to18000=False,
    initial_admission='Only an actual1000-step diagnostic is admitted. Source review did not establish pose quality; wrong forward wrist depth must not be relabeled as finger uncertainty.')
save(OUT/'plan.json',plan)
print('WINTERHAT_NATIVE_ROI1000_DIAGNOSTIC_INPUTS_REVIEWED')
