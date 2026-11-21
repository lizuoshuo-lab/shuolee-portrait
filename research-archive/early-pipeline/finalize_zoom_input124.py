"""Complete review metadata before any GPU run; retain the original registration."""
import copy
import shutil
from pathlib import Path
import numpy as np
import train_indoor_person_zoom124 as trial
gs=trial.gs;out=trial.OUT
assert not (out/'preflight-launch.json').exists() and not (out/'launch.json').exists()
assert not (out/'plan-before-metadata-completion.json').exists()
shutil.copyfile(out/'plan.json',out/'plan-before-metadata-completion.json')
shutil.copyfile(out/'data/training-inputs.json',out/'data/training-inputs-before-box-mapping.json')
doc=gs.read(out/'data/training-inputs.json');mapping=gs.read(out/'pixel-mapping.json')
for row,item in zip(doc['records'],mapping['records']):
    assert row['frame']==item['frame']
    x0,y0,_,_=item['crop_xyxy'];offset=np.array([x0,y0,x0,y0])
    for key in ['initial_box_xyxy','face_box_xyxy']:
        if item['source'].get(key) is not None:row[key]=((np.asarray(item['source'][key])-offset)*2).tolist()
    if item['source'].get('other_person_boxes_xyxy'):
        row['other_person_boxes_xyxy']=((np.asarray(item['source']['other_person_boxes_xyxy'])-offset)*2).tolist()
gs.save(out/'data/training-inputs.json',doc)
receipt=gs.read(out/'smplify-gaussian-interface-receipt.json')
receipt['training_inputs_sha256']=gs.sha(out/'data/training-inputs.json');gs.save(out/'smplify-gaussian-interface-receipt.json',receipt)
review=dict(updated=gs.now(),actual_visual_review=True,frames=[x['frame'] for x in mapping['records']],
    evidence=str(out/'input-contact.png'),evidence_sha256=gs.sha(out/'input-contact.png'),
    finding='All6 existing person masks remain in the deterministic crops. Face, shirt, visible arms and shorts retained. Existing wrist/forearm mask defects and furniture/edge leg occlusions are not repaired by zoom.',
    transformed_boxes_registered=True,no_manual_frame_or_mask_selection=True,
    training_admitted=True,admission_scope='User-authorized bounded1000 ROI-plus2x experiment, not production-quality admission.',
    occluded_regions_outside_quality_acceptance=True,unknown_is_not_blank=True,quality_accepted=False)
gs.save(out/'validity-review.json',review)
plan=gs.read(out/'plan.json')
for path in [Path(trial.__file__),Path(__file__),out/'data/training-inputs.json',out/'smplify-gaussian-interface-receipt.json',out/'validity-review.json']:
    plan['source_hashes'][str(path)]=gs.sha(path)
plan.update(updated=gs.now(),state='input_reviewed_preflight_pending',
    metadata_completion='Mapped visualization-only face/person boxes into crop2x coordinates before GPU; original registration retained. Frozen fitted3D archive unchanged.',
    fitting_observation_domain='Original fit archive observations remain native; review maps2D labels explicitly, never treats them as crop pixels.')
gs.save(out/'plan.json',plan)
gs.save(out/'progress.json',dict(updated=gs.now(),state=plan['state'],step=0,pid=None))
print('ZOOM_INPUT_AND_METADATA_REVIEWED')
