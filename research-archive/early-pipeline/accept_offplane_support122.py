"""Record actually inspected N9 materials, retaining every original observation."""
from audit_offplane_support122 import B, O, read, save, sha, now

rows=[r for p in read(O/'patch-index.json') for r in p['rows']]
assert len(rows)==53
near_water={6,342,361,603,606,770,876,913}
distant_vegetation={381,890,900,906,992,1020}
woody={24,765,773,784,820}
shore={19,33,365,399,557,572,573,575,588,598,659,755,769,778,977,986,1081}
review=[]
for r in rows:
    t=r['track']
    category='water_edge_vegetation' if t in near_water else 'distant_vegetation_or_reflection' if t in distant_vegetation else 'wood_rocks_or_branch_context' if t in woody else 'shore_boundary' if t in shore else 'mudflat_ground_texture'
    review.append(dict(track=t,family=r['family'],native_views=[x['frame'] for x in r['observations']],category=category,
        physical_identity='contradicted' if t==399 else 'not_certified_at_subpixel',
        finding='Frames246/262 center the same near sandbar/vegetation boundary; frame278 centers the far bank, while the original near sandbar remains visibly below-left. This is a material switch, not just a centroid offset.' if t==399 else 'Context inspected in all3 crops. No exact static material/centroid claim from appearance alone; shoreline, water/reflection, vegetation and repeated ground textures remain ambiguous.'))
save(O/'visual-review.json',dict(updated=now(),state='completed_reviewed',reviewer='Codex actual image inspection',
    viewed_pages=[p['file'] for p in read(O/'patch-index.json')],viewed_extra=['frame278_detector_exclusion.png','track399_native_context.png'],
    reviewed_families=53,reviewed_crops=159,rows=review,
    evidence=[
        'All53 preselected FIT-family representatives inspected; this is the topdecile conditional plane-distance sample, not all608 families.',
        '52/53 selected representatives have maxparallax below3degrees; the sole11.30degree representative is the visually contradicted family399.',
        'The large source detector exclusion rectangle also removes visible ground, shrubs and shoreline outside the person. This is a coverage limitation; recovering it is not yet a proven remedy.',
        '935FIT tracks represent608 near-duplicate families, so trackcount is not an independent physical-point count.'
    ],
    material_contradictions=[399],original_observations_changed=False,heldout_reassigned=False,
    all_scene_static_certified=False,physical_face_admission=False,training_admitted=False,
    next='One preregistered family399 quarantine diagnostic with fixedK and all other original observations unchanged. Reinitialize from raw observations; compare against N5c. Do not combine a new detector mask, K or threshold change.'))
result=read(O/'result.json');result.update(state='completed_reviewed',visual_review='visual-review.json',
    visually_contradicted_fit_families=[399],selected_families_max_parallax_lt3=sum(r['geometry']['max_parallax_deg']<3 for r in rows))
save(O/'result.json',result);save(O/'progress.json',dict(state='completed_reviewed',pid=None,updated=now()))
print('N9_REVIEW_SAVED')
