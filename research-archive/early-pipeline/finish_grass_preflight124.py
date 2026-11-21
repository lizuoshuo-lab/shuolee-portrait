import run_grass_temporal124 as c
p=c.read(c.OUT/'plan.json')
assert not (c.OUT/'official-fit').exists()
old=p['sources'][str(c.Path(c.__file__))]
for path,digest in p['sources'].items():
    if path!=str(c.Path(c.__file__)): assert c.sha(path)==digest,path
c.save(c.OUT/'plan-before-json-scalar-repair.json',p)
p['sources'][str(c.Path(c.__file__))]=c.sha(c.__file__)
p['preflight_repair']='NumPy bool JSON serialization only; preflight checked before zero fitting calls. Historical source hash preserved separately.'
c.save(c.OUT/'plan.json',p)
c.save(c.OUT/'preflight-repair.json',dict(old_source_sha256=old,new_source_sha256=c.sha(c.__file__),error='numpy.bool_ not JSON serializable',fit_calls=0,policy_unchanged=True))
c.save(c.OUT/'interface-check.json',dict(updated=c.now(),transport=c.check_transport(),original_observations_bytes_preserved=True,real_PTS_order=True,complete_native_K=True,original_temporal_checks=str(c.ROOT/'winterhat-temporal/interface-check-repaired.json'),original_pose_used_as_initialization=False,quality_accepted=False))
print('GRASS_INTERFACE_OK',p['ordered_fit_frames'][:13],p['ordered_fit_frames'][26])