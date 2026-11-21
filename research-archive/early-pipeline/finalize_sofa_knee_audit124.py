"""Record the actual review; leave the measured audit/replay results unchanged."""
import audit_gaussianavatars_params123 as a
from datetime import datetime, timezone

root=a.R.parent/'local-clothed-surface-123/cross-subject-sh0-124/sofa-temporal-warmstart-v2'
cpu=root/'knee-objective-audit/result.json'
replay=root/'knee-stopping-replay/result.json'
review=root/'knee-stopping-replay/visual-review.json'
c,r,v=a.read(cpu),a.read(replay),a.read(review)
assert c['replay_final_loss_accepted'] and r['endpoint_parity_passed']
assert all(x==0 for x in r['endpoint_parity'].values())
assert not v['actual_visual_review']
stamp=datetime.now(timezone.utc).isoformat()
v.update(updated=stamp,state='actually_reviewed',actual_visual_review=True,
    conclusion='Native unscaled raw/stage3/stage4 overlay actually viewed: final face/hand stage moves the left knee down the shin. Stage3 has a better knee projection but face, clothing shape and hidden feet are not accepted.',
    quality_accepted=False,gaussian_admitted=False)
a.save(review,v)
conclusion=dict(updated=stamp,source_cpu_result=str(cpu),source_replay_result=str(replay),source_visual_review=str(review),
    readonly_cpu_seconds=c['elapsed_seconds'],exact_replay_fit_seconds=r['fit_seconds'],
    endpoint_parity=r['endpoint_parity'],replay_fitter_calls=1,new_sequence_fits=0,
    image=v['image'],actual_visual_review=True,quality_accepted=False,gaussian_admitted=False,
    summary='连续帧约束已生效；精确重放把左膝退化定位到最后的脸手联合阶段。前一阶段左膝161→31px，最后阶段31→101px。最后阶段手点权重0.1→2（平方后400倍）、面部点0→2，全身姿态仍参与优化，手脸误差降低时身体投影退化。末阶段因损失相对变化为零在第12次外层迭代停止，并未耗尽30次。',
    findings=['Saved-state CPU geometry/projection and loss replay passed; temporal term6.56 versus total10743.50. A complete old16 diagnostic state has lower same-objective total10147.61 and knee34.77px; it is not selected or reused as an initializer.',
              'One unchanged GPU diagnostic replay reproduced vertices, projected joints, VPoser latent, root, translation and beta exactly. Original stages3/4 took10/12 outer optimizer steps and both stopped by ftol with relative loss change0.',
              'Stage4 joint-weight transition raises hand data from0.1 to2 and face from0 to2; all other active stage weights remain equal. Over that stage, body data1923.10->3941.26, hand30424.41->2906.05, face11813.38->2144.77. Knee30.66->101.23px.',
              'The same-run17 initializer has knee160.65px under frame16 camera, but stage3 repairs it to30.66px. Therefore an initially bad neighbour is not a sufficient explanation for the final regression.'],
    limits=['This locates the regression to the combined face/hand stage, not uniquely to hand versus face observation errors.',
            'ftol stop with nonzero gradient does not prove that merely increasing the iteration cap or changing precision fixes the pose.',
            'No physical3D truth, full sequence acceptance, new Gaussian model or deployment is established.'],
    next_action='Do not restart either rejected sofa sequence, rerun this replay, increase temporal weights or stitch saved poses. Inspect the original final-stage parameter selection and prepare a minimal, uniform body-protection/face-hand refinement adapter using the original optimizer. Register a bounded diagnostic and body/face/feet protections before any new fitting; no full sequence or Gaussian is admitted. Keep cameras, shared shape, observation coordinates and existing SH0 viewer model unchanged. Full cross-subject goal remains active.')
dest=root/'knee-stopping-replay/conclusion.json'
assert not a.lp(dest).exists()
a.save(dest,conclusion)
print(conclusion['summary'])
