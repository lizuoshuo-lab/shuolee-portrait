from datetime import datetime, timezone
import audit_gaussianavatars_params123 as a

base=a.REC/'demo-vhap-initialization/body-gaussian-smplify-eight-dc-reviewedfit'
out=base/'sam2-limb-prompt-control'
plan=a.read(out/'plan.json')
script=a.H/'probe_sam2_limb_prompts123.py'
old=next(h for p,h in plan['files'].items() if p.endswith('/probe_sam2_limb_prompts123.py'))
assert not a.lp(out/'paired-control-amendment.json').exists()
assert not a.lp(out/'result.json').exists()
failure=a.read(out/'failure.json')
assert 'baseline mask' in failure['error']
note=dict(updated=datetime.now(timezone.utc).isoformat(),
    state='registered_before_one_paired_attempt',
    previous_script_sha256=old,script_sha256=a.sha(script),
    original_failure_sha256=a.sha(out/'failure.json'),
    reason='The historical mask replay differed at35 pixels before any candidate was computed. Its exact numerical cause is unresolved. A bitwise archived mask is not required to isolate prompt changes when the original prompt and limb prompt are both evaluated from the same live image embedding in the same process.',
    scope='One amended attempt, eight fixed frames, paired live original and limb prompts. Report archived/live discrepancy separately; do not attribute it to the limb prompt, overwrite old masks or claim bitwise replay was repaired.',
    changes='Comparator contract only. Native images, original box, original center, model weights, added positive-point rule, score sanity check and120second budget remain fixed.',
    stop='Stop on changed hashes, nonfinite/low-score outputs, occupied GPU, runtime budget or any other failure. No further retries or prompt sweeps. Require all8 physical source/mask review before adoption; candidate remains unaccepted by default.',
    training_started=False,mask_replaced=False)
a.save(out/'paired-control-amendment.json',note)
status=a.read(a.C/'local-research-status.json')
status['gpu_coordination_20261102']=dict(updated=note['updated'],
    state='reserved_for_one_paired_mask_prompt_control',timeout_seconds=120,
    training_updates=0,script='probe_sam2_limb_prompts123.py run',previous_attempt='failed_before_candidate')
a.save(a.C/'local-research-status.json',status)
print('PAIRED_MASK_AMENDMENT_REGISTERED')
