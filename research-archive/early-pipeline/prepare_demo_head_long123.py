"""Register the user-requested 300k repeat without altering the completed run."""
import ast,difflib,shutil
import audit_gaussianavatars_params123 as a
from run_official_gaussians123 import now

base=a.REC/'demo-vhap-initialization/head-gaussian-first-cycle'
out=a.REC/'demo-vhap-initialization/head-gaussian-long300k'
source=a.H/'train_demo_head_gaussians123.py'
target=a.H/'train_demo_head_gaussians123_long.py'
old=a.lp(source).read_text(encoding='utf-8')
assert a.sha(source)==a.read(base/'plan.json')['script_sha256']
assert a.read(base/'gaussian-run/result.json')['steps']==18000
assert not a.lp(out).exists() and not a.lp(target).exists()
new=old
edits=[
 ("OUT=ROOT/'head-gaussian-first-cycle'","OUT=ROOT/'head-gaussian-long300k'"),
 ('opt.iterations=opt.position_lr_max_steps=opt.densify_until_iter=18000','opt.iterations=opt.position_lr_max_steps=opt.densify_until_iter=300000'),
 ('opt.densify_from_iter=300;opt.densification_interval=60;opt.opacity_reset_interval=1800','opt.densify_from_iter=5000;opt.densification_interval=1000;opt.opacity_reset_interval=30000'),
 ('for step in range(1,18001):','for step in range(1,300001):'),
 ("if time.monotonic()-start-wait_seconds>900:raise RuntimeError('Registered900s Gaussian compute budget reached')","if time.monotonic()-start-wait_seconds>14400:raise RuntimeError('Registered14400s Gaussian compute budget reached')"),
 ('if 300<step<18000 and step%60==0:','if 5000<step<300000 and step%1000==0:'),
 ('if step<18000 and step%opt.opacity_reset_interval==0:','if step<300000 and step%opt.opacity_reset_interval==0:'),
 ('if step in [1000,6000,12000,18000]:','if step in [1000,10000,30000,60000,100000,150000,200000,250000,300000]:'),
 ("if step==18000 else 'stopped_after_review'","if step==300000 else 'stopped_after_review'"),
 ("choices=['prepare','train']","choices=['train']"),
]
for before,after in edits:
    assert new.count(before)==1,before
    new=new.replace(before,after)
ast.parse(new)
a.lp(out).mkdir(parents=True)
a.lp(target).write_text(new,encoding='utf-8')
a.lp(out/'source').mkdir()
shutil.copy2(a.lp(source),a.lp(out/'source/completed18000.py'))
shutil.copy2(a.lp(target),a.lp(out/'source/train300000.py'))
a.lp(out/'source/schedule-only.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='completed18000',tofile='long300000')),encoding='utf-8')
for name in ['inputs.json','fitted-head-meshes.npz']:
    shutil.copy2(a.lp(base/name),a.lp(out/name));assert a.sha(base/name)==a.sha(out/name)
plan=a.read(base/'plan.json')
plan.update(updated=now(),authorization='User: 再做一版30万步的训练. Complete sending first, then start this independent fresh long run.',
    schedule_steps=300000,densify_from=5000,densify_interval=1000,opacity_reset_interval=30000,
    max_train_seconds=14400,script_sha256=a.sha(target),prepared_seconds=None,
    comparison_run=str(base),first_review_step=1000,from_scratch=True,old_checkpoint_loaded=False,
    schedule_policy='Same normalized phase timing as original 600k defaults: halve LR decay, densification and opacity-reset time axes for300k. SH increments every1000 stay upstream. Losses, fullK, pixels, meshes, seeds and point cap unchanged.',
    estimate_seconds=[9000,10800],snapshots=[1000,10000,30000,60000,100000,150000,200000,250000,300000],
    limits='Longer optimization is not a guarantee of better novel-view geometry; the18000-step comparison failed quality. No new fitter or solver, no weight downloads, no camera changes.')
a.save(out/'plan.json',plan)
a.save(out/'progress.json',dict(updated=now(),state='prepared',target_steps=300000,training_started=False,quality_accepted=False))
print(str(out));print(a.sha(target))
