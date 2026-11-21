"""Input and original-stage contract check, without optimizer construction."""
import fcntl
import inspect
import time
import torch
import fit_smplify_vhap_sequential_stage123 as trial
from run_official_gaussians123 import read, save, sha, now


def main():
    p=read(trial.PLAN)
    assert p['execution_authorized'] is False and not trial.OUT.exists()
    assert sha(trial.__file__)==p['source_sha256']
    for path,h in {**p['input_hashes'],**p['source_hashes']}.items():
        assert sha(path)==h,path
    out=trial.ROOT/'smplify-vhap-sequential-preflight'
    assert not out.exists()
    lock=open('/tmp/shuolee-native122-gpu.lock','a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out.mkdir();start=time.monotonic();torch.set_num_threads(4)
    t,rows=trial.make_tracker(out)
    receipts=trial.bridge.initialize(t,rows)
    before={k:trial.bridge.digest_tensor(getattr(t,k)) for k in trial.bridge.PARAMS}
    groups=t.get_train_parameters('rgb_sequential_tracking')
    keys=list(groups)
    assert not any(k in groups for k in ('shape','static_offset','lights','cam','dynamic_offset'))
    assert all(k in groups for k in ('translation','rotation','body','lhand','rhand','expr','tex_extra'))
    assert t.cfg.pipeline.rgb_sequential_tracking.num_steps==50
    assert t.cfg.pipeline.rgb_global_tracking.num_epochs==30
    assert all(trial.bridge.digest_tensor(getattr(t,k))==h for k,h in before.items())
    save(out/'result.json',dict(updated=now(),state='preflight_passed_awaiting_decision',frames=len(rows),
        optimizer_updates=0,training_started=False,stage='original rgb_sequential_tracking',
        source=inspect.getsourcefile(t.optimize_stage),source_sha256=sha(inspect.getsourcefile(t.optimize_stage)),
        groups=keys,checks=receipts,initial_parameters_unchanged=True,
        seconds=time.monotonic()-start,quality_accepted=False))
    print('PREFLIGHT_PASSED all59, original groups, zero optimizer updates, no training started.',flush=True)


if __name__=='__main__':main()
