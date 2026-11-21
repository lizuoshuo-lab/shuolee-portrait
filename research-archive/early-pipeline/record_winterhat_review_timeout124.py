"""Reconcile measured process exit; retain the stale status and late review."""
import shutil
import subprocess
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone


class gs:
    ROOT = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/cross-subject-sh0-124')
    read = staticmethod(lambda p: json.loads(p.read_text(encoding='utf-8')))
    save = staticmethod(lambda p, d: p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'))
    sha = staticmethod(lambda p: hashlib.sha256(p.read_bytes()).hexdigest())


def main():
    out = gs.ROOT / 'winterhat-half8-paired18k124'
    launch = gs.read(out / 'launch.json')
    failed = gs.read(out / 'train-failure.json')
    decision = gs.read(out / 'review-1000-decision.json')
    assert 'Review timeout; checkpoint retained' in failed['error']
    assert datetime.fromisoformat(decision['updated']) > datetime.fromisoformat(failed['updated'])
    process = subprocess.run(['wsl', '-d', 'Ubuntu-20.04', '--', 'ps', '-p',
                              str(launch['pid']), '-o', 'args='], capture_output=True)
    command = process.stdout.decode('utf-8', errors='replace').strip()
    assert 'train_winterhat_half8_sh0124.py train' not in command
    old = out / 'progress-before-exit-reconciliation.json'
    assert not old.exists(), 'Reconciliation already recorded'
    shutil.copyfile(out / 'progress.json', old)
    prior = gs.read(old)
    now = datetime.now(timezone.utc).isoformat()
    receipt = dict(updated=now, actual_process_alive=False, old_pid=launch['pid'],
        observed_ps_command=command, recorded_exit_code=1, failure=failed['error'],
        failure_time=failed['updated'], decision_time=decision['updated'],
        decision_applied_to_original_process=False, original_optimizer_continued=False,
        checkpoint=str(out / 'gaussian-run/step-001000/checkpoint.pt'),
        checkpoint_sha256=gs.sha(out / 'gaussian-run/step-001000/checkpoint.pt'),
        finding_zh='训练在1000步审核等待600秒后退出。实际审阅已完成，但继续决定晚于退出，未被原进程执行。保存的模型和Adam状态正在核验恢复；原1000步、失败和决定均保留。',
        exact_continuation_possible=False, quality_accepted=False)
    gs.save(out / 'review-execution-receipt.json', receipt)
    gs.save(out / 'process-exit.json', dict(updated=now, exit_code=1,
        original_pid=launch['pid'], actual_process_alive=False,
        error=failed['error'], late_decision_applied=False))
    prior.update(updated=now, state='review_timeout_checkpoint_retained', pid=None,
        original_pid=launch['pid'], optimizer_waiting=False, training_running=False,
        late_decision_applied=False, recovery_started=False, quality_accepted=False)
    gs.save(out / 'progress.json', prior)
    print(receipt['finding_zh'])


if __name__ == '__main__':
    main()
