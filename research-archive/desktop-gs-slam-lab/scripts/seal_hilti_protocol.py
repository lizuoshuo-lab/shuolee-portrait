"""Seal the prepared local Hilti protocol before either actual replay starts."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/fast-livo2-hilti-gt-001/evaluation'
PLAN = ROOT / 'outputs/fast-livo2-hilti-gt-001/evaluation-plan.json'
EXPECTED_PLAN = 'e269d018d86cc92ee1cf5cb67494a8613ad56db027ee7084584afd38b48c5aa4'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    assert sha(PLAN) == EXPECTED_PLAN, 'prepared plan changed'
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    checked = {}
    for relative, expected in plan['prepared_source_SHA256'].items():
        actual = sha(ROOT / relative)
        assert actual == expected, f'prepared source changed: {relative}'
        checked[relative] = actual
    static_review = ROOT / plan['static_runner_review']
    assert sha(static_review) == plan['static_runner_review_sha256']
    receipt_path = OUT / 'preregistration.json'
    if receipt_path.exists():
        existing = json.loads(receipt_path.read_text(encoding='utf-8'))
        assert existing['evaluation_plan_sha256'] == EXPECTED_PLAN
        assert existing['prepared_source_sha256'] == checked
        print(json.dumps({'status': 'existing_seal_preserved', 'receipt': str(receipt_path)}, ensure_ascii=False))
        return
    runs = ROOT / 'outputs/fast-livo2-hilti-gt-001/runs'
    assert not runs.exists() or not any(runs.glob('*/status.json')), 'actual replay already exists; cannot retroactively preregister'
    OUT.mkdir(parents=True, exist_ok=True)
    receipt = {
        'status': 'sealed_before_actual_replays',
        'sealed_utc': datetime.now(timezone.utc).isoformat(),
        'evaluation_plan_sha256': EXPECTED_PLAN,
        'prepared_source_sha256': checked,
        'static_runner_review_sha256': sha(static_review),
        'local_only': True,
        'actual_replay_or_score_completed': False,
    }
    with receipt_path.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'status': receipt['status'], 'sealed_utc': receipt['sealed_utc'], 'path': str(receipt_path), 'sha256': sha(receipt_path)}, ensure_ascii=False))

if __name__ == '__main__':
    main()
