"""Archive reviewed read-only mechanism diagnostics; preserve all prior results."""
from __future__ import annotations
import argparse
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import uuid

LAB = Path(__file__).resolve().parent.parent
GPU = 'outputs/splatam-checkpoint-support-diagnostic-001/20261010T160510Z_6e42d904'
FAST = 'outputs/fast-livo2-hilti-vio-diagnostic-001/20261010T160003Z_aa0ce023'
INPUT = 'outputs/splatam-room-input-diagnostic-001/20261010T155109Z_3ba18586'
ROOM = 'outputs/splatam-room-mechanism-diagnostic-001/20261010T155434Z_570bc98c'
PROTECTED = {
    'experiments/30_splatam_tracking_ablation_repeat.md': '0444e16b188e5983882f23eb957ffa8828dca2a201d74c5758c9fd54250e11c7',
    'experiments/31_fast_livo2_hilti_gt.md': '1723606841ba279d7c71bbe7e16898d377ec2efb3c4b15c7eaefbc692840f7d5',
    'outputs/splatam-ablation-repeat-001/report.md': 'd19381104a3a4ebe8f95277b1ce6c655d4b1dfbd327fb9abaffdd9512f895c33',
    'outputs/fast-livo2-hilti-gt-001/report.md': 'd4198bde7a102b843780904c9f1fe7e0482036d841e8ec3461d314c8bc432e17',
}

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(relative): return json.loads((LAB / relative).read_text(encoding='utf8'))
def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-review', required=True)
    parser.add_argument('--fast-review', required=True)
    parser.add_argument('--article-sha256', required=True)
    args = parser.parse_args()
    for relative, expected in PROTECTED.items(): assert sha(LAB / relative) == expected, relative
    gpu, fast = read(GPU + '/metrics.json'), read(FAST + '/metrics.json')
    gpu_review, fast_review = read(args.gpu_review), read(args.fast_review)
    assert gpu_review['status'] == 'passed_independent_completed_fixed_map_diagnostic_readback'
    assert fast_review['status'] == 'passed_independent_raw_log_numeric_review'
    assert sha(LAB/args.gpu_review) == '4e52a27f20e718a253c44e513db57d18fc4356e6e07793b3e7b6f6cb98a72ca2'
    assert sha(LAB/args.fast_review) == 'f6ee62336245dfeb7272a5350e3a7245de13adbfbc842ea7edc71f4731ea67b4'
    for review, folder, script in [(gpu_review,GPU,'scripts/diagnose_splatam_checkpoint_support.py'),
            (fast_review,FAST,'scripts/diagnose_fast_livo2_hilti_vio.py')]:
        assert Path(review['diagnostic_metrics']).resolve() == (LAB/folder/'metrics.json').resolve()
        assert review['diagnostic_metrics_sha256'] == sha(LAB/folder/'metrics.json')
        assert review['diagnostic_source_sha256'] == sha(LAB/script)
    assert gpu_review['registered_plan_sha256'] == gpu['plan_sha256'] == sha(LAB/GPU/'plan.json')
    assert gpu['status'] == 'completed_registered_read_only_fixed_map_support_diagnostic'
    assert gpu['actual_condition_count'] == gpu['official_loss_recomputation_checks'] == 38
    assert gpu['actual_renderer_forward_count'] == 152
    assert gpu['inputs_and_official_sources_unchanged'] is True
    assert len(gpu['comparisons_to_base']) == 36
    assert all(not c['original_sum_lower'] and c['fixed_base_pixel_loss_higher'] for c in gpu['comparisons_to_base'])
    assert fast['exact_vio_pairs'] == 737 and fast['post_vio_sample_rows'] == 36
    assert fast_review['inputs_and_old_results_unchanged'] is True
    for folder, exit_field in [(GPU,'actual_process_exit_code'),(FAST,'actual_exit_code'),
            (INPUT,'actual_process_exit_code'),(ROOM,'actual_process_exit_code')]:
        terminal = read(folder + '/tool-exit-observation.json')
        assert terminal[exit_field] == 0, folder
    article = LAB / 'experiments/32_slam_drift_mechanism.md'
    assert sha(article) == args.article_sha256
    assert 'status: "mechanism_diagnostics_independently_verified_historical_cause_unresolved"' in article.read_text(encoding='utf8')
    progress_path = LAB / 'manifests/progress.json'
    before_bytes = progress_path.read_bytes()
    before_sha = hashlib.sha256(before_bytes).hexdigest()
    assert before_sha == '3cff0df26c41ee621afe41344140abaa4075da717f897b792603fbedb163bf48', before_sha
    before = json.loads(before_bytes.decode('utf8'))
    assert len(before['results']) == 46
    assert not any(r['id'] == 'slam-drift-mechanism-001' for r in before['results'])
    manifest_path = LAB / 'manifests/slam-drift-mechanism.json'
    assert not manifest_path.exists()
    now = dt.datetime.now(dt.timezone.utc)
    out = LAB / 'outputs/slam-drift-mechanism-001' / (now.strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    (out / 'progress-before.json').write_bytes(before_bytes)
    bindings = [GPU+'/metrics.json', GPU+'/plan.json', GPU+'/pose-support.csv', GPU+'/tool-exit-observation.json',
        FAST+'/metrics.json', FAST+'/vio-transitions.csv', FAST+'/tool-exit-observation.json', INPUT+'/metrics.json',
        INPUT+'/tool-exit-observation.json', ROOM+'/metrics.json', ROOM+'/tool-exit-observation.json',
        args.gpu_review, args.fast_review, 'experiments/32_slam_drift_mechanism.md', 'scripts/archive_slam_mechanism.py']
    manifest = {
        'status': 'mechanism_diagnostics_independently_verified_historical_cause_unresolved',
        'created_utc': now.isoformat(), 'local_only': True,
        'article': 'experiments/32_slam_drift_mechanism.md',
        'diagnostic_directories': {'room_inputs': INPUT, 'room_frozen_result': ROOM, 'fixed_map_gpu': GPU, 'fast_vio': FAST},
        'independent_reviews': [args.gpu_review, args.fast_review],
        'bindings_sha256': {p: sha(LAB / p) for p in bindings},
        'protected_old_documents_and_reports_sha256': PROTECTED,
        'new_accuracy_score': False,
        'findings': [
            'Fixed-map masks vary with pose. None of the 36 nonbaseline probes lowered the original joint sum or fixed-baseline-pixel sum.',
            'A depth-sum decrease at checkpoint1100 ry+15deg coexists with fewer supported pixels and a worse depth MAE; the joint RGB-D sum still rises.',
            '737 exactly paired visual calls, 736 retain net state changes at logged precision; changes do not prove ground-truth improvement.',
            'Historical tracking masks/iteration residuals at room838/1133 and internal FAST visual acceptance/residual descent are absent from original logs.',
        ],
        'limits': ['Read-only diagnostics; no new model training, trajectory correction, GT fitting or replacement of prior ATE/RPE.',
            'Saved post-mapping checkpoints are not the historical online maps at error jumps.',
            'Calibration accuracy, hardware synchronization and cross-scene superiority remain unproven.'],
    }
    save(manifest_path, manifest)
    result = {
        'id': 'slam-drift-mechanism-001', 'title': '漂移机制：固定地图有效像素与视觉阶段状态修正',
        'status': manifest['status'],
        'metrics': {'room_input_frames': 1315, 'fixed_map_conditions': 38, 'renderer_forwards': 152,
            'official_loss_recomputations_passed': 38, 'nonbaseline_conditions': 36,
            'nonbaseline_joint_loss_below_base': 0, 'exact_vio_pairs': 737,
            'logged_net_state_changed_calls': 736, 'post_vio_samples_verified': 36},
        'artifacts': ['manifests/slam-drift-mechanism.json', GPU+'/metrics.json', GPU+'/pose-support.csv',
            GPU+'/fixed-map-support-and-loss.png', FAST+'/metrics.json', FAST+'/vio-transitions.csv',
            args.gpu_review, args.fast_review],
        'scope': '冻结完整运行的只读机制诊断与38条件GPU对照；明确分项损失、联合损失、支持像素和净状态修正的区别。未确定历史漂移因果，未新增或改变算法准确率成绩。',
    }
    after = copy.deepcopy(before)
    after['results'].append(result)
    after['updated'] = now.astimezone(dt.timezone(dt.timedelta(hours=8))).strftime('%Y-%m-%d')
    after['current_stage'] = 'SLAM_mechanism_diagnostics_verified_historical_cause_unresolved'
    after['mechanism_status'] = manifest['status']
    after['next_steps'] = list(before['next_steps']) + ['SplaTAM 需要新的受控完整运行逐迭代记录在线支持、分项损失、初值和最终位姿；稀疏post-mapping地图无法恢复原历史因果。']
    assert after['results'][:46] == before['results']
    for k in before:
        if k not in {'results','updated','current_stage','next_steps'}: assert after[k] == before[k], k
    assert sha(progress_path) == before_sha, 'progress changed concurrently'
    tmp = progress_path.with_name('progress-mechanism-pending.json')
    assert not tmp.exists()
    try:
        save(tmp, after)
        assert sha(progress_path) == before_sha, 'progress changed before replacement'
        assert sha(article) == args.article_sha256, 'article changed before replacement'
        tmp.replace(progress_path)
    except Exception as error:
        save(out/'failed-progress-merge.json', {'status':'failed_merge_not_a_completed_progress_record',
            'error_type':type(error).__name__, 'error':str(error), 'expected_progress_sha256':before_sha,
            'actual_progress_sha256':sha(progress_path), 'completed_diagnostic_manifest_exists':manifest_path.exists(),
            'pending_progress_file_exists':tmp.exists(), 'backup':str((out/'progress-before.json').relative_to(LAB).as_posix())})
        raise
    actual = read('manifests/progress.json')
    assert actual == after and actual['results'][:46] == before['results']
    for relative, expected in PROTECTED.items(): assert sha(LAB / relative) == expected
    receipt = {'status': 'passed_progress_merge', 'before_count': 46, 'after_count': 47,
        'all_46_original_results_preserved_in_order': True, 'original_unknown_fields_preserved': True,
        'before_sha256': before_sha, 'after_sha256': sha(progress_path), 'manifest_sha256': sha(manifest_path),
        'article_sha256': sha(article), 'protected_old_document_hashes_unchanged': True,
        'source_sha256': sha(Path(__file__)), 'backup': str((out/'progress-before.json').relative_to(LAB).as_posix())}
    save(out/'receipt.json', receipt)
    print(json.dumps({'status':receipt['status'], 'output':str(out), 'result_count':47}, ensure_ascii=False))

if __name__ == '__main__': main()
