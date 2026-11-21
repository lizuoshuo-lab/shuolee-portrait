"""Collect completed City metrics for the knowledge page and editable slides.

Read-only with respect to all experiment outputs. Refuses partial evidence.
Does not render, train, launch auditors, or assert a visual boundary conclusion.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/large-scene-gs-001'
EVALS = BASE / 'independent-evaluation'
RUN = EVALS / 'small-city006-heldout-1404-001'
POST = EVALS / 'small-city006-heldout-posterior-CPU-001'
CPU = EVALS / 'small-city006-heldout-perceptual-CPU-001-actual1404'
BOUNDARY = EVALS / 'small-city006-heldout-boundary-CPU-001'
CHAIN = BASE / 'research/city-heldout-postprocessing-018/outcome.json'
OUTPUT = BASE / 'research/city-final-comparison-020'
MODELS = ('merged-unoptimized', 'merged-post')
SETTINGS = ('leaves', 'tau1', 'tau3', 'tau5', 'tau6', 'tau15')
QUALITIES = ('psnr_numpy_mean_per_image_db', 'ssim_mean_per_image', 'lpips_alex_mean_per_image')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-readiness', action='store_true')
    args = parser.parse_args()
    chain = read(CHAIN)
    ready = chain['status'] == 'CPU_verifiers_complete_manual_review_and_delivery_pending'
    if args.check_readiness:
        print(json.dumps(dict(ready=ready, chain_status=chain['status'], writes=False)))
        return
    require(ready, 'Wait for the existing GPU/CPU chain; no partial aggregate is written')
    require(not OUTPUT.exists(), 'Output exists; inspect it instead of overwriting')
    sources = {}

    def source(path):
        sources[str(path)] = sha(path)
        return read(path)

    source(CHAIN)
    actual = source(RUN / 'outcome.json')
    require(actual['status'] == 'complete117_each_requested_actual_local_pre_post_setting'
            and actual['cuda_used'] is True, 'Full actual rendering is incomplete')
    require(chain['evaluation_process_exit'] == 0 and len(chain['steps']) == 3
            and all(s['status'] == 'completed' and s['exit_code'] == 0 for s in chain['steps']),
            'Dependency chain lacks successful real process exits')
    require(chain['evaluation_outcome_sha256'] == sha(RUN / 'outcome.json'), 'Evaluation identity changed')
    for step in chain['steps']:
        path = EVALS / step['run'] / 'outcome.json'
        source(path)
        require(sha(path) == step['outcome_SHA'], 'CPU result changed since chain completion')
    posterior = source(POST / 'completion-audit.json')
    require(posterior['status'] == 'actual1404_complete_independent_CPU_posterior_audit_passed'
            and posterior['actual_verified_frames'] == 1404, 'Independent full posterior unavailable')
    overall = source(RUN / 'metrics.json')
    require(sha(RUN / 'metrics.json') == posterior['source_overall_metrics_sha256'], 'Overall metrics changed')
    require(overall['complete_full_leaves_tau1_3_5_6_15_protocol'] is True, 'Six settings are incomplete')
    cpu = source(CPU / 'perceptual-recalculation.json')
    require(cpu['status'] == 'actual1404_CPU_SSIM_AlexLPIPS_recalculation_completed'
            and len(cpu['per_image']) == 1404, 'Full CPU perceptual check unavailable')
    boundary = source(BOUNDARY / 'report.json')
    require(boundary['status'] == 'completed_actual_city_geometry_selected_original_boundary_image_index'
            and boundary['actual_original_frame_sets'] == 32 and boundary['fixed_patch_count'] == 24,
            'Fixed original boundary evidence incomplete')
    proofs = {(s['model'], s['setting']): s for s in posterior['summaries']}
    rows = []
    for model in MODELS:
        for setting in SETTINGS:
            folder = RUN / model / setting
            metric = source(folder / 'metrics.json')
            timing = source(folder / 'timings.json')
            require(sha(folder / 'metrics.json') == proofs[(model, setting)]['actualmetrics_sha256'],
                    'Setting metrics changed since posterior')
            require(metric['view_images'] == timing['valid_frames'] == 117
                    and metric['timing'] == timing, 'Timing detached from complete setting')
            stages = timing['stages']
            r = dict(model=model, setting=setting, views=117,
                     threshold_training_scope=timing['threshold_training_scope']['all_views_in_training_range'],
                     quality={k: metric[k] for k in QUALITIES},
                     stage_p50_ms={k: v['p50'] * 1000 for k, v in stages.items()},
                     stage_p95_ms={k: v['p95'] * 1000 for k, v in stages.items()},
                     selected_tree_G=timing['selected_tree_G'],
                     torch_peak_allocated_GiB=timing['max_peak_allocated_bytes'] / 2**30,
                     torch_peak_reserved_GiB=timing['max_peak_reserved_bytes'] / 2**30,
                     resident_memory=timing['resident_memory'])
            require(all(math.isfinite(v) for v in r['quality'].values()), 'Nonfinite quality')
            rows.append(r)
    indexed = {(r['model'], r['setting']): r for r in rows}
    for r in rows:
        leaf = indexed[(r['model'], 'leaves')]
        key = 'sum_synchronized_GPU_stages_seconds'
        r['p50_stage_time_ratio_to_same_model_leaves'] = r['stage_p50_ms'][key] / leaf['stage_p50_ms'][key]
        r['mean_selected_G_ratio_to_same_model_leaves'] = r['selected_tree_G']['mean'] / leaf['selected_tree_G']['mean']
    patch_comparisons = []
    for setting in ('leaves', 'tau3'):
        for key in ('raw_RGB_MSE', 'official_GTm2_predm_patch_MSE',
                    'symmetric_single_mask_patch_MSE'):
            pairs = []
            for view in boundary['views']:
                for patch in view['fixed_patch_diagnostics']:
                    d = patch['RGB_patch_diagnostics']
                    require(key in d['merged-unoptimized/' + setting]
                            and key in d['merged-post/' + setting], 'Missing fixed patch diagnostic')
                    a = d['merged-unoptimized/' + setting][key]
                    b = d['merged-post/' + setting][key]
                    if a is not None and b is not None:
                        pairs.append((a, b))
            if pairs:
                patch_comparisons.append(dict(setting=setting, diagnostic=key, valid_patches=len(pairs),
                                              improved=sum(b < a for a, b in pairs),
                                              equal=sum(b == a for a, b in pairs),
                                              worsened=sum(b > a for a, b in pairs),
                                              mean_pre=statistics.mean(a for a, b in pairs),
                                              mean_post=statistics.mean(b for a, b in pairs)))
    result = dict(status='complete_numeric_comparison_manual_visual_review_still_separate',
                  created_utc=datetime.now(timezone.utc).isoformat(), rows=rows,
                  paired_merge_stage_comparison=overall['paired_merge_stage_comparison'],
                  camera_list_group_metrics=source(RUN / 'camera-list-group-metrics.json'),
                  CPU_perceptual_check={k: cpu[k] for k in
                                        ('status', 'max_abs_CPU_GPU_SSIM_delta', 'max_abs_CPU_GPU_AlexLPIPS_delta')},
                  fixed_patch_comparisons=patch_comparisons, manual_visual_review_complete=False,
                  limits_zh=[
                      '117 个留出视角 × 两种模型 × 六种设置；公开 SfM 可能使用全部照片。',
                      'τ=1、3、5 位于本次训练阈值范围，τ=6、15 为范围外诊断。',
                      '阶段耗时是逐段 CUDA 同步后的和，排除文件写入、CPU 审计和指标计算；不是连续播放帧率。',
                      '桌面负载未隔离，单次时延分布不能作为严格的硬件性能排名。',
                      'Torch 峰值不覆盖全部原生临时分配；整树及共享背景常驻，无流式加载。',
                      '误差遵循原实现 GT 二次掩膜、预测一次掩膜的全图计算；对称掩膜辅助指标单列。',
                      '24 个局部区域由稀疏几何预先选定，不能代表全部接缝或证明误差成因。',
                      '三块历史训练与一块只读观测训练混合，原失败保留；不宣称四块同条件正式复现。'],
                  source_files_sha256=sources)
    require(all(sha(Path(p)) == h for p, h in sources.items()), 'A consumed source changed during collection')
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / 'summary-source.py').write_bytes(Path(__file__).read_bytes())
    (OUTPUT / 'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=result['status'], path=str(OUTPUT / 'comparison.json'), rows=len(rows))))


if __name__ == '__main__':
    main()
