"""Publish reviewed local knowledge and terminal evidence; no experiment execution."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / 'outputs/rgbd-affine-icp-fallback-report25-draft-001'
PUBLIC = ROOT / 'outputs/rgbd-affine-icp-fallback-comparison-001'
CHECK = ROOT / 'outputs/rgbd-affine-icp-fallback-public-field-check-001'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new, 1)


def main():
    assert sha(PUBLIC / 'summary.json') == '9701897445669ddbd2fa9b11d6140e2d253c9abc4cba4f33dc10619e3e1cbb3f'
    assert sha(CHECK / 'metrics.json') == '80cdfdbcdbb259f9db41c3fe79db49a6ea9295cc03ef0e24ae8fb41b61c9cb55'
    checked = read(CHECK / 'metrics.json')
    assert checked['status'] == 'passed_independent_stdlib_public_hybrid_field_preservation'
    assert checked['frame_rows_exactly_preserved'] == 1297
    assert checked['original_history_rows_bound'] == 33900
    assert checked['original_history_trials_bound'] == 35746
    summary = read(PUBLIC / 'summary.json')
    proof = {
        'status': 'passed_root_terminal_and_independent_public_field_review',
        'summary_sha256': sha(PUBLIC / 'summary.json'),
        'field_check_sha256': sha(CHECK / 'metrics.json'),
        'actual_experiment_audit_sha256': summary['actual_independent_audit_sha256'],
        'frame_rows': 1297, 'original_history_rows': 33900, 'original_trials': 35746,
        'scope': 'Root source review plus exact persisted field checker; not a new solver, GT formula, or accuracy claim',
        'reviewed_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    with (CHECK / 'root_public_summary_review.json').open('x', encoding='utf-8') as stream:
        json.dump(proof, stream, ensure_ascii=False, indent=2)

    report = (DRAFT / 'report25-pending.md').read_text(encoding='utf-8-sig')
    assert sha(DRAFT / 'report25-pending.md') == '2b3064f25506a238c1499bc2944aa4d1e3f3e3e39c014d1bf7af49c8599112f7'
    report = report.split('\n\n', 2)[2]
    report = report.replace('## 已存在的小体积来源', '## 可核对的来源')
    report = report.replace('独立字段检查待主代理完成后再补实际链接与状态。', '独立字段保留检查已实际通过，覆盖所有帧、历史绑定和原始输入哈希。')
    report = report.replace('本草稿链接以当前 outputs 目录为基准，正式移动到 experiments 时需转换。', '')
    report = report.replace('不能替代另一次独立公开字段复核。', '独立公开字段复核已另行执行，通过全部 1,297 行及其原始绑定。')
    report = re.sub(r'\]\(\.\./(rgbd-[^)]+|tum-[^)]+)\)', r'](../outputs/\1)', report)
    report = report.replace('](../../scripts/', '](../scripts/').replace('](../../experiments/', '](')
    front = ('---\nid: "gslab-exp-affine-icp-fallback-controls-20261009"\nslug: "affine-icp-fallback-controls"\n'
             'title: "25 相邻 ICP 回退：解除停滞与定位误差的取舍"\norder: 25\n'
             'status: "two_full_CPU_paths_and_public_fields_independently_checked"\nupdated: "2026-10-09"\n'
             'sources:\n  - "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats"\n'
             '  - "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/tools"\n---\n\n')
    report = front + '# 相邻 ICP 回退：解除停滞与定位误差的取舍\n\n' + report
    report += ('\n## 完整分页与字段保留\n\n'
               '[独立字段核对](../outputs/rgbd-affine-icp-fallback-public-field-check-001/metrics.json)以标准库逐行比较原终态数据、完整传感器记录、最终候选、四条原误差数组和历史哈希，未运行配准或重新评价 GT。全部 1,297 行通过，2,698 份消费文件保持；其中 2,672 份是原 collector 输入。'
               ' [主代理核对](../outputs/rgbd-affine-icp-fallback-public-field-check-001/root_public_summary_review.json)记录其范围。\n\n')
    for scene in ('desk', 'xyz'):
        entries = summary['scenes'][scene]['new_hybrid']['public_frame_pages']
        report += scene + ' 帧页：' + ' · '.join(
            f"[{e['first_frame_id']}–{e['last_frame_id']}](../outputs/rgbd-affine-icp-fallback-comparison-001/{e['path']})" for e in entries) + '\n\n'
    report += '输入哈希：' + ' · '.join(
        f"[分页 {i+1}](../outputs/rgbd-affine-icp-fallback-comparison-001/{e['path']})" for i, e in enumerate(summary['input_hash_pages'])) + '\n\n'
    report += ('网站对这 12 份帧页仍执行本地路径清洗，随后保留紧凑 JSON，并再次检查 1 MiB 上限。原科学证据的 source SHA 与网站 served SHA 分别记录；网页排版变化不被声称为原文件字节完全相同。\n')

    knowledge_path = ROOT / 'knowledge/17_rejection_reference_recovery.md'
    knowledge = knowledge_path.read_text(encoding='utf-8')
    knowledge = replace_once(knowledge, 'status: "complete_sequence_evidence_and_recovery_design_boundaries"',
                             'status: "complete_sequence_and_affine_ICP_fallback_evidence_independently_audited"')
    old = knowledge.split('\n\n短段和完整序列', 1)[1].split('\n\n', 1)[0]
    knowledge = replace_once(knowledge, '短段和完整序列' + old,
        '短段、完整 last-accepted 轨迹与新的 primary/相邻 ICP fallback 都已有固定实现和真实序列证据。原完整反例见[实验 24](../experiments/2026-10-09-full-affine-reference-controls.md)，新的两条完整回退路径见[实验 25](../experiments/2026-10-09-affine-icp-fallback-controls.md)。新路径在 desk、xyz 都没有最终 hold，但这没有保证定位更准确；本页将状态机方向、尝试失败和完整轨迹误差分别解释。')
    knowledge = replace_once(knowledge, '## primary / fallback / hold：先把设计写成状态机',
                             '## primary / fallback / hold：固定实现与真实接受路径')
    start = '下面只定义一种可审核的设计：'
    old = start + knowledge.split(start, 1)[1].split('\n\n', 1)[0]
    knowledge = replace_once(knowledge, old,
        '新的[固定回退管线](../scripts/track_rgbd_affine_icp_fallback.py)让八维 primary 对齐 current 与最后一次最终接受的参考；primary 被拒绝后，才允许选择 current 到立即前一传感器帧的合格 ICP 相对候选。两路各自保存候选、传感器 ID 和门槛。优先级预先固定，GT 不参与选择。这里复用已完成完整 ICP 的相邻相对候选，没有使用其绝对位姿或将其作为 primary 初始化。')
    knowledge = replace_once(knowledge, 'fallback 接受后将 held reference 设为 k 是这里显式指定的设计选择，不是数学定理。',
        '在这份固定实现中，fallback 接受后将 held reference 设为 k 是明确的状态转移规则，不是数学定理。')
    start = '记录也需要区分“尝试失败”'
    old = start + knowledge.split(start, 1)[1].split('\n\n', 1)[0]
    knowledge = replace_once(knowledge, old,
        '记录区分尝试失败与最终 hold。每个非首帧保存 primary.accepted、primary.failure_flags、fallback.eligible、fallback.ineligible_reasons、最终 selection/accepted/failure_flags，以及 primary_rejected、fallback_selected、参考 before/after、两个候选和传感器记录。相对 ICP 证据提前存在，只有 primary 拒绝后才被选择。primary 失败而 fallback 接受时，尝试 flags 保留，最终 flags 为空；不能把原单路判据套到两路尝试 flags 的并集上。')
    start = '亮度参数 a、b 不是位姿'
    old = start + knowledge.split(start, 1)[1].split('\n\n', 1)[0]
    knowledge = replace_once(knowledge, old,
        '亮度参数 a、b 不是位姿，不能参与 c2w 乘法。八维 primary 每对从 identity/a=0/b=0 开始，未把失败 primary 的亮度估计传给纯几何回退。两条完整 CPU 路线已执行并通过独审，但最终接受不等于位姿准确。这次均无双拒绝，真实结果没有验证双拒绝之后遗漏的历史运动如何恢复。')
    proposal = (DRAFT / 'k17-patch-proposal.md').read_text(encoding='utf-8')
    section = proposal.split('### 实测：', 1)[1].split('\n## 6.', 1)[0]
    knowledge = replace_once(knowledge, '## hold 必须进入全帧评价与审计',
        '## 实测：' + section + '\n平移表列统一为 mm；旋转 RPE 为度。\n\n## hold 必须进入全帧评价与审计')
    knowledge += ('\n[新的回退结果独审](../outputs/rgbd-affine-icp-fallback-validation-001/independent_affine_icp_fallback_result_audit.json)另行核对 1,297 帧、33,900 条 primary 步骤与 35,746 条试探，检查传感器身份、分支选择、保存步进和完整轨迹。它显式复用既有数学核查，不是另一套全新求解器。公开分页保留每帧最终候选，中间试探留本地并以条数和哈希绑定；旧六维字段补充不属于本次轨迹重算。\n')

    target = ROOT / 'experiments/2026-10-09-affine-icp-fallback-controls.md'
    assert not target.exists()
    for path, text in [(target, report), (knowledge_path, knowledge)]:
        for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', text):
            if '://' in link: continue
            p = (path.parent / link).resolve()
            assert p.is_relative_to(ROOT) and (p == target or p.is_file()), str(p)
            if p.is_file() and p.suffix != '.md': assert p.stat().st_size < 1048576, str(p)
    target.write_text(report, encoding='utf-8', newline='\n')
    knowledge_path.write_text(knowledge, encoding='utf-8', newline='\n')

    progress_path = ROOT / 'manifests/progress.json'
    patch = read(DRAFT / 'progress-entry-proposal.json')
    assert sha(progress_path) == patch['base_sha256']
    progress = read(progress_path)
    progress.update(patch['replace_top_level'])
    for change in patch['update_existing_results']:
        row = next(x for x in progress['results'] if x['id'] == change['id'])
        assert row[change['replace_field']] == change['from']
        row[change['replace_field']] = change['to']
    for new in patch['append_results']:
        assert not any(x['id'] == new['id'] for x in progress['results'])
        for scene, fields in new['metrics']['scenes'].items():
            for key, value in fields.items(): assert value == summary['scenes'][scene]['new_hybrid'][key]
        new['artifacts'].append('outputs/rgbd-affine-icp-fallback-public-field-check-001/metrics.json')
        progress['results'].append(new)
    progress['results'].append({
        'id': 'gaussian-map-pose-correction-cpu-controls-001',
        'title': '轨迹更正与高斯地图坐标一致性',
        'status': 'synthetic_matrix_controls_and_independent_math_review_passed',
        'metrics': {'passed_controls': 58, 'renderer_calls': 0, 'optimizer_steps': 0, 'GPU_calls': 0},
        'artifacts': ['knowledge/18_gaussian_map_pose_correction.md', 'outputs/gaussian-map-pose-correction-cpu-controls-001/metrics.json',
                      'outputs/gaussian-pose-transport-independent-source-review-001/metrics.json'],
        'scope': '固定 float64 合成刚体、协方差、四元数与一阶投影；跨锚点接缝反例。未读取真实地图或 GT、未验证光栅化、SH、回环或 BA。',
    })
    progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'status': 'published_local_reviewed_notes', 'report_sha256': sha(target), 'knowledge17_sha256': sha(knowledge_path),
                      'progress_results': len(progress['results']), 'root_public_review_sha256': sha(CHECK / 'root_public_summary_review.json')}))


if __name__ == '__main__':
    main()
