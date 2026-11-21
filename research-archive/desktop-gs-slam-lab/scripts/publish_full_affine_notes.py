"""Publish reviewed complete-sequence notes locally; never train or host remotely."""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/rgbd-affine-full-comparison-001'
DRAFT = ROOT / 'outputs/rgbd-affine-full-report24-draft-001'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))

def main():
    assert sha(OUT / 'summary.json') == 'd6a5f9e304579e98b0b97912762eb6c7618273c89398bf3c95e09809e76635d3'
    guard = ROOT / 'outputs/rgbd-affine-full-6d-proposal-addendum-001/metrics.json'
    assert sha(guard) == '1d75beecb78ed33a93c1072ddb93e6e494ccdc5a1f3b2ebb3f95887808642a00'
    review = read(OUT / 'root_public_summary_review.json')
    assert review['status'] == 'passed_independent_full_public_field_preservation_review'
    assert review['complete_frame_rows'] == 2594
    report = (DRAFT / 'report24-pending.md').read_text(encoding='utf-8-sig')
    assert sha(DRAFT / 'report24-pending.md') == 'eb79a476debabd4f2fd0e80f20de7d260fd25e18e149962de6110fffa66993f9'
    report = report.split('## 完整序列保留所有拒绝帧', 1)[1]
    report = '## 完整序列保留所有拒绝帧' + report.split('## 已存在的材料路径', 1)[0]
    report = report.replace('正式材料将链接小体积协议', '下方材料包含小体积协议')
    report = re.sub(r'\]\(\.\./(rgbd-[^)]+)\)', r'](../outputs/\1)', report)
    report = report.replace('](../../scripts/', '](../scripts/')
    intro = ('四轮完整 CPU 对照保留全部 2,594 个关联帧。desk 的六维与八维模型都在 134–548 连续拒绝 415 帧，参考停留在 133；八维没有解除停滞。xyz 的八维位置类指标更高，相邻旋转 RPE 略低。短片段成功、局部满秩与进程正常结束都不能代替完整序列稳定性的证据。\n\n'
             '原独立结果审计、真实六维字段补充和公开字段保留检查均已通过。拒绝帧仍进入指标；模型与阈值保持冻结，GT 不参与优化或候选选择。本报告接续[参考策略对照](2026-10-09-last-accepted-reference-recovery.md)、[八维短序列对照](2026-10-09-affine-photogeometric-real-sequences.md)与[恢复边界知识](../knowledge/17_rejection_reference_recovery.md)。\n\n')
    resources = [
        ('outputs/rgbd-affine-full-cpu-preparation-001/protocol.json', '四轮冻结协议'),
        ('outputs/rgbd-affine-full-cpu-preparation-001/root_protocol_review.json', 'root 执行前源码审阅'),
        ('outputs/rgbd-affine-full-cpu-preparation-001/root-independent-full-preflight-review.json', '完整输入与模拟控制独立预检'),
        ('outputs/rgbd-affine-full-validation-001/metrics.json', '四个真实进程终态'),
        ('outputs/rgbd-affine-full-validation-001/independent_full_affine_result_audit.json', '原完整轨迹与保存步进独审'),
        ('outputs/rgbd-affine-full-comparison-001/summary.json', '完整四组汇总'),
        ('outputs/rgbd-affine-full-comparison-001/completion.json', '完成文件大小与 hash 清单'),
        ('outputs/rgbd-affine-full-comparison-001/root_public_summary_review.json', '全部公开字段的 root 保留核对'),
        ('outputs/rgbd-affine-full-comparison-001/input_hashes.json', '5,252 份消费文件索引'),
    ]
    for scene, dim in [('desk', 6), ('desk', 8), ('xyz', 6), ('xyz', 8)]:
        for suffix, label in [('summary', '小摘要'), ('frames', '完整帧表')]:
            resources.append((f'outputs/rgbd-affine-full-comparison-001/{scene}-{dim}d-{suffix}.json', f'{scene} {dim}D {label}'))
    for index in range(6): resources.append((f'outputs/rgbd-affine-full-comparison-001/input-hashes-{index:02d}.json', f'输入校验分页 {index + 1}'))
    for name, label in [('track_rgbd_affine_full.py', '完整序列管线'), ('run_rgbd_affine_full.py', '固定串行驱动'),
                        ('audit_rgbd_affine_full_results.py', '原完整结果独审源码'), ('summarize_rgbd_affine_full_results.py', '纯 JSON 汇总源码')]:
        resources.append(('scripts/' + name, label))
    resources += [('outputs/tum-online-v4-full-sequence-recovery-summary-001/summary.json', '只读旧完整 ICP 与 GS 汇总'),
                  ('outputs/tum-online-v4-full-sequence-recovery-summary-001/independent_summary_audit.json', '旧 ICP 数值审计证据')]
    report += '\n## 可复核材料\n\n' + '\n'.join(f'- [{label}](../{path})' for path, label in resources) + '\n'
    front = ('---\nid: "gslab-exp-full-affine-reference-controls-20261009"\nslug: "full-affine-reference-controls"\n'
             'title: "24 完整序列：光度模型的拒绝链与参考停滞"\norder: 24\n'
             'status: "completed_four_full_CPU_controls_independently_audited"\nupdated: "2026-10-09"\n'
             'sources:\n  - "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats"\n'
             '  - "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/tools"\n'
             '  - "https://www.open3d.org/docs/0.19.0/tutorial/pipelines/rgbd_odometry.html"\n---\n\n')
    report = front + '# 完整序列：光度模型的拒绝链与参考停滞\n\n' + intro + report
    knowledge = (DRAFT / 'knowledge17_draft.md').read_text(encoding='utf-8-sig')
    assert sha(DRAFT / 'knowledge17_draft.md') == '1381c45244d35b416361d72a33fcdab2ba96c160fb96c2765648f7df7c599ca4'
    knowledge = knowledge.replace('status: "draft_not_published"', 'status: "complete_sequence_evidence_and_recovery_design_boundaries"')
    knowledge = knowledge.replace('sources: []', 'sources:\n  - "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats"\n  - "https://www.open3d.org/docs/0.19.0/tutorial/pipelines/rgbd_odometry.html"')
    old = knowledge.split('\n\n草稿范围：', 1)[1].split('\n\n', 1)[0]
    new = ('短段和完整序列的观察来自已完成的固定实现与审计；primary/fallback/hold 部分说明恢复状态机的设计和数学边界，暂不包含该混合策略的真实运行结果。完整数据见[实验 24](../experiments/2026-10-09-full-affine-reference-controls.md)。')
    knowledge = knowledge.replace('草稿范围：' + old, new)
    knowledge = knowledge.replace('](../../scripts/', '](../scripts/').replace('](../../experiments/', '](../experiments/')
    knowledge = re.sub(r'\]\(\.\./(rgbd-[^)]+)\)', r'](../outputs/\1)', knowledge)
    knowledge = knowledge.replace('本草稿仅给出', '本节仅给出')
    targets = [(ROOT / 'experiments/2026-10-09-full-affine-reference-controls.md', report),
               (ROOT / 'knowledge/17_rejection_reference_recovery.md', knowledge)]
    for path, text in targets:
        assert not path.exists(), 'Unique new formal document'
        for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', text):
            if '://' in link: continue
            destination = (path.parent / link).resolve()
            assert destination.is_relative_to(ROOT.resolve())
            if destination not in [p for p, _ in targets]:
                assert destination.is_file(), str(destination)
                assert destination.stat().st_size <= 1048576
    for path, text in targets: path.write_text(text, encoding='utf-8', newline='\n')
    print(json.dumps({'status': 'published_local_markdown_only', 'documents': [{'path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p)} for p, _ in targets]}))

if __name__ == '__main__': main()
