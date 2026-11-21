"""Persist the actual complete-frame visual review, not automatic acceptance."""
from datetime import datetime, timezone
import numpy as np
import audit_gaussianavatars_params123 as a


def main():
    root = a.REC / 'demo-vhap-initialization'
    run = root / 'smplify-vhap-single-landmark-weight'
    control = root / 'smplify-vhap-normalized-laplacian'
    report = a.read(run / 'comparison/result.json')
    sheets = a.read(run / 'comparison/contact-sheet-manifest.json')
    assert report['frames'] == 59 and report['images'] == 295 and len(sheets['pages']) == 30
    for page in sheets['pages']:
        assert a.sha(page['path']) == page['sha256']
    with np.load(a.lp(run / 'initial/tracked-smplx-params.npz'), allow_pickle=False) as current, \
         np.load(a.lp(control / 'initial/tracked-smplx-params.npz'), allow_pickle=False) as previous:
        assert set(current.files) == set(previous.files)
        assert all(np.array_equal(current[k], previous[k]) for k in current.files)
    stamp = datetime.now(timezone.utc).isoformat()
    findings = [
        '完整59帧、全身/面部/双手/脚部共295张原生裁片已实际审阅；30页检查总览全覆盖，另核对18、33、59帧原生全身。图内无文字，记录图片未缩放或扭曲。',
        '确认关键点权重重复：分组函数内部各乘10，外层再乘10；原版VHAP只加权一次。本轮仅取消第二次乘10，同一SMPLify初始化的全部保存数组逐值相同，没有续用任何优化器。',
        '与上一轮归一化平滑直接对照，IoU中位88.94%→88.90%，未覆盖前景7.40%→5.83%，但区域外占比4.53%→5.53%，58/59帧越界指标变差。更大覆盖不等于形状正确。',
        '面部检测残差中位汇总0.925→2.159px、脚部2.919→7.380px，两项59帧均退化；左手1.347→3.020px、右手3.374→5.353px。肘部7.312→12.011px，>50px帧6→7。',
        '实际可见的问题：早段伸臂侧视与正面过渡仍有异常肘形；部分嘴唇和鼻部贴合变差；鞋尖/鞋面越界明显，仍保留裸脚模板特征；袖口、衣摆、帽檐没有形成可靠衣物外形。口袋中的手没有可见三维真值，不把检测器误差当验收。',
        '共享offset中位0.927→0.108mm、P90为10.100→7.244mm；沿模板法线外移超过1mm的顶点2032→1567。当前手臂由共享体型beta引起的位移中位14.38mm，offset中位仅0.92mm，说明本轮更多变化发生在体型参数，不能叫衣物外扩成功。',
        '本轮2270更新、283.91秒完成，仅是VHAP阶段；全流程15分钟目标未达成。取消重复权重虽在公式上恢复单次加权，但这份结果不采用，不准入高斯、不覆盖历史模型。'
    ]
    nxt = ('保留归一化平滑的较好控制及本轮失败证据；下一步先核验共享体型beta、身份先验与衣物offset的职责，'
           '再准备只涉及原版参数组的单变量方案。不要继续减弱全部关键点、不重复手部关闭或邻帧初值试验，不用外扩掩盖原始肘部错误。')
    review = dict(updated=stamp, state='complete_quality_rejected', actual_visual_review=True,
                  quality_accepted=False, frames=59, images=295, inspection_pages=30,
                  reviewed_frames=[r['frame_id'] for r in report['per_frame']], reviewed_pages=sheets['pages'],
                  native_body_extra_review=['frame_000018','frame_000033','frame_000059'],
                  summary_zh='重复关键点权重已定位；取消重复乘法的完整59帧对照退化，不采用。',
                  findings_zh=findings, next=nxt, gaussian_training_started=False,
                  checkpoint_sha256=report['checkpoint_sha256'], initial_arrays_bitwise_equal_to_control=True,
                  same_original_camera_and_observations=True, production_defaults_changed=False,
                  causal_evidence=str(control / 'joint-force-audit/result.json'),
                  plan_metadata_note='Registered plan inherited the preceding operator-audit causal_evidence field. The current hypothesis/changes/source hashes are correct; this review supplies the current joint-force evidence pointer without rewriting the preregistration.',
                  limits=['A confirmed duplicate multiplier is not proof that removing it alone improves reconstruction.',
                          'Template normals and displacement are not physical clothing ground truth.',
                          'All-frame image review is not an unobserved-view geometry validation.'])
    a.save(run / 'comparison/visual-review.json', review)
    report.update(actual_visual_review=True, quality_accepted=False, state='complete_quality_rejected')
    a.save(run / 'comparison/result.json', report)
    sheets['actual_visual_review'] = True
    a.save(run / 'comparison/contact-sheet-manifest.json', sheets)
    plan_path = a.R / 'long-term-plan.md'
    content = a.lp(plan_path).read_text(encoding='utf-8')
    heading = '## 2026-11-01 单次关键点权重对照'
    paragraph = ('\n\n'+heading+'\n\n完整59帧只读当前损失/梯度核算28.91秒、零更新，分组损失与原总项数值闭合。'
                 '确认SMPL-X分支关键点分组已乘权重，外层再次乘10，区别于原版VHAP单次加权。仅取消重复外层因子的独立完整试验2270步、283.91秒完成。'
                 '全部59帧295图和30页总览实际审阅：IoU88.94%→88.90%，脸0.925→2.159px、脚2.919→7.380px均59帧退化；offsetP9010.10→7.24mm。'
                 '体型beta的手臂形变中位14.38mm而offset仅0.92mm，更多覆盖来自体型变化。拒绝这份质量结果，不修改生产默认、不启动高斯。'+nxt+'\n')
    assert heading not in content
    a.lp(plan_path).write_text(content+paragraph,encoding='utf-8')
    print(review['summary_zh'])


if __name__ == '__main__':
    main()
