from pathlib import Path
import json

H=Path(__file__).parent
O=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/smplifyx-reference-control/framewise-canonical-route/shared-three-frame/final-state-diagnosis')
review=json.loads((H/'sequence-diagnosis123-slide-review.json').read_text(encoding='utf-8'))
fix=json.loads((O/'optimizer-fix-result.json').read_text(encoding='utf-8'))
tests=json.loads((O/'optimizer-cpu-regression.json').read_text(encoding='utf-8'))
assert fix['tests_passed']==8 and tests['tests_run']==8 and tests['failures']==0 and tests['errors']==0
assert fix['real_refit_performed'] is False and fix['gaussian_steps']==0
review['subtitle']='求解检查已修复，实拍改善尚待验证'
review['result']='8项CPU回归通过。初始化只剩两肩，支持不足。下一步核验专用面部观测。'
review['notes']=[s.replace('当前正在修复根姿态观测资格及收敛判断。修复后的真实拟合效果尚未验证。','根姿态观测资格及收敛判断已修复，8项CPU合成回归通过。修复后的真实拟合效果尚未验证。') for s in review['notes']]
review['notes'] += [
    '通用修复已完成：根姿态初始化冻结观测和资格，忽略的观测不进入数据项。每次LBFGS步后重新计算终点损失和梯度，并分别核验各阶段恢复状态及最终状态。阶段执行完毕、最终一阶驻点与数值收敛分别记录。',
    '8项CPU优化控制回归通过，未初始化CUDA，没有新增真实SMPL-X拟合、检测器推理或高斯训练。CPU合成回归不证明实拍对齐改善。',
    '现有观测资格只留下两个肩点，少于3个有效根姿态锚点时在优化前明确停止并保留初始化。通用替代初始化策略另行登记，不静默添加关节。3点数量要求本身不构成可辨识性或几何秩证明。',
    'Evidence: '+str(O/'optimizer-fix-result.json'),
    'Evidence: '+str(O/'optimizer-cpu-regression.json'),
]
(H/'sequence-diagnosis123-fixed-slide-review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
source=(H/'update_deck_sequence_diagnosis123.mjs').read_text(encoding='utf-8')
changes={
    "tag='sequence-diagnosis123'":"tag='sequence-diagnosis123-fixed'",
    "b+'/shared-sequence123-deck-receipt.json'":"b+'/sequence-diagnosis123-deck-receipt.json'",
    "b+'/sequence-diagnosis123-slide-review.json'":"b+'/sequence-diagnosis123-fixed-slide-review.json'",
    "replace('共享体型逐帧拟合的三帧结果'":"replace('共享体型拟合的求解与观测核查'",
    "replace('接口检查通过，实拍对齐仍未通过'":"replace('最终解尚未收敛，先修复通用求解检查'",
    "replace('面部检测点中位1.87 / 1.17 / 1.42px。脚部仍有偏差，未进入高斯训练。'":"replace('保存解仍有非零梯度，初始化使用了正式拟合禁用的髋点。面部观测待单独核验。'",
    "replace('原视频155帧：左原图，右共享体型SMPL-X叠加。手部保留先验。'":"replace('历史结果，原视频155帧：左原图，右共享体型SMPL-X叠加。手部保留先验。'",
    '02_有限视角重建优化路线_123共享体型求解核查.pptx':'02_有限视角重建优化路线_123共享体型求解检查修复.pptx',
    'preserve_deck_sequence_diagnosis123.py':'preserve_deck_sequence_diagnosis123_fixed.py',
}
for old,new in changes.items():
    assert old in source,old
    source=source.replace(old,new)
(H/'update_deck_sequence_diagnosis123_fixed.mjs').write_text(source,encoding='utf-8')
print('Prepared completed-fix wording for slide52; no canonical change.')
