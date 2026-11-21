from pathlib import Path
import json

H=Path(__file__).parent
O=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/smplifyx-reference-control/framewise-canonical-route/shared-three-frame')
review=json.loads((O/'slide-review.json').read_text(encoding='utf-8'))
diagnosis=json.loads((O/'final-state-diagnosis/result.json').read_text(encoding='utf-8'))
assert diagnosis['state']=='complete'
assert abs(diagnosis['objective']-937.5182401569739)<1e-9
review.update(
    title='共享体型拟合的求解与观测核查',
    subtitle='最终解尚未收敛，先修复通用求解检查',
    result='保存解仍有非零梯度，初始化使用了正式拟合禁用的髋点。面部观测待单独核验。',
    caption='历史结果，原视频155帧：左原图，右共享体型SMPL-X叠加。手部保留先验。',
    notes=[
        '本轮仅对已保存的共享体型三帧解做只读诊断，未新增拟合，未启动高斯训练。图像沿用此前三帧试验中的原视频155帧结果，不代表本轮效果改善。',
        '保存状态回放的网格和投影与归档完全一致。最终目标函数937.5182401569739，其中数据项388.26314708434916，先验项549.2550930726247。',
        '局部列缩放后的总梯度最大绝对值为0.2032662979，最终状态尚不满足收敛要求。数据项与先验梯度有抵消，但这种抵消不能证明先验是错误根因。',
        '更正上一页备注中的停止原因：阶段3在第30次外循环按relative_objective退出，只有阶段4以outer_budget退出。阶段执行结束不能作为已收敛的证据。',
        '根姿态初始化的观测资格与正式拟合不一致，初始化使用了正式拟合禁用的髋点。当前正在修复根姿态观测资格及收敛判断。修复后的真实拟合效果尚未验证。',
        '面部观测仍需单独核验。检测器语义点与物理材料对应不同，现有三帧结果没有证明独立物理特征点达到亚像素精度。',
        'Source: https://github.com/vchoutas/smplify-x/tree/68f8536707f43f4736cdd75a19b18ede886a4d53',
        'Local generic sequence adapter, not an official multi-frame release. One shared10-beta tensor, per-frame root/body/jaw/expression, prior-only hands. Camera K and E unchanged.',
        'Evidence: '+str(O/'final-state-diagnosis/result.json'),
        'Figure: '+review['figure'],
    ])
(H/'sequence-diagnosis123-slide-review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
source=(H/'update_deck_shared_sequence123.mjs').read_text(encoding='utf-8')
changes={
    "tag='shared-sequence123'":"tag='sequence-diagnosis123'",
    "b+'/framewise-route123-deck-receipt.json'":"b+'/shared-sequence123-deck-receipt.json'",
    "review=await read(root+'/slide-review.json')":"review=await read(b+'/sequence-diagnosis123-slide-review.json')",
    "replace('逐帧姿态与静态高斯输出'":"replace('共享体型逐帧拟合的三帧结果'",
    "replace('共享体型和高斯，各帧拟合姿态，优先保证面部'":"replace('接口检查通过，实拍对齐仍未通过'",
    "replace('网格绑定数值检查通过。共享体型逐帧拟合尚未完成，高斯0步。'":"replace('面部检测点中位1.87 / 1.17 / 1.42px。脚部仍有偏差，未进入高斯训练。'",
    "replace('原视频155帧：左原图，右SMPL-X叠加。仍有身体和脚部残差。'":"replace('原视频155帧：左原图，右共享体型SMPL-X叠加。手部保留先验。'",
    '02_有限视角重建优化路线_123共享体型三帧适配.pptx':'02_有限视角重建优化路线_123共享体型求解核查.pptx',
    'preserve_deck_shared_sequence123.py':'preserve_deck_sequence_diagnosis123.py',
}
for old,new in changes.items():
    assert old in source,old
    source=source.replace(old,new)
# Preserve the original slide52 image bytes and geometry; the historical caption is explicit.
source=source.replace("if(i===51)await fs.writeFile(b+'/'+tag+'-final-52.png',bytes);else if(h!==prior.renders[i].render_sha256)","await fs.mkdir(b+'/'+tag+'-renders',{recursive:true});await fs.writeFile(b+'/'+tag+'-renders/slide-'+String(i+1).padStart(2,'0')+'.png',bytes);if(i===51)await fs.writeFile(b+'/'+tag+'-final-52.png',bytes);else if(h!==prior.renders[i].render_sha256)")
(H/'update_deck_sequence_diagnosis123.mjs').write_text(source,encoding='utf-8')
print('Prepared one-slide diagnosis edit; canonical remains unchanged.')
