"""Clarify an existing diagnostic; preserve original models and experiment results."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import shutil

H = Path(__file__).parent
T = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R = T / 'shuolee/current/robust-portrait-122'
REC = R.parent / 'local-clothed-surface-123'
G = R / 'video10-global-initializer'
A = REC / 'observation-visibility-consumer-audit'
LOG = Path('D:/00/工作日志/2026-10-29/人物对齐核验123')

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def save(p, v):
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

assert not (A / 'publication.json').exists()
audit = read(A / 'result.json')
assert audit['state'] == 'complete_read_only' and audit['no_training']
stamp = datetime.now(timezone.utc).isoformat()
summary = '蓝点是图像检测目标，红点是SMPL-X网格对应点的相机投影。直接重投影与保存检查点相差不足0.0004px，而灰衣面部检测残差中位43.99px，属于实际拟合不一致。'
limitations = '数值链路一致不能证明物理相机正确。旧身体粗门槛、鼻嘴软约束与未建模解剖可见性已确认；相机、对应、静态假设与模板/优化的贡献尚未唯一分离。'
nexttext = '观测消费只读核验已结束，不重复计算旧权重或旧拟合。下一步在未来拟合入口接入独立观察可见性和材料身份；未知状态不能默认可见。田地仅有未翻转ROI检测，缺少生产拟合输入，当前不补造损失权重、不启动训练。'
title = '为什么检测点与SMPL-X投影不重合'
name = 'grayT_frame076_raw_detector_model.png'
source = REC / 'grayT-smplx-audit-review/frame_000076_native_face_raw_detector_model.png'
shutil.copy2(source, G / name)
section = f'''<section id="projection-explained"><h2>{title}</h2>
<p>{summary}</p>
<table><thead><tr><th>图中对象</th><th>实际来源</th><th>含义</th></tr></thead><tbody>
<tr><td>蓝色检测点</td><td>RTMW在每张照片独立预测的二维鼻嘴位置</td><td>拟合目标，不是已经一致的三维材料点，也不是SMPL-X投影。</td></tr>
<tr><td>红色模型点</td><td>从拟合SMPL-X网格三角面按模板重心坐标取点，再使用本帧K和w2c投影</td><td>模型预测；它与蓝点的距离就是检测器到模型的残差。</td></tr>
</tbody></table>
<p>本链路的检查点投影可直接由同一网格、同一相机重算，三个人物最大数值差小于0.0004px。这里没有发现两份模型投影整体错开的证据；此结论不覆盖任意查看器、导出模型或物理标定。</p>
<figure><img src="{name}" width="870" height="298" style="max-width:100%;height:auto"><figcaption>灰衣拟合帧76，原视频第375帧。从左到右是原图、蓝色检测点、红色SMPL-X对应投影。每格290×298原生像素，图像文件未重新缩放、未添加文字。该帧参与面部点的残差中位100.42px；整体中位43.99px。红点落在脸颊/耳前，拟合未通过。</figcaption></figure>
<p>当前拟合只有29个鼻嘴点生效，眼眉22个模板点被置零；所有视角共享一个人体姿态与形状，表情和下颌未优化。面部是归一化后的鲁棒软目标，系数0.15，与身体、先验等同时折中。不能仅比较这个系数就断言它是唯一原因。</p>
<p>旧门槛仅按身体高度比例验收，灰衣身体中位残差允许约50–53px；没有要求鼻嘴、眼周、脚分别通过。旧实际拟合也没有启用面部解剖可见性，人物mask和检测响应不能证明侧脸或背面预测可见。这些是已确认的验收/观测缺口，不是可接受的精准拟合。</p>
<p>{limitations}</p><p>另一个已有对照：固定相机与观测，将29个点从SMPL-X模板解耦为独立静态三维点，灰衣中位残差仍为47.05px；局部求解并不证明全局下界，但说明“只换模板就一定能对齐”缺乏证据。</p>
<p>本次只读核验确认三个历史人物实际使用的拟合源码与当前冻结源码相同。田地第440帧的68个后脑勺面部预测尚未进入拟合；它缺少翻转差异、mask/相机和最终权重，不能宣称这些预测已经造成了新模型错误。</p>
<p>{nexttext}</p><p><a href="observation-visibility-consumer-audit/result.json">实际消费链与已有结果的范围</a> · <a href="observation-visibility-consumer-audit/source-evidence.json">源码位置与SHA</a></p></section>'''
p = G / 'index.html'
s = p.read_text(encoding='utf-8')
assert 'id="projection-explained"' not in s
s = s.replace('<section id="camera-entry">', section + '<section id="camera-entry">', 1)
p.write_text(s, encoding='utf-8')
destinations = [T/'viewer/diagnostics/video10-global-initializer',
    Path('D:/00/工作/查看器/diagnostics/video10-global-initializer'), LOG/'相机多解与人物对齐']
for dest in destinations:
    assert dest.is_dir()
    shutil.copy2(G/'index.html', dest/'index.html')
    shutil.copy2(G/name, dest/name)
    (dest/'observation-visibility-consumer-audit').mkdir(exist_ok=True)
    for filename in ['result.json', 'source-evidence.json']:
        shutil.copy2(A/filename, dest/'observation-visibility-consumer-audit'/filename)

article = f'<article id="smplx123-video10-global-ambiguity"><h2>{title}</h2><p>{summary}</p><p>{limitations}</p><p><a href="diagnostics/video10-global-initializer/index.html#projection-explained">原图、检测与模型投影对照</a></p><p>{nexttext}</p></article>'
(G/'viewer-article.html').write_text(article, encoding='utf-8')
preserved = []
for folder, filename in [(T/'viewer', 'body-smplx.html'), (Path('D:/00/工作/查看器'), 'progress.html')]:
    p = folder/filename
    s, count = re.subn(r'<article id="smplx123-video10-global-ambiguity">.*?</article>', lambda _: article,
                      p.read_text(encoding='utf-8'), count=1, flags=re.S)
    assert count == 1
    p.write_text(s, encoding='utf-8')
    p = folder/'latest-result.json'
    v = read(p)
    previous = {k: x for k, x in v.items() if k != 'latestDiagnostic'}
    v['latestDiagnostic'] = dict(id='smplx123-video10-global-ambiguity', round='123', updated=stamp,
        title=title, summary=summary, reportUrl=filename+'#smplx123-video10-global-ambiguity', qualityAccepted=False)
    save(p, v)
    assert previous == {k: x for k, x in read(p).items() if k != 'latestDiagnostic'}
    preserved.append(str(p))
p = LOG/'index.html'
s, count = re.subn(r'<section id="video10-global-ambiguity">.*?</section>',
    lambda _: f'<section id="video10-global-ambiguity"><h2>{title}</h2><p>{summary}</p><p>{limitations}</p><p><a href="相机多解与人物对齐/index.html#projection-explained">原图对照与核验说明</a></p></section>',
    p.read_text(encoding='utf-8'), count=1, flags=re.S)
assert count == 1
p.write_text(s, encoding='utf-8')
for root in [R, REC]:
    q = read(root/'queue-state.json')
    q.update(updated=stamp, current_stage='observation_consumer_readonly_complete',
        current_subject='field', active_subject='field', next=nexttext, next_action=nexttext,
        latest_evidence=str(A/'result.json'), latest_diagnosis=str(A/'result.json'),
        training_running=False, training_allowed=False, local_GPU_task_active=False)
    save(root/'queue-state.json', q)
q = read(REC/'next-action.json')
q.update(updated=stamp, current_subject='field', observed_progress=summary,
    next=nexttext, next_action=nexttext, latest_result=str(A/'result.json'),
    latest_diagnosis=str(A/'result.json'), next_plan=None,
    completed_this_turn=['Read-only source identity and consumer trace; no new fitting',
                         'Clarified detector target versus mesh projection in the existing report'],
    actual_UI_verified=False, diagnostic_DOM_verified=False)
save(REC/'next-action.json', q)
q = read(REC/'evidence-ledger.json')
q['entries'].append(dict(id='observation_consumer_projection_semantics', updated=stamp,
    summary=summary, limitation=limitations, evidence=str(A/'result.json'),
    original_experiments_reused=True, new_experiment=False, quality_accepted=False))
save(REC/'evidence-ledger.json', q)
p = H.parent/'shuolee-coordination/local-research-status.json'
q = read(p)
q.update(updated_at=stamp, current_stage='observation_consumer_readonly_complete',
    current_subject='field', next=nexttext, training_running=False,
    current_projection_clarification=dict(result=str(A/'result.json'), source_identity_verified=True,
        historical_errors_reused=True, field_weights_replayed=False, physical_quality_accepted=False))
save(p, q)
save(A/'publication.json', dict(updated=stamp, page_saved=True, diagnostic_UI_checked=False,
    source_image_sha256=sha(source), published_image_sha256=sha(G/name),
    model_metadata_preserved=preserved, historical_models_unchanged=True,
    original_PPT_unchanged=True, reason='Explanation and source provenance audit, not a new experimental result.',
    new_training=False, quality_accepted=False))
print(json.dumps(dict(page=str(G/'index.html'), anchor='projection-explained',
    model_preserved=True, report_updated=True, training=False), ensure_ascii=False))
