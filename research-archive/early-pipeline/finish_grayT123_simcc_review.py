"""Publish evidence from the completed optional-decoder control, no new fit."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=T/'shuolee/current/local-clothed-surface-123';R=T/'shuolee/current/robust-portrait-122'
O=REC/'grayT-simcc-decoder-review';C=H.parent/'shuolee-coordination'
LOG=Path('D:/00/工作日志/2026-10-28/人物自由表面训练123')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
stamp=datetime.now(timezone.utc).isoformat();result=read(O/'result.json');controls=read(O/'controls.json')
assert result['inferences']==12 and result['original_decoder_max_parity_px']==0
assert result['original_score_max_difference']==0 and controls['all_passed']
assert all(all(r['nose_valid']) for r in result['records'])
review=dict(updated=stamp,actual_visual_review=True,frames=[40,48,56,64,72,80],
    reviewed_images=[dict(path=str(O/'all6_native_decoder_control.png'),sha256=sha(O/'all6_native_decoder_control.png'))],
    summary='已实际审阅6帧原生160px裁片的原图、原解码青点与连续解码橙点。候选只微调鼻点位置，整体语义布局几乎不变。后段侧脸仍存在接近可见轮廓或属于远侧推断的点，连续峰位和数值有效标记不能判定解剖可见性或同一材料身份。未获得独立的逐点误差改善证据，不把此开关安装成已验收修复。',
    raw_crops_have_no_burned_text=True,image_shape_preserved=True,visible_subject_alignment_fixed=False,
    independent_accuracy_ground_truth_available=False,quality_accepted=False)
save(O/'actual-review.json',review)
summary=('SMPL-X仍未对齐。现有导出包明确将RTMW的连续SimCC细化关闭，已在保持原图、bbox、RGB归一化、模型、置信度、翻转和相机完全不变的条件下做6帧12次CPU推理。'
    '原argmax坐标与旧观测精确一致，独立逆仿射最大差0.0000953像素，未发现这一步坐标映射错误。'
    '连续细化改变54个鼻点位置的中位幅度0.843像素、最大1.816像素。实际原生图比较仍有侧脸轮廓/遮挡推断点，不能把连续浮点坐标当成亚像素材料真值。'
    '这仅确认解码量化的一项可控影响，没有证明真实定位误差降低，也没有解决此前数十像素的SMPL-X错位。相机、模板形状、跨帧运动和语义对应仍未唯一分离。未修改观测、相机或历史模型，未启动新拟合/训练。')
next_text=('解码变量已经有界核验，保留原始SimCC响应，不再重复推理或开启DARK后直接训练。'
    '下一步读取现有SMPL-X面部语义点的表面支持、前后可见性和损失调用，确认遮挡点进入损失的具体路径，'
    '对观测可见性单独制定并验证保守的支持合同；不能用当前错位网格或SAM mask冒充解剖真值。'
    '若缺少能区分可见材料和遮挡推断的独立证据，保留未知并停止相应精细约束，不靠换阈值获得通过。')
con=dict(updated=stamp,summary=summary,next=next_text,frames=result['frames'],model_inferences=12,
    original_observation_parity_px=0.,independent_inverse_affine_max_px=result['independent_inverse_affine_max_px'],
    nose_candidate_shift_native_px=result['nose_candidate_shift_native_px'],controls_passed=True,
    actual_native_image_review=True,physical_accuracy_validated=False,quality_accepted=False,
    no_camera_or_geometry_change=True,no_training=True,
    metadata_correction='导出包旧detail/preprocess残留192x256，而实际ONNX与SimCC解码均384x288。此次以真实形状与精确观测复放核验，不能据旧文本断言模型输入错配。',
    serialization_issue='12次推理和result.json均已完成，图片目录index.json因NumPy整型裁剪坐标写入失败；只从保存响应重算核对并修复图片目录，没有新增推理，原result.json保持。',
    sources=[dict(path=str(p),sha256=sha(p)) for p in [O/'plan.json',O/'controls.json',O/'result.json',O/'raw-responses.npz',O/'actual-review.json',O/'serialization-recovery.json']])
save(REC/'simcc-decoder-conclusion.json',con);save(LOG/'simcc-decoder-conclusion.json',con)
key='grayT123-simcc-continuous-decoder-control'
for base in [REC,R]:
    p=base/'evidence-ledger.json';d=read(p)
    if base==REC:
        assert not any(x.get('id')==key for x in d['entries'])
        d['entries'].append(dict(id=key,updated=stamp,summary=summary,evidence=str(REC/'simcc-decoder-conclusion.json'),quality_accepted=False))
    else:
        if summary not in d['confirmed']:d['confirmed'].append(summary)
        d['artifacts'].append(str(REC/'simcc-decoder-conclusion.json'))
    d['updated']=stamp;save(p,d)
q=read(REC/'queue-state.json');q.update(updated=stamp,current_stage='simcc-decoder-control-completed',
    reason=summary,latest_diagnosis=str(REC/'simcc-decoder-conclusion.json'),next_action=next_text,next_plan=None,
    state='active_alignment_diagnosis',training_started=False,training_allowed=False,training_running=False,
    local_GPU_task_active=False,worker_pid=None,worker_command=None,goal_stays_active=True)
save(REC/'queue-state.json',q)
n=read(REC/'next-action.json');n.update(updated=stamp,next=next_text,next_plan=None,
    latest_diagnosis=str(REC/'simcc-decoder-conclusion.json'),actual_UI_verified=False,diagnostic_DOM_verified=False)
save(REC/'next-action.json',n)
c=read(C/'local-research-status.json');c.update(updated_at=stamp,current_local_decoder_control=con,
    next=next_text,local_GPU_task_active=False);save(C/'local-research-status.json',c)
long=T/'shuolee/long-term-plan.md'
with long.open('a',encoding='utf-8') as f:f.write('\n\n## 2026-10-28 原始SimCC与连续解码单因素核验\n\n'+summary+'\n\n'+next_text+'\n\n证据：'+str(REC/'simcc-decoder-conclusion.json')+'\n')
save(REC/'simcc-decoder-completion.json',dict(updated=stamp,state='scientific_control_reviewed_publication_pending',
    actual_native_image_review=True,actual_browser_UI_review=False,PPT_updated=False,
    quality_accepted=False,goal_active=True,training_started=False))
for f in ['queue-state.json','next-action.json','simcc-decoder-completion.json']:
    shutil.copy2(REC/f,LOG/f)
print(json.dumps({k:v for k,v in con.items() if k not in ['sources']},ensure_ascii=False))
