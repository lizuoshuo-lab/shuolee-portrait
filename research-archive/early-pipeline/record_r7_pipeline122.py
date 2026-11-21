from pathlib import Path
from datetime import datetime,timezone
import hashlib,html,json,shutil
H=Path(__file__).parent;T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';O=R/'native-residual-pipeline';REC=T/'shuolee/current/local-clothed-surface-123'
C=H.parent/'shuolee-coordination';L=Path('D:/00/工作日志/2026-10-29/人物对齐核验123')
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
stamp=datetime.now(timezone.utc).isoformat();cpu=read(O/'cpu-result.json');gpu=read(O/'gpu-result.json')
assert all(c['passed'] for c in cpu['checks']+gpu['checks']) and gpu['total_optimizer_updates']==12
summary=('R7原生训练入口已完成22项CPU与14项CUDA工程检查。复现并修复了图像缓存标识被通道名覆盖的问题，重复访问现在复用同一原生张量；原先像素值一致，此错误不是已证实的人脸错位原因。完整K、外参和6+2+4输出残余系数随每帧传入训练，缺失系数不默认为0。96点合成场景共12次优化更新，双视角梯度与独立联合损失最大差3.73e-9，原生NPZ/相机回读及RGB/alpha回放一致。新入口对既有102目标、13个失败脸部观察仍拒绝，最大13.049像素未被删除或放宽。没有新人物拟合或训练，静态SMPL-X对齐仍未通过。')
next_text=('只读审阅00刚完成的FAN逐帧对照18k、全帧验证和退步帧，区分姿态补偿、原相机误差及自由点表面分布；这仍是诊断对照，不直接改成本静态主线。R7已完成的合成检查不重跑。没有新的物理材料/相机证据时不靠重复训练声称修复。')
review=dict(updated=stamp,actual_visual_review=True,quality_accepted=False,scope='synthetic-control-only',
    image=read(O/'synthetic-figure.json'),observations=['4个目标视角及4个原生回读小图实际查看；保持48×40原生像素、无图内文字。','各相机投影位置不同，第二视角可见点仅38/96，已检验两视角统计均被计入。','目标和4步结果允许存在优化残差；回读与同一已训练状态的RGB/alpha逐值一致。','这是96点已知合成场景，不是任何人的重建或对齐证据。'])
save(O/'actual-review.json',review)
con=dict(updated=stamp,summary=summary,next=next_text,CPU_checks=22,GPU_checks=14,CPU_seconds=cpu['seconds'],GPU_seconds=gpu['seconds'],
    source_cache=sha(H/'train_native_person122_v2.py'),source_loop=sha(H/'native_training122_v5.py'),original_failure_preserved=True,
    initial_points_synthetic=96,optimizer_updates_synthetic=12,person_fit=False,person_training=False,physical_quality_accepted=False,
    GPU_context_currently_owned=False,scope='synthetic engineering validation plus actual failed-person input refusal',
    evidence=[str(O/n) for n in ['plan.json','cache-initial-failure.json','cache-repair.json','cpu-result.json','gpu-result.json','actual-review.json']])
save(O/'conclusion.json',con);save(O/'progress.json',dict(updated=stamp,state='engineering_validated_publication_pending',test_executed=True,source_drafts_only=False,person_training=False,quality_accepted=False))
for root in [R,REC]:
 q=read(root/'queue-state.json');q.update(updated=stamp,current_stage='R7_native_training_pipeline_engineering_complete',next_action=next_text,next=next_text,
    latest_diagnosis=str(O/'conclusion.json'),training_allowed=False,training_started=False,quality_accepted=False,
    last_goal_turn=dict(updated=stamp,classification='progress',result=str(O/'conclusion.json'),summary=summary,next=next_text))
 save(root/'queue-state.json',q)
e=read(R/'evidence-ledger.json');assert str(O/'conclusion.json') not in e['artifacts'];e['artifacts'].append(str(O/'conclusion.json'));e['confirmed'].append(summary);e['updated']=stamp;save(R/'evidence-ledger.json',e)
e=read(REC/'evidence-ledger.json');e['entries'].append(dict(id='R7_native_pipeline_controls',updated=stamp,summary=summary,evidence=str(O/'conclusion.json'),quality_accepted=False));e['updated']=stamp;save(REC/'evidence-ledger.json',e)
n=read(REC/'next-action.json');n.update(updated=stamp,next=next_text,next_action=next_text,latest_diagnosis=str(O/'conclusion.json'),next_plan=None,training_started=False,diagnostic_DOM_verified=False);save(REC/'next-action.json',n)
c=read(C/'local-research-status.json');c.update(updated_at=stamp,next=next_text,current_R7_native_pipeline=con,local_GPU_task_active=False);save(C/'local-research-status.json',c)
lp=T/'shuolee/long-term-plan.md';s=lp.read_text(encoding='utf-8');marker='### 2026-10-29 R7原生训练入口完成工程控制';assert marker not in s;lp.write_text(s+'\n\n'+marker+'\n\n'+summary+'\n\n下一步：'+next_text+'\n',encoding='utf-8')
dest=L/'R7原生训练入口';dest.mkdir(exist_ok=True)
for name in ['plan.json','cache-initial-failure.json','cache-repair.json','cpu-result.json','gpu-result.json','actual-review.json','conclusion.json','synthetic-native-comparison.png','synthetic-figure.json']:
 shutil.copy2(O/name,dest/name)
section='<section id="native-pipeline-r7"><h1>静态对齐未通过；原生训练入口完成工程检查</h1><p>'+html.escape(summary)+'</p><p>人像现状：新面部观测对照的同一35个皮肤点留出仍有32个≥1像素，眼眉没有参与拟合，逐帧变化尚不能代表静态模型。此结论未被本次合成测试改变。</p><figure><figcaption>合成场景：上排已知目标，下排4步训练后原生回读；4列为不同相机。48×40原生小图，非人物结果。</figcaption><img style="width:768px;max-width:100%;image-rendering:pixelated" src="R7原生训练入口/synthetic-native-comparison.png"></figure><p>'+html.escape(next_text)+'</p></section>'
log=L/'index.html';s=log.read_text(encoding='utf-8');assert 'id="native-pipeline-r7"' not in s;pos=s.index('</style>')+len('</style>');log.write_text(s[:pos]+section+s[pos:],encoding='utf-8')
article='<!-- R7_NATIVE_PIPELINE_BEGIN --><article id="robust122-r7"><h3>原生训练入口检查完成；SMPL-X静态对齐仍未通过</h3><p>'+html.escape(summary)+'</p><p>'+html.escape(next_text)+'</p><p>同一皮肤点留出仍32/35≥1像素；眼眉未约束。新相机和新人物训练均未准入。</p></article><!-- R7_NATIVE_PIPELINE_END -->'
for folder,page in [(T/'viewer','body-smplx.html'),(Path('D:/00/工作/查看器'),'progress.html')]:
 body=folder/page;s=body.read_text(encoding='utf-8');assert '<!-- R7_NATIVE_PIPELINE_BEGIN -->' not in s
 marker='<article id="face-observation">';assert marker in s;s=s.replace(marker,article+marker,1);body.write_text(s,encoding='utf-8')
 latest=folder/'latest-result.json';d=read(latest);before={k:v for k,v in d.items() if k!='latestDiagnostic'}
 d['latestDiagnostic']=dict(round='robust122-r7',title='SMPL-X：静态对齐未通过，训练入口检查完成',summary=summary,reportUrl=page+'#robust122-r7',updated=stamp,quality_accepted=False,training_started=False)
 save(latest,d);assert {k:v for k,v in read(latest).items() if k!='latestDiagnostic'}==before
save(O/'publication.json',dict(updated=stamp,report='http://127.0.0.1:8870/body-smplx.html#robust122-r7',daily_log=str(log),historical_model_preserved=True,
    browser_DOM_verified=False,actual_browser_pixel_review=False,PPT_updated=False,physical_quality_accepted=False))
print(json.dumps(dict(report='http://127.0.0.1:8870/body-smplx.html#robust122-r7',checks=36,person_training=False),ensure_ascii=False))
