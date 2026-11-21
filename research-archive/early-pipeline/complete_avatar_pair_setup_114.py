from pathlib import Path
import json,hashlib,datetime
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee');P=B/'current/avatar-repository-pair-114';V=B.parent/'viewer';G=B.parents[2]/'repos/GaussianAvatars_SMPLX-shuolee'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');q.replace(p)
protocol=read(P/'protocol.json');protocol.update(state='gaussianavatar_training_smplx_queued',updated=datetime.datetime.now().astimezone().isoformat(),followup_automation_id='automation')
protocol['method_differences'].append('Initial point count is identical, not spatial samples: GaussianAvatar uses official UV sampling with 200000 of 200681 valid texels at resolution524; SMPLX covers every mesh triangle with area-weighted samples.')
protocol['method_differences'].append('Both use the same interior-only L1/SSIM plus soft alpha; native GaussianAvatar LPIPS is disabled for this controlled local adaptation. Model-specific offset/scale/feature regularizers remain. Coarse shared body is frozen during Gaussian optimization.')
protocol['method_differences'].append('GaussianAvatar static canonical queries include the fresh pose-corrective base so initial UV/LBS positions exactly match the newly fitted mesh; official network and relative LBS residual transforms are retained.')
save(P/'protocol.json',protocol)
source=read(P/'source-audit.json');source.update(training_started=True,final_trainer_sha256=sha(G/'avatar_pair_114.py'),final_static_fitter_sha256=sha(G/'fit_avatar_pair_static_114.py'));save(P/'source-audit.json',source)
save(V/'body-avatar114/smplx/progress.json',dict(state='queued_after_gaussianavatar',arm='smplx',step=0,target_steps=18000,quality_accepted=False))
page=V/'body-smplx.html';s=page.read_text(encoding='utf-8')
old="p.state==='preflight'?'预检中':'待同相机验收'";new="p.state==='preflight'?'预检中':p.state==='queued_after_gaussianavatar'?'预检通过，排队训练':'待同相机验收'";assert old in s;s=s.replace(old,new)
hook='<p>完成后用同相机、同裁剪检查鼻嘴眼镜、肩袖衣摆、腿脚及连续新视角；'
assert hook in s
insert='''<details id="avatar114-direct"><summary>两条路线的直接对照</summary><p id="avatar114-comparison-state">等待两组完成；这里将展示相同相机、相同裁剪的输入与两组渲染。</p><div id="avatar114-comparison-media"></div></details>
<script>(()=>{async function check(){try{const r=await fetch('body-avatar114/comparison/result.json',{cache:'no-store'});if(!r.ok)return;const p=await r.json();document.getElementById('avatar114-comparison-state').textContent='两条路线的18,000步渲染已生成；重影和新视角质量需实际审阅。';const box=document.getElementById('avatar114-comparison-media');if(!box.children.length)box.innerHTML='<figure><img src="body-avatar114/comparison/face-000100.jpg" alt="同一相机的原照片、GaussianAvatar与SMPLX面部对照"><figcaption>左输入，中GaussianAvatar，右GaussianAvatars_SMPLX。保持图像比例。</figcaption></figure><figure><img src="body-avatar114/comparison/body-000100.jpg" alt="两条路线的同相机全身对照"><figcaption>全身同相机比较。</figcaption></figure><video controls preload="metadata" src="body-avatar114/comparison/continuous-ab.mp4"></video>';clearInterval(timer);}catch(e){}}const timer=setInterval(check,30000);check();})();</script>'''
s=s.replace(hook,insert+hook,1);page.write_text(s,encoding='utf-8')
queue=read(P/'queue-state.json');save(P/'handoff.json',dict(updated=datetime.datetime.now().astimezone().isoformat(),training_exec_session=12188,training_pid=334,queue_exec_session=41190,queue_pid=queue['pid'],automation_id='automation',expected_source_sha256=source['final_trainer_sha256'],state=protocol['state'],local_viewer_report='http://127.0.0.1:8870/body-smplx.html#avatar114-pair',old113_report_updated=True,canonical52slide_PPT_updated_only_slide52=True,next='Do not modify running avatar_pair_114.py or duplicate either arm. Supervisor performs export/evaluation and second fresh arm. Inspect failures before retry; finish per-arm actual OpenFile UI, paired visual comparison and promotion.',quality_accepted=False))
print('Pair protocol, queued state and handoff synchronized')
