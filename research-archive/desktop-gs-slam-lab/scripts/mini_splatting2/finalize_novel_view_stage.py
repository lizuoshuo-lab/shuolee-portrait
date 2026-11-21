"""Record observed stage results and refresh display metadata without promoting candidates."""
import datetime
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[2]/'outputs/mini-splatting2-mip-depth-001'
V=ROOT/'viewer'
for arm in ['control','candidate']:
    path=V/f'model-difix-distill-{arm}.json';meta=json.loads(path.read_text(encoding='utf-8'))
    meta.update(iterations=1200, refinement_steps=1200, pretrained_iterations=18000)
    if 'depth_guidance' in meta: meta['initialization_depth_guidance']=meta.pop('depth_guidance')
    meta['current_depth_guidance']=dict(coefficient=0,used=False)
    path.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
html=(V/'index.html').read_text(encoding='utf-8')
if 'id="current-model-status"' not in html:
    html=html.replace('</select><div class="actions">','</select><p id="current-model-status" class="muted"></p><div class="actions">',1)
    html=html.replace('各组的实验口径分别报告。','第四组比较训练分辨率。第五组从同一完整训练模型开始各微调1200步，对照真实照片与额外生成的俯仰视角；未增密、未添加新深度损失。各组的实验口径分别报告。',1)
    (V/'index.html').write_text(html,encoding='utf-8')
js=(V/'viewer.js').read_text(encoding='utf-8')
if "$('current-model-status').textContent" not in js:
    anchor="$('model-count').textContent=meta.gaussianCount.toLocaleString();"
    assert anchor in js
    js=js.replace(anchor,anchor+"$('current-model-status').textContent=meta.status||'实验模型';",1)
    js=js.replace("'MiniSplatting2_18000_'+Date.now()","'MiniSplatting2_'+variant+'_'+Date.now()")
    (V/'viewer.js').write_text(js,encoding='utf-8')

summary=json.loads((ROOT/'difix-distill-pilot-001/summary.json').read_text())
review=dict(status='initial_visual_review_complete',created=time.time(),reviewer='assistant',user_ratings=0,
    reviewed_stills=['control/paths/elevation-015.png','candidate/paths/elevation-015.png',
                     'candidate/paths/elevation-022.png','control/paths/elevation-037.png','candidate/paths/elevation-037.png',
                     'control/paths/distance-015.png','candidate/paths/distance-015.png'],
    observations=[
        'At supervised high-pitch pose15, facial ghosting remains in both3Darms; no large clarity gain visible.',
        'At unsupervised pose37, gross facial/clothing appearance remains similar; no decisive detail improvement.',
        'At unsupervised distance15, fine floating dark background texture persists in both3Darms.',
        'Generated2Dteacher frames can be clearer than3Doutput but sometimes change appearance; not sufficient to accept3Dreconstruction.'
    ],
    subjective_scoring='User A/B interface available; no fabricated numeric human ratings',
    observed_psnr=dict(control=summary['models']['control']['evaluation']['mean_psnr'],candidate=summary['models']['candidate']['evaluation']['mean_psnr']),
    conclusion='Exploratory pilot complete; clarity problem unresolved. Retain both inspectable models; do not promote default.',
    accepted=False,source_preserved=True)
(ROOT/'difix-distill-pilot-001/visual-review.json').write_text(json.dumps(review,indent=2),encoding='utf-8')
state=dict(updated=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='stage_complete_followup_prepared',
    user_direction='All videos horizontal; introduce diffusion for novel pitch and judge subjective clarity as well as observed-view error.',
    work_video='HOVER_VERSA_0102.mp4',
    latest_report='http://127.0.0.1:8820/novel-views.html',local_viewer_models=19,
    completed=['120 Difix novel-view frames','paired1200step3Drefinement','all120 path renders from each3Dmodel','human A/B form and JSON export'],
    evidence=['difix-novel-paths-001/summary.json','difix-novel-paths-001/continuity.json',
              'difix-distill-pilot-001/summary.json','difix-distill-pilot-001/visual-review.json'],
    subjective_ratings_not_fabricated=True,geometry_problem_solved=False,accepted=False,
    native_head_stage=dict(training_steps=18000,complete=True,accepted=False,report='native-head.html',
                           observation='Small observed-head PSNR gain; novel-pitch ghosting persists. No model promotion.'),
    next='Joint video diffusion on two fixed25framepitchclips; requires authorized official SVD weights.',
    weights=dict(source='https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1',http_status=401),
    remote_delivery=dict(current_new_stage_delivered=False,last_verified_port=8860,last_verified_model_count=17),
    qq=dict(recipient='小润',sent_local_time='2026-10-19 08:45',visible_sent_bubble_verified=True),
    original_model_unchanged=True,unrelated_asr_process_touched=False)
(ROOT/f'progress-checkpoint-novel-{int(time.time())}.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(status=state['status'],models=19,accepted=False)))
