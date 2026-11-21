from pathlib import Path
from datetime import datetime
import json,hashlib,shutil
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee');R=B/'current/person-static-native-align-fresh-113';V=B.parent/'viewer';W=Path(__file__).parent
read=lambda p:json.loads(p.read_text(encoding='utf8-sig'))
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8');t.replace(p)
receipt=read(W/'human113-deck-receipt.json');validation=read(W/'human113-validation.json')
assert receipt['changed_slides']==[52] and validation['packageIntegrity']['status']=='pass' and validation['presentationLayout']['finding_count']==0
canonical=Path(receipt['source']);final=Path(receipt['finalPath']);assert sha(canonical)==receipt['source_sha256'] and sha(final)==receipt['final_sha256']
shutil.copyfile(final,canonical);assert sha(canonical)==receipt['final_sha256']
receipt['canonical_replaced']=True;receipt['visually_reviewed']=True;save(W/'human113-deck-receipt.json',receipt)
shutil.copyfile(W/'human113-final-52.png',R/'slide52.png');shutil.copyfile(W/'human113-validation.json',R/'presentation-validation.json')
save(R/'presentation-review.json',dict(changed_slides=[52],first51_render_hashes_identical=True,visually_reviewed=True,canonical_sha256=sha(canonical),image_step=1800,quality_accepted=False,native_powerpoint_opened=False))
page=V/'body-smplx.html';html=page.read_text(encoding='utf8')
old='document.getElementById(\'native113-caption\').textContent=`本轮实际原生渲染，第 ${Number(p.review_step).toLocaleString()} 步：每对左侧为实拍，右侧为同相机渲染。属于阶段结果。`;'
new='document.getElementById(\'native113-caption\').textContent=`本轮实际原生渲染，第 ${Number(p.review_step).toLocaleString()} 步：每对左侧为实拍，右侧为同相机渲染。${[6000,9000].includes(p.review_step)?\'此快照恰在训练计划重置不透明度后保存，因此暂时变淡；不用于最终颜色评价。\':\'属于阶段结果，质量尚未验收。\'}`;'
assert old in html;html=html.replace(old,new,1);tmp=page.with_suffix('.html.tmp');tmp.write_text(html,encoding='utf8');tmp.replace(page)
save(R/'opacity-reset-review.json',dict(verified_step=6000,configured_reset_interval=3000,densify_until=15000,exported_opacity_max=.009999998845160007,exported_opacity_mean=.009843740612268448,source='actualsavedPLY+trainingconfig+source reviewafterreset order',conclusion='Translucent6000snapshot was saved after scheduledopacityreset; notfinalquality.',model_modified_by_check=False))
for filename in ['publication.json','handoff.json']:
    d=read(R/filename);d.update(updated=datetime.now().astimezone().isoformat(),presentation_pending=False,ui_verified=True,actual_published_gaussian='098',quality_accepted=False,goal_complete=False)
    if filename=='handoff.json':
        d.update(training_exec_session=24914,progress_sync_exec_session=8295,postprocess_wait_exec_session=37512,
            postprocess_script=str(W/'finish_native_person_113.sh'),current_progress=read(R/'gaussian-run/progress.json'),
            canonical_presentation=str(canonical),canonical_presentation_sha256=sha(canonical),
            next='Poll existing113training, notnewrun. Waitingpostprocess37512 automatically evaluates same-camera175views+heldtargetqueries, exportsstandard/+Yfullcovariance/allSH andcontinuousnovelview onlyafter18k. Then inspectimages/videos, actualOpenFiletest andstageviewer; do notclaimqualitypassed automatically. Userhistoricaltab15mustremainuntouched.',
            stage_picture_caveat='6000/9000 snapshots immediatelyfollowopacityreset and aretranslucent; verified6000PLYmaxopacity=.01. Reportcaptionexplains; final18knotatreset.')
    save(R/filename,d)
print(json.dumps(dict(canonical_sha256=sha(canonical),changed_slides=[52],training_running=read(R/'gaussian-run/progress.json')['running']),ensure_ascii=False))
