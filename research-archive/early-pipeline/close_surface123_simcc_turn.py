"""Record completed controls and explicit remaining scientific/UI limitations."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=T/'shuolee/current/local-clothed-surface-123';C=H.parent/'shuolee-coordination'
LOG=Path('D:/00/工作日志/2026-10-28/人物自由表面训练123');V=T/'viewer';S=Path('D:/00/工作/查看器')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
stamp=datetime.now(timezone.utc).isoformat()
receipt=read(H/'surface123-simcc-deck-receipt.json')
assert receipt['canonical_replaced'] and receipt['actual_last_page_visual_review']
assert sha(Path(receipt['canonical_path']))==receipt['final_sha256']
assert receipt['changed_slides']==[52] and all(not x['changed'] for x in receipt['renders'][:51])
model=V/'models/local-surface-123-whiteT-s18000/shuolee_123_whiteT_s18000.ply'
assert sha(model)=='df4b686f0e667f6cef08a4f24d6d1688d9b5efde74e8d53e06d20b2ab5702841'
for b in [V,S]:assert read(b/'latest-result.json')['latestDiagnostic']['reportUrl'].endswith('#simcc-decoder')
ax=(REC/'simcc-decoder-main-viewer-ax.txt').read_text(encoding='utf-8')
assert 'SMPL-X：连续解码仍未解决对齐' in ax and '309,523' in ax and '93 个相机' in ax
assert not Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/shuolee-local-surface-123/grayT/gaussian-run').exists()
ui=dict(updated=stamp,browser='IAB5 existing binding',report_tab='32',main_viewer_tab='25',
    report_url='http://127.0.0.1:8870/local-surface-123/index.html#simcc-decoder',
    report_and_main_diagnostic_AX_verified=True,report_six_frame_gallery_expanded=True,
    model='local-surface-123-whiteT-s18000',points=309523,cameras=93,model_sha256=sha(model),
    actual_new_browser_screenshot=False,actual_new_Gaussian_or_points_visual_review=False,
    limitation='Existing report tab32 reloaded and actual AX click expanded six-frame gallery. Screenshot again returned Cannot take screenshot with 0 width. open_in_codex existing tab returned queued. Main tab25 reload completed and AX confirms new diagnostic and unchanged whiteT model. No new tabs or viewport/security workarounds.',
    open_in_codex_status='queued',history_tab15_unchanged=True,company00_used=False,quality_accepted=False)
save(REC/'simcc-decoder-ui-review.json',ui)
p=read(REC/'publication.json');p.update(UI_checked=False,DOM_checked=True,UI_limitation=ui['limitation'],latest_report=ui['report_url']);save(REC/'publication.json',p)
n=read(REC/'next-action.json');n.update(right_panel_tab='25',progress_tab='32',browser_id='5',
    actual_UI_verified=False,diagnostic_DOM_verified=True,latest_handoff=str(REC/'simcc-decoder-handoff.json'))
save(REC/'next-action.json',n)
completion=dict(updated=stamp,state='decoder_control_published_visual_browser_verification_pending',
    actual_UI_verified=False,diagnostic_AX_verified=True,actual_native_image_review=True,
    PPT_updated=True,PPT_path=receipt['canonical_path'],PPT_sha256=receipt['final_sha256'],PPT_changed_slides=[52],
    no_new_training_or_geometry=True,quality_accepted=False,goal_active=True,
    progress='Captured original SimCC responses once; exact replay and single-variable continuous decode measured. This narrows implementation causes but does not meet physical alignment acceptance.')
save(REC/'simcc-decoder-completion.json',completion)
handoff=dict(updated=stamp,conclusion=str(REC/'simcc-decoder-conclusion.json'),completion=completion,next=n['next'],browser=ui,
    do_not_repeat=['Completed12CPUinferences and6framecontinuousdecodercontrol;rawNPZretained',
        '41framevisibility/quantizationaudit','M1photometricmeasurementfailed9framesalready',
        'oldheadpose/material/BA/trainingdiagnostics'],
    caution='Previous SMPL-X audit already computed47.1%faceweight on self-occluded/backward model points. Do not repeat that audit as new evidence; current misaligned mesh is not anatomical truth.',
    active_goal_remains=True,training_allowed=False,company00_handoff_sent=False)
save(REC/'simcc-decoder-handoff.json',handoff)
c=read(C/'local-research-status.json');c.update(updated_at=stamp,current_local_decoder_publication=completion,
    latest_local_handoff=str(REC/'simcc-decoder-handoff.json'));save(C/'local-research-status.json',c)
for f in ['simcc-decoder-conclusion.json','simcc-decoder-completion.json','simcc-decoder-ui-review.json','simcc-decoder-handoff.json',
          'simcc-decoder-main-viewer-ax.txt','simcc-decoder-report-ax.txt','queue-state.json','next-action.json']:
    shutil.copy2(REC/f,LOG/f)
shutil.copy2(H/'surface123-simcc-final-52.png',LOG/'报告末页_连续解码核查.png')
print(json.dumps(completion,ensure_ascii=False))
