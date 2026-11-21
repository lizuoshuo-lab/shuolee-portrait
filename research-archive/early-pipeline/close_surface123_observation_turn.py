"""Close this diagnostic phase with explicit browser visibility limits."""
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, shutil
H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=T/'shuolee/current/local-clothed-surface-123'
C=H.parent/'shuolee-coordination'
LOG=Path('D:/00/工作日志/2026-10-28/人物自由表面训练123')
V=T/'viewer';S=Path('D:/00/工作/查看器')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
stamp=datetime.now(timezone.utc).isoformat()
receipt=read(H/'surface123-observation-deck-receipt.json')
assert receipt['canonical_replaced'] and receipt['actual_last_page_visual_review']
assert sha(Path(receipt['canonical_path']))==receipt['final_sha256']
assert receipt['changed_slides']==[52] and all(not r['changed'] for r in receipt['renders'][:51])
model=V/'models/local-surface-123-whiteT-s18000/shuolee_123_whiteT_s18000.ply'
assert sha(model)=='df4b686f0e667f6cef08a4f24d6d1688d9b5efde74e8d53e06d20b2ab5702841'
for base in [V,S]:
    d=read(base/'latest-result.json');assert d['latestDiagnostic']['reportUrl'].endswith('#observation-quality')
assert not Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/shuolee-local-surface-123/grayT/gaussian-run').exists()
ui=dict(updated=stamp,browser='IAB5',report_tab='32',main_viewer_tab='25',
    report_url='http://127.0.0.1:8870/local-surface-123/index.html#observation-quality',
    report_and_main_diagnostic_DOM_verified=True,
    main_viewer_DOM_evidence=str(REC/'observation-quality-main-viewer-ax.txt'),
    model='local-surface-123-whiteT-s18000',points=309523,cameras=93,model_sha256=sha(model),
    actual_new_browser_screenshot=False,actual_new_Gaussian_or_points_visual_review=False,
    browser_limitation='Prior IAB2 binding expired. Existing tabs recovered in IAB5, no new tabs. Report screenshot twice failed with zero width, including after visibility set(true). open_in_codex returned queued. Summary click had one visible match but deadline exceeded. No viewport/security bypass attempted.',
    open_in_codex_status='queued',history_tab15_unchanged=True,company00_used=False,quality_accepted=False)
save(REC/'observation-quality-ui-review.json',ui)
p=read(REC/'publication.json');p.update(UI_checked=False,DOM_checked=True,
    UI_limitation=ui['browser_limitation'],latest_report=ui['report_url']);save(REC/'publication.json',p)
n=read(REC/'next-action.json');n.update(right_panel_tab='25',progress_tab='32',browser_id='5',
    actual_UI_verified=False,diagnostic_DOM_verified=True,latest_handoff=str(REC/'observation-quality-handoff.json'),
    publication_next='When this chat is visible, review report32 and latestviewer25, capture actualscreenshots, preserve historytab15. No screenshot success is currently claimed.')
save(REC/'next-action.json',n)
co=read(REC/'observation-quality-completion.json')
co.update(updated=stamp,state='diagnostics_published_visual_browser_verification_pending',
    actual_UI_verified=False,diagnostic_DOM_verified=True,PPT_updated=True,
    PPT_path=receipt['canonical_path'],PPT_sha256=receipt['final_sha256'],PPT_changed_slides=[52],
    no_new_training_or_geometry=True,quality_accepted=False)
save(REC/'observation-quality-completion.json',co)
handoff=dict(updated=stamp,conclusion=str(REC/'observation-quality-conclusion.json'),completion=co,
    next=n['next'],sources=str(H),browser=ui,
    do_not_repeat=['41frame visibility/quantization audit','M1 LK/localization;9bound failures already stopped before geometry',
        'shared/perframe head diagnostic','older material point/camera BA/training'],
    next_engineering_boundary='Read existing RTMW SimCC response/official continuous decoding. Register controlled known-peak and pixel-transform tests before any new real inference. Quantization and anatomical visibility remain separate variables.',
    active_goal_remains=True,training_allowed=False)
save(REC/'observation-quality-handoff.json',handoff)
c=read(C/'local-research-status.json');c.update(updated_at=stamp,current_local_observation_publication=co,
    latest_local_handoff=str(REC/'observation-quality-handoff.json'));save(C/'local-research-status.json',c)
for f in ['observation-quality-conclusion.json','observation-quality-completion.json','observation-quality-ui-review.json',
          'observation-quality-handoff.json','observation-quality-main-viewer-ax.txt','queue-state.json','next-action.json']:
    shutil.copy2(REC/f,LOG/f)
shutil.copy2(H/'surface123-observation-final-52.png',LOG/'报告末页_关键点精度与可见性.png')
print(json.dumps(dict(PPT_sha256=receipt['final_sha256'],model_sha256=sha(model),
    diagnostic_DOM_verified=True,actual_new_browser_screenshot=False,training_started=False,quality_accepted=False),ensure_ascii=False))
