"""Commit display/deck verification only. Human reconstruction remains unaccepted."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
H=Path(__file__).parent;T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';O=R/'residual-edge-association';A=R/'residual-edge-identity';C=H.parent/'shuolee-coordination'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (O/'viewer-ui-review.json').exists()
prefix='edgeAssociation-122';receipt=H/(prefix+'-deck-receipt.json');r=read(receipt);source=Path(r['source']);final=Path(r['finalPath'])
assert sha(source)==r['source_sha256'] and sha(final)==r['final_sha256']
assert r['changed_slides']==[52] and len(r['renders'])==52
assert read(H/(prefix+'-exact-package.json'))['other_members_byte_identical']
assert read(H/(prefix+'-final-validation.json'))['finalSha256']==r['final_sha256']
for n in ['report-ui.png','report-lower-ui.png','viewer-ui.png']:assert (O/n).stat().st_size>10000
stamp=datetime.now(timezone.utc).isoformat()
shutil.copy2(final,source);mirror=T/'shuolee/reports/02_有限视角重建优化路线.pptx';shutil.copy2(final,mirror)
assert sha(source)==sha(mirror)==r['final_sha256']
r.update(canonical_replaced=True,updated=stamp,actual_slide52_reviewed=True,first51_render_and_package_unchanged=True);save(receipt,r)
save(O/'ppt-update.json',dict(updated=stamp,path=str(source),sha256=sha(source),slides=52,changed=[52],first51_unchanged=True,actual_slide52_reviewed=True,receipt=str(receipt)))
save(O/'viewer-ui-review.json',dict(updated=stamp,actual_report_tab='40',actual_viewer_tab='31',report_url='http://127.0.0.1:8870/body-smplx.html#robust122-d3',
 verified=['D2/D3 report actually reloaded and reviewed, raw nativecrop and110/170candidate table visible, no accuracy improvement claim.',
 'Mainviewer actually retains119sofa/89cameras and D2/D3 diagnostic, no fake newmodel.',
 'Slide52 actually inspected, native image aspect preserved; other51slides package/render unchanged; tab15 notoperated.'],
 screenshots={n:sha(O/n) for n in ['report-ui.png','report-lower-ui.png','viewer-ui.png']},
 new_human_training=False,new_human_pointcloud=False,compensation_applied=False,physical_calibration_accepted=False,company00_tested=False))
p=read(O/'publication.json');p.update(updated=stamp,report_UI_verified=True,UI_evidence='viewer-ui-review.json',PPT_pending=False,PPT='ppt-update.json');save(O/'publication.json',p)
q=read(R/'queue-state.json');q.update(updated=stamp,phase='D2_D3_association_audited_and_reviewed',running=None)
q['last_goal_turn'].update(publication_complete=True,completed='D2 actualreview/analyticbound established D1measurement contamination. D3uniqueassociation tested and reviewed, incomplete and not promoted. Report/119UI/slide52 updated andactuallyreviewed. Goalunachieved.');save(R/'queue-state.json',q)
c=read(C/'local-research-status.json');c.update(updated_at=stamp,source_queue_updated=stamp,latest_residual_edge_association_PPT=str(O/'ppt-update.json'),local_GPU_task_active=False)
c['viewer'].update(current_122_ui_verified=True,current_122_ui_scope='D2/D3 report and retained119mainpreview; no newcalibration or personmodel.');save(C/'local-research-status.json',c)
save(C/'local-residual-edge-handoff.json',dict(updated=stamp,stage='D2/D3',evidence=[str(A/'review.json'),str(O/'review.json')],
 result='D1native strongestgradient samples jumpnearbyborders/crossjoints; wholebox analytic bound supports inability of that smoothfield tostraighten those samples. D1negative cannotruleout real lensresidual. D3association-only exactprofiles trial gives170partial proposals/5ambiguous anchors; canreject oneblurrededge asmultiplepeaks. Not accepted productionextractor ornewcompensation.',
 company_action='Informationonly. CPU-only localwork. No 00package,113source,renderercontractortrainingchange. Noackloop.',
 limits='No newhumantraining, oldK/119preserved, original13faceheldoutmax13.04875764px unchanged.',next=q['next']))
print(json.dumps(dict(publication_complete=True,goal_active=True,compensation_applied=False,first51_unchanged=True,ppt=str(source)),ensure_ascii=False))
