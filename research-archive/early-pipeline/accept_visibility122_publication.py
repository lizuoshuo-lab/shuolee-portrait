from pathlib import Path
from zipfile import ZipFile
from datetime import datetime,timezone
import hashlib,json,shutil
H=Path(__file__).parent;T=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab\outputs\mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';I=R/'video-update-20261027';M=I/'material-visibility';N=I/'native-temporal-material';C=H.parent/'shuolee-coordination'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
stamp=datetime.now(timezone.utc).isoformat();receipt=read(H/'visibility122-deck-receipt.json')
source,final=Path(receipt['source']),Path(receipt['finalPath']);canonical=T/'shuolee/reports/02_有限视角重建优化路线.pptx'
assert not receipt['canonical_replaced']
assert sha(source)==sha(canonical)==receipt['source_sha256'] and sha(final)==receipt['final_sha256']
with ZipFile(source) as a,ZipFile(final) as b:
    changed=[n for n in a.namelist() if a.read(n)!=b.read(n)];added=list(set(b.namelist())-set(a.namelist()))
    assert set(changed)=={'ppt/slides/slide52.xml','ppt/notesSlides/notesSlide52.xml','ppt/slides/_rels/slide52.xml.rels'}
    assert added==['ppt/media/n1_material122.png'] and not(set(a.namelist())-set(b.namelist()))
assert len(receipt['renders'])==52 and all(not s['changed'] for s in receipt['renders'][:51])
validation=read(H/'visibility122-final-validation.json')
assert validation['finalSha256']==receipt['final_sha256'] and validation['packageIntegrity']['status']=='pass' and validation['presentationLayout']['finding_count']==0
assert all((M/n).is_file() for n in ['report-ui.png','viewer-ui.png'])
for destination in [canonical,source]:
    temp=destination.with_suffix('.replacement.pptx');assert not temp.exists()
    shutil.copy2(final,temp);assert sha(temp)==receipt['final_sha256'];temp.replace(destination)
receipt.update(accepted=stamp,canonical_replaced=True,canonical=str(canonical),delivery=str(source),changed_zip_members=changed,
    added_zip_members=added,first51_members_and_render_hashes_identical=True,changed_slide_visually_reviewed=True,
    PowerPoint_or_WPS_runtime_verified=False,
    visual_review='Before/after PNG inspected. New six-column two-row native occlusion crop image preserved aspect ratio and contains no text. Caption and conclusions fit. Other51 pages byte/render identical.')
save(H/'visibility122-deck-receipt.json',receipt);save(M/'ppt-update.json',{k:v for k,v in receipt.items() if k!='renders'})
ui=dict(updated=stamp,report_tab='40',viewer_tab='31',report_url='http://127.0.0.1:8870/body-smplx.html#robust122-n1',
    actual_report_reviewed=True,latestDiagnostic_display_verified=True,model='new-capture-119-sofa-clothed',cameras=89,
    viewer_version='2026.10.27.5',quality_accepted=False,historical_tab15_untouched=True,native_interactive_GUI_verified=False,
    screenshots=[dict(file=f,sha256=sha(M/f)) for f in ['report-ui.png','viewer-ui.png']],
    remote00_verified=False,WPS_cloud_verified=False,training_started=False,new_PLY=False)
save(M/'viewer-ui-review.json',ui)
p=read(R/'publication.json');p.update(updated=stamp,N1_N2_UI_verified=True,N1_N2_UI_review='video-update-20261027/material-visibility/viewer-ui-review.json',N1_N2_PPT_update='video-update-20261027/material-visibility/ppt-update.json');save(R/'publication.json',p)
q=read(R/'queue-state.json');q['updated']=stamp
q['last_goal_turn']['remaining']='Observation admission contract and updated outputK provenance; allframe independent physical subpixel camera admission and fresh static training'
q['last_goal_turn']['evidence']+=['video-update-20261027/material-visibility/viewer-ui-review.json','video-update-20261027/material-visibility/ppt-update.json']
save(R/'queue-state.json',q)
ledger=read(R/'evidence-ledger.json');ledger['updated']=stamp
ledger['confirmed'].append('N1/N2 evidence report actually refreshed/reviewed in tab40. Tab31 shows new diagnostic and retained119 model/89cameras. Same52pagePPT slide52 text,image,notes updated only;first51 members and rendered hashes unchanged. No new model/training/nativeGUI/00verification.')
ledger['artifacts']+=q['last_goal_turn']['evidence'][-2:];save(R/'evidence-ledger.json',ledger)
status=read(C/'local-research-status.json');status.update(updated_at=stamp,source_queue_updated=stamp)
status['latest_material_diagnosis']=dict(stage='N1_N2_completed',raw_video='32be30636adabfc4c25a1c8387cf19302d3ca39b4f040072081646d36523317c',
    native_frames=557,known_occlusion_counterexamples=6,RoMa_rejects_known_cases=6,full_camera_admission=False,
    report_UI_verified=True,same_PPT_last_page_updated=True,company00_deployed=False)
save(C/'local-research-status.json',status)
protocol=read(R/'protocol.json');protocol['next']='N3 once-only CPU observation-admission contract checks on preserved N1/N2 evidence; no new inference, calibration, observation filtering or training. Then separately document updated outputK provenance.';save(R/'protocol.json',protocol)
plan_path=I/'observation-admission-plan.json';assert not plan_path.exists()
save(plan_path,dict(stage='N3',registered=stamp,type='CPU evidence-contract engineering, not a new reconstruction experiment',
    hypothesis='Algorithmic confidence must not silently promote a physically contradicted or unknown observation into certified camera/face validation.',
    immutable_inputs={str(p):sha(p) for p in [N/'seeds.json',N/'observations.npz',N/'visual-review.json',M/'result.json',M/'visual-review.json']},
    contract_fields=['source_sha256','frame_id','native_pixel_center_convention','track_id','frozen_family_id','frozen_split',
        'raw_uv','tracker_numeric_validity','visibility_evidence','material_identity_evidence','measurement_uncertainty',
        'evidence_source','development_exposure','camera_fit_permission','independent_face_admission'],
    required_controls=['Keep all raw observations and immutable split/family identity',
        'All6 visually contradicted N1 examples fail physical admission even with tinyFB',
        'N2 confidence/FB passing alone never sets physical identity or calibrated uncertainty',
        'Unknown/occluded/material-contradicted cases cannot silently count as observed face success',
        'An empty accepted set cannot meet whole-face coverage or allframe camera admission',
        'Holdout observations cannot enter camera fitting; method-development exposure is separately declared',
        'Require independent observation prediction for each heldout target, not self-reprojection of a point fit with that target'],
    budget=dict(CPU_seconds=60,runs=1),new_inference=False,new_K_or_pose=False,new_training=False,
    stop='On implementation error preserve failed receipt and repair demonstrated contract defect only. No new data-dependent thresholds or optimization.'))
print(json.dumps(dict(PPT_published=str(source),N1_N2_UI_verified=True,N3_registered_not_run=True,training_started=False),ensure_ascii=False))
