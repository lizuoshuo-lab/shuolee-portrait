from pathlib import Path
from zipfile import ZipFile
from datetime import datetime,timezone
import hashlib,json,shutil,xml.etree.ElementTree as ET
H=Path(__file__).parent;T=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab\outputs\mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';O=R/'video-update-20261027/background-geometry/material-region-comparison';C=H.parent/'shuolee-coordination'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
stamp=datetime.now(timezone.utc).isoformat();receipt=read(H/'region122v2-deck-receipt.json')
source,final=Path(receipt['source']),Path(receipt['finalPath']);canonical=T/'shuolee/reports/02_有限视角重建优化路线.pptx'
assert not receipt['canonical_replaced']
assert sha(source)==sha(canonical)==receipt['source_sha256'] and sha(final)==receipt['final_sha256']
with ZipFile(source) as a,ZipFile(final) as b:
    changed=[n for n in a.namelist() if a.read(n)!=b.read(n)]
    assert set(changed)=={'ppt/slides/slide52.xml','ppt/notesSlides/notesSlide52.xml','ppt/media/n1_material122.png'}
    xml=ET.fromstring(b.read('ppt/slides/slide52.xml'));ns={'p':'http://schemas.openxmlformats.org/presentationml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
    pics=xml.findall('.//p:pic',ns);assert len(pics)==1
    extent=pics[0].find('p:spPr/a:xfrm/a:ext',ns);assert abs(int(extent.attrib['cx'])/int(extent.attrib['cy'])-2)<1e-6
assert len(receipt['renders'])==52 and all(not a['changed'] for a in receipt['renders'][:51])
v=read(H/'region122v2-final-validation.json');assert v['finalSha256']==receipt['final_sha256']
assert v['packageIntegrity']['status']=='pass' and v['presentationLayout']['finding_count']==0 and v['firstPartyImport']['passed']
shots=['report-ui.png','report-images-ui.png','report-comparison-ui.png','viewer-ui.png'];assert all((O/f).exists() for f in shots)
for p in [canonical,source]:
    temp=p.with_suffix('.replacement.pptx');assert not temp.exists();shutil.copy2(final,temp);assert sha(temp)==receipt['final_sha256'];temp.replace(p)
receipt.update(accepted=stamp,canonical_replaced=True,canonical=str(canonical),delivery=str(source),
    changed_zip_members=changed,first51_members_and_render_hashes_identical=True,changed_slide_visually_reviewed=True,
    evidence_image_aspect_ratio=2,PowerPoint_or_WPS_runtime_verified=False,
    local_build_repairs='Provided required bundled-runtime module variable to finalizer. Firstrender stretched2:1evidenceinto3:1frame; discardedcandidate, fixedactualimageframe ratio, rendered/verifiednewcandidate. Researchnumbersunchanged.')
save(H/'region122v2-deck-receipt.json',receipt);save(O/'ppt-update.json',{k:x for k,x in receipt.items() if k!='renders'})
ui=dict(updated=stamp,actual_report_UI_reviewed=True,actual_mainviewer_UI_reviewed=True,report_tab='40',viewer_tab='31',
    report_url='http://127.0.0.1:8870/body-smplx.html#robust122-n8',
    verified=['NewN7/N8table,limitations,semanticcorrection andnativecrops displayed','Mainviewer latestDiagnostic updated, loaded119sofa204649points/89cameras retained'],
    screenshots=[dict(file=f,sha256=sha(O/f)) for f in shots],historical_tab15_untouched=True,
    training_started=False,new_PLY=False,quality_accepted=False,remote00_verified=False,WPS_cloud_verified=False)
save(O/'viewer-ui-review.json',ui)
p=read(O/'publication.json');p.update(updated=stamp,UI_verified=True,PPT_updated=True);save(O/'publication.json',p)
p=read(R/'publication.json');p.update(updated=stamp,N7_N8_UI_verified=True,N7_N8_PPT_updated=True,latest_region_review='video-update-20261027/background-geometry/material-region-comparison/viewer-ui-review.json');save(R/'publication.json',p)
q=read(R/'queue-state.json');q.update(updated=stamp,phase='residual_region_diagnosis_reviewed_and_published',running=None)
q['last_goal_turn'].update(completed='N7/N8 numerical andactualnativecropreview plusreportUI/samePPT52publication; confirmedfootROIsemanticsproblem andface/clothingcommonoffset',remaining='Review existing off-plane physicalscene support and actualfoot coverage before preregisteredcausalchange. Fullcamera/face<1px andtraining notadmitted.')
q['last_goal_turn']['evidence']+=['video-update-20261027/background-geometry/material-region-comparison/viewer-ui-review.json','video-update-20261027/background-geometry/material-region-comparison/ppt-update.json'];save(R/'queue-state.json',q)
e=read(R/'evidence-ledger.json');e['updated']=stamp;e['confirmed'].append('N7/N8 samecamera regiontable/nativecrops andsemanticcorrection reviewed inactualreporttab40 andmainviewer31. Same52-pagePPT onlylastpagechanged, first51bytes/renderhashes unchanged; finalimage2:1ratio explicitlyverified. No physicalqualitypass ornewtraining.');e['artifacts']+=q['last_goal_turn']['evidence'][-2:];save(R/'evidence-ledger.json',e)
s=read(C/'local-research-status.json');s.update(updated_at=stamp,source_queue_updated=stamp,N7_N8_UI_verified=True,same_PPT_lastpage_updated=True)
s['viewer']['current_122_ui_scope']='N7/N8 report andlatestDiagnostic actuallyverified,119modelretained. R1e syntheticnativeUI previouslyverified.'
s['renderer_readiness']['stage']=q['phase'];company=read(C/'company00-status.json')
s['company00_handoff_received']=dict(updated_at=company['updated_at'],source='company00-status.json',state=company['state'],scope='Read latest00engineeringstatus only. Root122didnotrunSMPLXserviceorchangeproductcode. Theirfailed focalestimate remainsseparate from122conditionalK research.')
save(C/'local-research-status.json',s)
print(json.dumps(dict(delivery=str(source),PPT_sha256=receipt['final_sha256'],UI_verified=True,first51_unchanged=True,training=False),ensure_ascii=False))
