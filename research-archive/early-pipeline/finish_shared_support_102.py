from pathlib import Path
import datetime,hashlib,json,shutil,ast
W=Path('C:/Users/Administrator/Documents/New project');BUILD=W/'shuolee-slide-update'
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
R=B/'current/person-static-shared-support-fresh-102'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
now=datetime.datetime.now().astimezone().isoformat();receipt=read(BUILD/'human102-deck-receipt.json');v=read(BUILD/'human102-validation.json')
source=Path(receipt['source']);final=Path(receipt['finalPath'])
assert sha(source)==receipt['source_sha256'] and sha(final)==receipt['final_sha256']==v['finalSha256']
assert receipt['changed_slides']==[52] and len(receipt['renders'])==52 and not any(r['changed'] for r in receipt['renders'][:51])
assert v['packageIntegrity']['status']=='pass' and v['presentationLayout']['finding_count']==0
assert (R/'published-report.png').stat().st_size>10000
assert not (R/'gaussian-run/current.pt').exists() and not (R/'gaussian-run/loss.jsonl').exists()
shutil.copy2(final,source);assert sha(source)==receipt['final_sha256']
receipt.update(canonical_replaced=True,canonical_sha256=sha(source),actual_slide52_visual_review=True,
    first51_render_hashes_unchanged=True,reviewed_at=now,native_powerpoint_tested=False)
save(BUILD/'human102-deck-receipt.json',receipt)
for src,name in [(BUILD/'human102-final-52.png','presentation-slide52.png'),(BUILD/'human102-validation.json','presentation-validation.json')]:shutil.copy2(src,R/name)
save(R/'presentation-review.json',receipt)
pub=read(R/'publication.json');pub.update(ui_verified=True,verified_at=now,report_tab=2,latest_viewer_tab=15,
    actual_latest_status='102 stopped beforefirstupdate;098310961points remains selected',
    screenshot=str(R/'published-report.png'),presentation=str(source),presentation_sha256=sha(source))
save(R/'publication.json',pub)
state=read(B/'work-status.json');state.update(updated=now,local_training_running=False)
state['latest_diagnostic'].update(report_url=pub['report'],ui_verified=True,presentation_updated=True)
save(B/'work-status.json',state)
handoff=read(R/'handoff.json');handoff.update(presentation_updated=True,ui_verified=True,
    numeric_preflight_passed=True,initial_support_preflight_passed=False,
    primary_open_issue='CommonXYZ contract holds, but43observations havezero initial raster contribution. 28are confidentmaskinterior,10belowhalfmask. Need image-specific semantic/visibility and camera attribution; do not quietly removegate.')
save(R/'handoff.json',handoff)
repo=B.parents[2]/'repos/PGSR-shuolee';snapshot=R/'source-diagnostic';snapshot.mkdir(exist_ok=True)
files=[repo/'audit_initial_feature_support_102.py',repo/'audit_shared_support_centers_102.py',BUILD/'publish_shared_support_102.py',BUILD/'update_deck_102.mjs',Path(__file__)]
manifest=[]
for src in files:
    if src.suffix=='.py':ast.parse(src.read_text(encoding='utf8'))
    dest=snapshot/src.name;assert not dest.exists();shutil.copy2(src,dest);manifest.append(dict(path=str(src),sha256=sha(src)))
save(snapshot/'manifest.json',manifest)
print(json.dumps(dict(canonical=str(source),sha256=sha(source),report=pub['report'],gaussian_updates=0,goal_complete=False)))
