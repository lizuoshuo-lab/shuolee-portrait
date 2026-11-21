from pathlib import Path
import json,hashlib,shutil,datetime,re
W=Path('C:/Users/Administrator/Documents/New project');T=W/'shuolee-slide-update'
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
O=B/'current/shared-point-ray-consistency-099';V=B.parent/'viewer'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(p)
r=read(T/'human099-deck-receipt.json');validation=read(T/'human099-validation.json')
source=Path(r['source']);final=Path(r['finalPath'])
assert sha(source)==r['source_sha256'] and sha(final)==r['final_sha256']==validation['finalSha256']
assert all(not v['changed'] for v in r['renders'][:51]) and len(r['renders'])==52
assert validation['packageIntegrity']['status']=='pass' and validation['presentationLayout']['findingCount']==0
shutil.copy2(final,source);assert sha(source)==r['final_sha256']
now=datetime.datetime.now().astimezone().isoformat();r.update(canonical_replaced=True,canonical_sha256=sha(source),updated=now)
save(T/'human099-deck-receipt.json',r)
shutil.copy2(T/'human099-final-52.png',O/'presentation-slide52.png')
shutil.copy2(T/'human099-validation.json',O/'presentation-validation.json')
save(O/'presentation-review.json',dict(path=str(source),sha256=sha(source),slides=52,changed_slides=[52],
    first51_render_hashes_unchanged=True,changed_slide_visually_reviewed=True,native_powerpoint_verified=False,pc00_deployed=False))
state=read(B/'work-status.json');state['latest_report']=dict(path=str(source),sha256=sha(source),changed_slides=[52],slides=52,updated=now,remote_deployed=False);save(B/'work-status.json',state)
# Keep the previous training result available under its original anchors,
# while opening the report on the current compact geometric finding.
page=V/'body-smplx.html';html=page.read_text(encoding='utf-8')
match=re.search(r'(<section id="shared-world-points">.*?</section>)',html,re.S);assert match
html=html[:match.start()]+'<details id="shared098-history"><summary>098共享点到平面约束与高斯结果（历史）</summary>'+match[1]+'</details>'+html[match.end():]
page.write_text(html,encoding='utf-8')
pub=read(O/'publication.json');pub.update(report_content_verified_in_browser=True,latest_viewer_status_and_report_verified=True,
    report_screenshot=str(O/'published-report.png'),presentation=str(source),presentation_sha256=sha(source));save(O/'publication.json',pub)
print(json.dumps(dict(canonical=str(source),sha256=sha(source),new_gaussian=False,quality_accepted=False)))
