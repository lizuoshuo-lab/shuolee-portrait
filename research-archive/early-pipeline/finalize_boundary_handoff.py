from pathlib import Path
import json,hashlib,shutil,datetime,urllib.request
W=Path('C:/Users/Administrator/Documents/New project');D=W/'shuolee-slide-update'
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
r=read(D/'static-deck-receipt.json');v=read(D/'static-constraints-validation.json')
src=Path(r['source']);final=Path(r['finalPath'])
assert sha(src)==r['source_sha256'] and sha(final)==r['final_sha256']==v['finalSha256']
assert v['packageIntegrity']['status']=='pass' and v['presentationLayout']['findingCount']==0
assert v['packageIntegrity']['slide_count']==52
save(D/'static-visual-review.json',dict(inspected_original_numbers=[2,20,32,45,51,52,55],
    final_slide_numbers=[2,20,32,44,50,51,52],reviewed=True,
    all_other_retained_slide_contents_checked_before_footer_update=True,
    full_native_powerpoint_application_inspected=False,conclusion='未见文字、公式或图片裁切，方法与效果在前，不足集中在末页。'))
backup=D/'static-source-before.pptx'
if not backup.exists():shutil.copy2(src,backup)
assert sha(backup)==r['source_sha256']
shutil.copy2(final,src);assert sha(src)==r['final_sha256']
r['canonical_replaced']=True;r['canonical_sha256']=sha(src);save(D/'static-deck-receipt.json',r)
manifest=read(B/'reports/report-manifest.json');now=datetime.datetime.now().astimezone().isoformat(timespec='seconds')
manifest['updated']=now
for row in manifest['reports']:
    if row['key']=='reconstruction-optimization':
        row.update(slides=52,sha256=sha(src),bytes=src.stat().st_size,local_validation=str(D/'static-constraints-validation.json'),local_visual_review=str(D/'static-visual-review.json'),latest_scope='静态人物、边界机制诊断和单变量迭代；动态路线已废弃',current_local_report_deployed_to_00=False)
save(B/'reports/report-manifest.json',manifest)
index=read(B/'gaussian-deliveries.json')
for n in ['066','069']:
    index['rounds'][n].update(local_standalone_open_verified=True,local_gaussian_and_centers_verified=True,quality_accepted=False)
index['updated']=now;save(B/'gaussian-deliveries.json',index)
# Delete only disposable PPT candidates produced in this turn, after canonical verification.
removed=[]
for name in ['static-prior-validation.pptx','static-second-validation.pptx','static-constraints-candidate.pptx']:
    p=(D/name).resolve();assert p.is_relative_to(D.resolve()) and p.suffix=='.pptx'
    if p.exists():removed.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size));p.unlink()
save(D/'static-disposable-cleanup.json',dict(canonical=str(src),canonical_sha256=sha(src),removed=removed,original_source_backup=str(backup)))
checks=[]
for suffix in ['body-smplx.html','body-boundary-current/head-comparison.jpg','body-boundary-current/analysis.json','models/whole-body-static-boundary/shuolee_069_s18000_f0100.ply?download=1']:
    req=urllib.request.Request('http://127.0.0.1:8870/'+suffix,method='HEAD')
    with urllib.request.urlopen(req,timeout=10) as response: checks.append(dict(url=req.full_url,status=response.status,bytes=response.headers.get('Content-Length')))
save(D/'boundary-handoff.json',dict(updated=now,http_checks=checks,presentation=str(src),new_training_started=False,next='先核验边界高斯的真实合成贡献、跨视图遮挡与投影，再冻结单变量训练协议。'))
print(json.dumps(dict(ppt=str(src),slides=52,sha256=sha(src),web_checked=len(checks),training_started=False),ensure_ascii=False))
