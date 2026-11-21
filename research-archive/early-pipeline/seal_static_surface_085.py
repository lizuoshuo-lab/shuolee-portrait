import pathlib,json,hashlib,shutil,datetime
W=pathlib.Path(__file__).parent
B=pathlib.Path(r'C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
O=B/'current/observed-static-face-surface-085';V=B.parent/'viewer'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8');q.replace(p)
now=datetime.datetime.now().astimezone().isoformat(timespec='seconds')
d=read(W/'human085-deck-receipt.json');v=read(W/'human085-validation.json');src=pathlib.Path(d['finalPath']);dst=pathlib.Path(d['source'])
assert sha(src)==d['final_sha256']==v['finalSha256'] and sha(dst)==d['source_sha256']
assert v['packageIntegrity']['status']=='pass' and v['firstPartyImport']['passed']
assert len(d['renders'])==52 and sum(not r['changed'] for r in d['renders'])==51
save(W/'human085-deck-visual-review.json',dict(updated=now,actually_inspected=True,changed_slide=52,
    all51_unchanged_render_hashes_verified=True,no_new_images_or_layouts=True,opened_in_PowerPoint=False,
    final_slide_sha256=sha(W/'human085-final-52.png')))
shutil.copy2(src,dst);assert sha(dst)==d['final_sha256'];d.update(canonical_replaced=True,updated=now);save(W/'human085-deck-receipt.json',d)
m=read(B/'reports/report-manifest.json');entry=next(x for x in m['reports'] if x['key']=='reconstruction-optimization')
entry.update(sha256=sha(dst),bytes=dst.stat().st_size,local_validation=str(W/'human085-validation.json'),local_visual_review=str(W/'human085-deck-visual-review.json'),
    latest_scope='085静态表面预检：直接逐顶点拟合第4步触发局部网格有效性停止，未进入高斯训练。',current_local_report_deployed_to_00=False)
m.update(updated=now,last_local_revision=dict(path=str(dst),sha256=sha(dst),bytes=dst.stat().st_size,slides=52,remote_delivered=False,transfer_performed=False,updated=now));save(B/'reports/report-manifest.json',m)
meta=read(V/'model-whole-body-static-human-init.json');meta.update(report='body-smplx.html#static-surface-fit',latestDiagnostic='085');save(V/'model-whole-body-static-human-init.json',meta)
latest=read(V/'latest-result.json');latest.update(updated=now,report=meta['report'],latestDiagnostic='085');latest['revision']=hashlib.sha256(json.dumps(latest,sort_keys=True).encode()).hexdigest();save(V/'latest-result.json',latest)
p=read(O/'publication.json');p.update(deck_updated=True,deck=str(dst),deck_sha256=sha(dst));save(O/'publication.json',p)
s=read(B/'work-status.json');s.update(updated=now)
s['latest_report']=dict(path=str(dst),sha256=sha(dst),changed_slides=[52],slides=52,updated=now,remote_deployed=False)
s['pending_report_delivery']=dict(status='local_updated_remote_deferred',source=str(dst),file=dst.name,sha256=sha(dst),bytes=dst.stat().st_size,remote_delivered=False)
save(B/'work-status.json',s)
for n in ['seal_static_surface_085.py','publish_static_surface_085.py','update_deck_085.mjs']:shutil.copy2(W/n,O/n)
print(json.dumps(dict(deck_updated=True,deck_sha256=sha(dst),metadata_report=meta['report'],new_ply=False)))
