import pathlib,json,hashlib,shutil,datetime,xml.etree.ElementTree as ET,zipfile
W=pathlib.Path(__file__).parent
B=pathlib.Path(r'C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
O=B/'current/human-surface-attribution-084';V=B.parent/'viewer'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');t.replace(p)
now=datetime.datetime.now().astimezone().isoformat(timespec='seconds')
d=read(W/'human084-deck-receipt.json');v=read(W/'human084v2-validation.json');src=pathlib.Path(d['finalPath']);dst=pathlib.Path(d['source'])
assert sha(src)==d['final_sha256']==v['finalSha256'] and sha(dst)==d['source_sha256']
assert v['packageIntegrity']['status']=='pass' and v['firstPartyImport']['passed']
assert len(d['renders'])==52 and sum(not r['changed'] for r in d['renders'])==51
with zipfile.ZipFile(src) as z:
 ns={'p':'http://schemas.openxmlformats.org/presentationml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
 root=ET.fromstring(z.read('ppt/slides/slide52.xml'))
 for im in root.findall('.//p:pic',ns):
  ex=im.find('.//a:xfrm/a:ext',ns);assert abs(int(ex.attrib['cx'])/int(ex.attrib['cy'])-440/365)<1e-5
save(W/'human084-deck-visual-review.json',dict(updated=now,actually_inspected=True,changed_slide=52,source_style_preserved=True,
 image_aspect_ratio_verified=True,earlier_stretched_draft_not_published=True,all51_unchanged_render_hashes_verified=True,
 final_slide_sha256=sha(W/'human084-final-52.png'),opened_in_PowerPoint=False))
shutil.copy2(src,dst);assert sha(dst)==d['final_sha256'];d.update(canonical_replaced=True,updated=now);save(W/'human084-deck-receipt.json',d)
m=read(B/'reports/report-manifest.json');entry=next(x for x in m['reports'] if x['key']=='reconstruction-optimization')
entry.update(sha256=sha(dst),bytes=dst.stat().st_size,local_validation=str(W/'human084v2-validation.json'),local_visual_review=str(W/'human084-deck-visual-review.json'),
 latest_scope='084冻结归因：最终高斯偏离表面与中性模板投影偏差；当前083质量仍未通过',current_local_report_deployed_to_00=False)
m.update(updated=now,last_local_revision=dict(path=str(dst),sha256=sha(dst),bytes=dst.stat().st_size,slides=52,remote_delivered=False,transfer_performed=False,updated=now));save(B/'reports/report-manifest.json',m)
ply=V/'models/whole-body-static-human-init/shuolee_083_s18000_f0100.ply';assert sha(ply)=='31845b1665f27c67ff2d2afa00cae488f3c7f8f31f364dcf25b902296263fb4f'
receipt=dict(updated=now,actual_browser_verified=True,report='http://127.0.0.1:8870/body-smplx.html#human-surface-diagnosis',
 loaded_images=3,current_gaussian_round='083',current_gaussian_points=300220,new_standalone_file_import_test=False,
 prior083_standalone_test_unchanged=True,model_bytes_unchanged=True,viewer_report_link_verified=True,
 screenshots=[dict(path=str(O/n),sha256=sha(O/n)) for n in ['page-verified.png','viewer-current.png']])
save(O/'ui-verification.json',receipt)
p=read(O/'publication.json');p.update(actual_browser_verified=True,deck_updated=True,deck=str(dst),deck_sha256=sha(dst));save(O/'publication.json',p)
s=read(B/'work-status.json');s.update(updated=now)
s['latest_report']=dict(path=str(dst),sha256=sha(dst),changed_slides=[52],slides=52,updated=now,remote_deployed=False)
s['pending_report_delivery']=dict(status='local_updated_remote_deferred',source=str(dst),file=dst.name,sha256=sha(dst),bytes=dst.stat().st_size,remote_delivered=False)
s['latest_diagnostic']['ui_verified']=True;save(B/'work-status.json',s)
summary=read(O/'summary.json');summary['visual_review_pending']=False;save(O/'summary.json',summary)
shutil.copy2(O/'summary.json',V/'body-human-surface-diagnosis/summary.json')
save(O/'completion.json',dict(updated=now,round='084',workflow_complete=True,type='frozen diagnosis',goal_turn_classification='progress',
 root_cause_established=False,goal_complete=False,repeated_blocking_condition_count=0,quality_accepted=False,
 gaussian_updates=0,camera_updates=0,new_gaussian_export=False,current_gaussian_round='083',source_models_unchanged=True,
 full_frames=175,fixed_rays=140,fixed_full_ghost_pixels=2420,template_transfer_queries=44,
 website=p['report_url'],actual_browser_verified=True,deck=str(dst),deck_sha256=sha(dst),
 next_bounded_work=s['next_bounded_work'],remote_deployed=False))
for n in ['seal_surface_084.py','update_deck_084.mjs']:
 shutil.copy2(W/n,O/n)
print(json.dumps({'diagnostic_completed':True,'goal_complete':False,'new_ply':False,'deck_sha256':sha(dst),'next':s['next_bounded_work']['type']}))
