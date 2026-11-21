from pathlib import Path
import hashlib,json,shutil,zipfile,datetime as dt,urllib.request
W=Path('C:/Users/Administrator/Documents/New project'); D=W/'shuolee-slide-update'
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee'); O=B/'current/portrait-alignment-open-source-116'; V=B.parent/'viewer'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
now=dt.datetime.now(dt.timezone.utc).isoformat();r=read(D/'human116-p3-deck-receipt.json');src=Path(r['source']);final=Path(r['exact_final_path'])
assert sha(src)==r['source_sha256'] and sha(final)==r['exact_final_sha256']
allowed={'ppt/slides/slide52.xml','ppt/slides/_rels/slide52.xml.rels','ppt/notesSlides/notesSlide52.xml','ppt/notesSlides/_rels/notesSlide52.xml.rels','ppt/media/image5.jpeg'}
with zipfile.ZipFile(src) as a,zipfile.ZipFile(final) as b:
 assert set(a.namelist())==set(b.namelist())
 assert set(n for n in a.namelist() if a.read(n)!=b.read(n))==allowed
 assert b.testzip() is None
checks=[]
for name in ['hsfm-decision.json','hsfm-joint-result.json','hsfm-joint-review.json','hsfm-raw000075-face-held-projections.jpg','hsfm-raw000125-whole-held-projections.jpg','hsfm-background-held-review-3.jpg']:
 with urllib.request.urlopen('http://127.0.0.1:8870/body-camera116/'+name,timeout=10) as response:
  data=response.read();assert data==(V/'body-camera116'/name).read_bytes();checks.append(dict(file=name,http_status=response.status,bytes=len(data)))
tmp=src.with_suffix('.pptx.tmp');shutil.copy2(final,tmp);tmp.replace(src);assert sha(src)==r['exact_final_sha256']
r.update(canonical_replaced=True,canonical_sha256=sha(src),updated=now,actual_last_slide_visual_review=True,first51_members_byte_identical=True)
save(D/'human116-p3-deck-receipt.json',r);save(O/'ppt-update-116-final.json',dict(updated=now,canonical=str(src),sha256=sha(src),changed_slides=[52],other51_package_and_render_unchanged=True,source_sha256=r['source_sha256'],render=str(D/'human116-p3-exact-52.png'),quality_accepted=False))
publication=read(O/'viewer-publication-p3-final.json');latest=read(V/'latest-result.json')
assert {k:v for k,v in latest.items() if k!='latestDiagnostic'}==publication['latest_model_unchanged']
ui=dict(updated=now,state='diagnostic_ui_verified',page='http://127.0.0.1:8870/body-smplx.html#alignment116-hsfm',browser='ExistingIABtab2',
 inspected=['NewP3status and static model boundary','Samecamera face075 comparison','Body125 lower section and ceiling063 failure evidence','Updated rootcause uncertainty'],
 screenshots=[dict(path=str(O/f),sha256=sha(O/f)) for f in ['viewer-p3-final.png','viewer-p3-background.png']],
 historical_tab15_preserved=True,existing_model_unchanged=True,new_model_open_file_not_applicable=True,quality_accepted=False,deployed00=False,http_checks=checks)
save(O/'viewer-ui-review-p3-final.json',ui);publication.update(updated=now,state='published_actual_ui_reviewed',ui_review='viewer-ui-review-p3-final.json',ppt_update_pending=False,ppt_record='ppt-update-116-final.json');save(O/'viewer-publication-p3-final.json',publication)
print(json.dumps(dict(canonical_ppt=str(src),only_slide52_changed=True,actual_browser_checked=True,model_unchanged=True)))
