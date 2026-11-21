from pathlib import Path
import hashlib,json,datetime,shutil
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
O=B/'current/fullscene-person-constraint-audit-076';V=B.parent/'viewer'
W=Path('C:/Users/Administrator/Documents/New project/shuolee-slide-update')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
record=read(O/'publication.json');latest=read(V/'latest-result.json')
assert latest['round']=='075' and latest['latest_diagnostic']=='076' and not latest['qualityAccepted']
ply=V/'models/whole-body-static-fullscene/shuolee_075_s18000_f0100.ply'
assert sha(ply)==latest['sha256']=='8e59a8e4c7566034d7597890f86176526eb3a7b0c189679cb937b7863c5b6850'
assert sha(Path(record['deck']))==record['deck_sha256']
for name in ['page-verified.png','viewer-verified.png']:assert (O/name).stat().st_size>1000
for item in record['published_assets']:assert sha(V/'body-fullscene-constraint-current'/item['file'])==item['sha256']
src=O/'source';src.mkdir(exist_ok=True)
paths=[B.parents[2]/'repos/PGSR-shuolee/audit_fullscene_ghost_076.py',B.parents[2]/'repos/PGSR-shuolee/audit_fullscene_correspondence_076.py']
# The lab root is explicit rather than inferred from a variable-depth output tree.
repo=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/repos/PGSR-shuolee')
paths=[repo/'audit_fullscene_ghost_076.py',repo/'audit_fullscene_correspondence_076.py',W/'witness_fullscene_076.py',W/'publish_constraint_076.py']
for p in paths:shutil.copy2(p,src/p.name)
proof=dict(updated=datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
    diagnostic_completed=True,new_gaussian_training=False,gaussian_round='075',gaussian_sha256=sha(ply),
    model_unchanged=True,actual_browser_verified=True,report_images_loaded=4,
    source_files=[dict(name=p.name,sha256=sha(p)) for p in src.iterdir() if p.is_file()],
    page_screenshot=str(O/'page-verified.png'),viewer_screenshot=str(O/'viewer-verified.png'),
    ppt_local_updated=True,ppt_changed_slides=[52],pc00_deployed=False,
    quality_accepted=False,root_cause_established=False)
save(O/'completion.json',proof)
print(json.dumps({k:v for k,v in proof.items() if k!='source_files'},ensure_ascii=False,indent=2))
