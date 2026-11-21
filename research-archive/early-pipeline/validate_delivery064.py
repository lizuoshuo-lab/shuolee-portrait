"""Record completed real-browser checks only after evidence files exist."""
from pathlib import Path
import datetime,hashlib,json,urllib.request
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
R=B/'current/person-static-pgsr-fresh-064';V=B.parent/'viewer'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8');q.replace(p)
pub=V/'models/whole-body-static-pgsr/delivery.json';d=read(pub);ply=Path(d['download_path'])
assert hashlib.sha256(ply.read_bytes()).hexdigest()==d['sha256']
with urllib.request.urlopen(urllib.request.Request(d['download_url'],method='HEAD'),timeout=15) as response:
    assert response.status==200 and int(response.headers['Content-Length'])==ply.stat().st_size
names=['preset-gaussian-proof.png','standalone-ply-gaussian-proof.png','standalone-ply-points-proof.png','delivery-page-proof.png']
assert all((R/'review'/n).stat().st_size>1000 for n in names)
obs=read(R/'review/browser-observation.json')
assert obs['points']==d['points'] and obs['sh_degree']==3 and obs['images_loaded']==4
assert obs['standalone_open_file'] and obs['default_y_up'] and not obs['keep_view']
check=dict(created=datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
    local_viewer=True,actual_browser_interaction=True,preset_load_passed=True,standalone_open_file_passed=True,
    keep_previous_view=False,orientation='default +Y up',gaussian_mode_passed=True,center_point_mode_passed=True,
    displayed_points=obs['points'],displayed_sh_degree=3,quality_accepted=False,pc00_opened_this_round=False,
    no_external_template_or_transform=True,proof_sha256={n:hashlib.sha256((R/'review'/n).read_bytes()).hexdigest() for n in names},
    page_images_loaded=4,http_download=dict(status=200,bytes=ply.stat().st_size),ply_sha256=d['sha256'])
save(R/'review/viewer-validation.json',check)
for p in [pub,R/'gaussian-run/viewer-export/delivery.json']:
    v=read(p);v['local_viewer_validation']=check;save(p,v)
idx=read(B/'gaussian-deliveries.json');idx['rounds']['064']['local_viewer_validation']=check;idx['updated']=check['created'];save(B/'gaussian-deliveries.json',idx)
con=read(R/'phase-conclusion.json');con['local_viewer_validation']=check;save(R/'phase-conclusion.json',con);save(V/'body-static-pgsr-current/conclusion.json',con)
st=read(B/'work-status.json');st['updated']=check['created'];st['latest_local_execution']['local_viewer_passed']=True;save(B/'work-status.json',st)
print(json.dumps(dict(viewer_checks=True,download=d['download_url'],bytes=ply.stat().st_size,pc00_actual_test=False),ensure_ascii=False))
