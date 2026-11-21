from pathlib import Path
import datetime as dt,hashlib,json,urllib.request
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
O=B/'current/portrait-alignment-open-source-116';H=O/'hsfm';V=B.parent/'viewer'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
now=dt.datetime.now(dt.timezone.utc).isoformat()
runtime=read(H/'runtime-check.json');runtime['subject_inference']=True
runtime['subject_inference_scope']='Native four-directed-pair dense scene predictions; no human mesh fit or camera optimization'
save(H/'runtime-check.json',runtime)
receipt=read(O/'viewer-publication-p3-runtime.json');latest=read(V/'latest-result.json')
assert {k:v for k,v in latest.items() if k!='latestDiagnostic'}==receipt['latest_model_unchanged']
assert receipt['ppt_sha256']==sha(B/'reports/02_有限视角重建优化路线.pptx')=='c8f1546cd34e568c5cbb40b0a1d231ac58af52f9bfaaac0e8d85d42eeff87922'
responses=[]
for path in ['body-smplx.html','body-camera116/hsfm-body23-2.jpg','body-camera116/hsfm-native-smoke-result.json','body-camera116/hsfm-weight-acquisition.json','latest-result.json']:
    with urllib.request.urlopen('http://127.0.0.1:8870/'+path,timeout=10) as r:
        data=r.read();assert r.status==200
        if path=='body-smplx.html':assert '尚未执行人体与相机联合拟合'.encode() in data
        responses.append(dict(path=path,status=r.status,bytes=len(data)))
shot=O/'viewer-ui-p3-runtime.png';assert shot.stat().st_size>10000
review=dict(updated=now,state='actual_browser_review_completed',browser_tab='2',
    page='http://127.0.0.1:8870/body-smplx.html#alignment116-hsfm',
    observed=['HSfM runtime status and joint-fit-not-run notice visible',
        'New body23 image loaded with original proportions',
        'Eight native outputs are available through collapsed details',
        'No new point cloud or quality acceptance claimed'],
    screenshot=shot.name,screenshot_sha256=sha(shot),
    preserved_historical_tab='15: whole-body-early-trace; marked handoff without navigating',
    actual_open_file_PLY_review='not_applicable_no_new_PLY',remote00_deployed=False,
    latest_model_identity_unchanged=True,ppt_unchanged=True,http_checks=responses)
save(O/'viewer-ui-p3-runtime-review.json',review)
receipt.update(updated=now,state='published_and_actual_browser_reviewed',browser_review='viewer-ui-p3-runtime-review.json')
save(O/'viewer-publication-p3-runtime.json',receipt)
print(json.dumps(dict(published=True,browser_review=True,model_unchanged=True,ppt_unchanged=True,gaussian_started=False)))
