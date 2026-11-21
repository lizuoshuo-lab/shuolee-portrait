from pathlib import Path
import json,hashlib,re,datetime
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee');P=B/'current/avatar-repository-pair-114';V=B.parent/'viewer'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
run=P/'gaussianavatar/gaussian-run';r=read(run/'native-vs-ply.json');assert r['step']==1000 and r['passed']
d=dict(created=datetime.datetime.now().astimezone().isoformat(),step=1000,actual_images_reviewed=[dict(path=str(run/n),sha256=sha(run/n)) for n in ['heads-current.jpg','review-current.jpg']],observations='Coherent coarse clothed body is present in all six views. Face, glasses and hair are blurred; sleeve/hem/shoes retain contour errors. Smoothing is not evidence that duplicated geometry was corrected.',quality_accepted=False,stop_condition_triggered=False,action='Continue the same prespecified18000-step run without modifying its settings; compare direct A/B after both complete.')
(P/'gaussianavatar/first-stage-review.json').write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
page=V/'body-smplx.html';s=page.read_text(encoding='utf-8')
start=s.index('<section id="avatar114-pair">');end=s.index('<section id="native113-result">',start);part=s[start:end]
part,n=re.subn(r'(<figure><a href="body-avatar114/gaussianavatar/initial-mesh.jpg">.*?</figure>)',r'<details><summary>本轮静态人体初始化叠图</summary>\1</details>',part,count=1,flags=re.S);assert n==1
part=part.replace('<div id="avatar114-gaussianavatar-images"></div>','<p>首批1,000步已实际检查：穿衣全身轮廓已形成，脸部、镜框和衣服边缘仍模糊；不能据此判断重影已解决。继续同一轮预定训练。</p><div id="avatar114-gaussianavatar-images"></div>',1)
page.write_text(s[:start]+part+s[end:],encoding='utf-8')
print('1000-step actual review recorded; no quality claim')
