from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,urllib.request
import numpy as np
from native_scene_package122_v5 import load_scene
H=Path(__file__).parent;T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';O=R/'native-residual-ui';A=O/'launcher-repair'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
assert not (O/'review.json').exists()
p=read(A/'plan.json');s=read(A/'ui-state.json');d=read(O/'browser-downloads.json');checks=[]
def ok(n,v,**details):
 checks.append(dict(name=n,passed=bool(v),**details))
 if not v:raise AssertionError(n)
ok('Exactly two real UI renders complete',len(s['jobs'])==2 and all(j['state']=='complete' for j in s['jobs']))
ok('Frozen sources unchanged',all(sha(path)==digest for path,digest in p['immutable_inputs'].items()))
ok('Downloaded native ZIP exact bytes',sha(d['native'])==p['package']['zip_sha256'])
ok('Actual downloaded ZIP imported',s['imported']['zip_sha256']==sha(d['native']))
scene=load_scene(A/'imported/scene.json',check_runtime=False)
source=load_scene(R/'native-residual-capacity/native-package/scene.json',check_runtime=False)
ok('All full linear arrays bit identical',all(np.array_equal(scene.parameters[k],source.parameters[k]) for k in ['means','quats','scales','opacities','sh']))
ok('Each full K E D record exact',scene.cameras==source.cameras)
comparison=[]
for j in s['jobs']:
 with np.load(A/j['id']/'native-rgb-alpha.npz') as z:actual={k:z[k] for k in z.files}
 with np.load(p['references'][j['camera_id']]) as z:reference={k:z[k] for k in z.files}
 diff={k:float(np.max(np.abs(actual[k]-reference[k]))) for k in ['rgb','alpha']}
 ok(j['camera_id']+' original-size response',max(diff.values())<=p['tolerance']['native_float_response_max'],**diff)
 ok(j['camera_id']+' complete camera in worker receipt',j['native_receipt']['camera']==source.cameras[j['camera_id']])
 ok(j['camera_id']+' exact canonical arrays',j['native_receipt']['canonical_parameters']=='native-parameters.npz')
 ok(j['camera_id']+' worker and output provenance',j['native_linux_pid']>0 and all(sha(A/j['id']/n)==v for n,v in j['files'].items()))
 comparison.append(dict(camera_id=j['camera_id'],size=[j['camera']['width'],j['camera']['height']],max_abs=diff))
ok('Downloaded float response bytes exact',sha(d['float'])==s['jobs'][0]['files']['native-rgb-alpha.npz'])
old=read(O/'previous-ui-state.json')
for version,port in [('v1',8874),('v2',8875),('v3',8876)]:
 with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/state',timeout=10) as f:x=json.load(f)
 ok(version+' previous jobs and budget retained',dict(jobs=x['jobs'],remaining=x['remaining'])==old[version])
screens=['portrait-rgb-ui.png','portrait-alpha-ui.png','landscape-rgb-ui.png','landscape-alpha-ui.png','old-package-rejection-ui.png','cached-reload-ui.png']
ok('Actual UI review screenshots present',all((O/n).stat().st_size>10000 for n in screens))
prior=read(O/'ui-state.json')
ok('Launcher failure retained with no CUDA output',prior['jobs'][0]['state']=='failed' and not (O/prior['jobs'][0]['id']).exists())
stamp=datetime.now(timezone.utc).isoformat()
review=dict(updated=stamp,state='passed',checks=checks,count=len(checks),comparisons=comparison,
 actual_file_import=True,actual_RGB_alpha_review=True,actual_download_hash_verified=True,old_package_rejection_verified=True,
 native_replay_passed=True,physical_quality_accepted=False,person_training=False,
 continuous_orbit_verified=False,arbitrary_third_party_import=False,official_SIBR_GUI=False,independent_Spark_GUI=False,company00_verified=False,
 screenshots={n:sha(O/n) for n in screens},launcher_repair=p['repair'],
 findings_zh=[
 '实际从浏览器下载原生ZIP，再通过打开文件入口导入。20万点线性数组与两套完整K/E/D逐项一致。',
 '两次实际原生GPU回放分别为1520×2704和2704×1520，RGB与Alpha同导出前参考浮点误差均为0。切换和重载使用已保存结果，没有新渲染任务。',
 '旧零畸变包在新入口被拒绝，已加载的残余畸变场景保持可用；不会静默补零或删去系数。',
 '首次界面调用因Windows CRLF换行在Bash第2行失败，未启动Python/CUDA。保留失败后仅改为LF，新进程完成原定两个渲染。',
 '该界面只验收已登记合成包和保存视角，连续未知视角、任意第三方包、官方SIBR或Spark及00尚未验收。真实人物标定与质量仍未通过。'])
save(O/'review.json',review)
print(json.dumps(dict(checks=len(checks),comparisons=comparison,physical_quality_passed=False)))
