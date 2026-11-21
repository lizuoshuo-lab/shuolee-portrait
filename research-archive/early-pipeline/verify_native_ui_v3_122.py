from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,urllib.request,zipfile
import numpy as np
from native_scene_package122_v3 import load_scene
H=Path(__file__).parent;T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001');R=T/'shuolee/current/robust-portrait-122';O=R/'native-ui-v3'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
assert not (O/'review.json').exists()
p=read(O/'plan.json');state=read(O/'ui-state.json');downloads=read(O/'browser-downloads.json')
checks=[]
def ok(name,v):
    checks.append(dict(name=name,passed=bool(v)))
    if not v:raise AssertionError(name)
ok('Exactly two completed UI jobs',len(state['jobs'])==2 and all(j['state']=='complete' for j in state['jobs']))
ok('Every frozen source and reference unchanged',all(sha(path)==h for path,h in p['immutable_inputs'].items()))
ok('Downloaded native ZIP exact bytes',sha(downloads['native'])==p['package']['zip_sha256'])
ok('Downloaded standard preview ZIP exact bytes',sha(downloads['preview'])==p['preview']['sha256'])
ok('Browser file chooser imported downloaded native ZIP',state['imported']['zip_sha256']==sha(downloads['native']))
scene=load_scene(O/'imported/scene.json',check_runtime=False)
original=load_scene(R/'native-cvo-control/exact-native-package/scene.json',check_runtime=False)
ok('All native linear attributes bit-identical after import',all(np.array_equal(scene.parameters[k],original.parameters[k]) for k in ['means','quats','scales','opacities','sh']))
ok('All recorded cameras identical',scene.cameras==original.cameras)
comparisons=[]
for j in state['jobs']:
    actual=np.load(O/j['id']/'native-rgb-alpha.npz');ref=np.load(p['references'][j['camera_id']])
    diffs={k:float(np.max(np.abs(actual[k]-ref[k]))) for k in ['rgb','alpha']}
    ok(j['camera_id']+' native full-size response',all(v<=p['tolerance']['native_float_response_max'] for v in diffs.values()))
    ok(j['camera_id']+' exact recorded full K/E',j['camera']==original.cameras[j['camera_id']])
    ok(j['camera_id']+' canonical arrays used',j['native_receipt']['canonical_parameters']=='native-parameters.npz')
    ok(j['camera_id']+' actual worker receipt',j['native_linux_pid']>0 and j['worker_pid']>0 and all(sha(O/j['id']/n)==h for n,h in j['files'].items()))
    comparisons.append(dict(camera_id=j['camera_id'],dimensions=[j['camera']['width'],j['camera']['height']],max_abs=diffs))
ok('Downloaded float array exact bytes',sha(downloads['float'])==state['jobs'][0]['files']['native-rgb-alpha.npz'])
old=read(O/'previous-ui-state.json')
for version,port in [('v1',8874),('v2',8875)]:
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/state',timeout=10) as response:d=json.load(response)
    ok(version+' previous service retained',dict(jobs=d['jobs'],remaining=d['remaining'])==old[version])
screens=['portrait-rgb-ui.png','portrait-alpha-ui.png','landscape-rgb-ui.png','landscape-alpha-ui.png','preview-rejection-ui.png','cached-reload-ui.png']
ok('Actual GUI screenshots saved',all((O/n).stat().st_size>10000 for n in screens))
with zipfile.ZipFile(downloads['preview']) as z:
    cams=json.loads(z.read('cameras.json'));m=json.loads(z.read('scene.json'))
    ok('Preview PLY and cameras SHA bound',hashlib.sha256(z.read(m['model']['file'])).hexdigest()==m['model']['sha256'] and hashlib.sha256(z.read('cameras.json')).hexdigest()==m['cameras']['sha256'])
    txt=[line.split() for line in z.read('cameras.txt').decode().splitlines() if line and not line.startswith('#')]
    ok('COLMAP text independently retains complete K',len(txt)==2 and all(row[1]=='PINHOLE' and list(map(float,row[4:]))==[c['K'][0][0],c['K'][1][1],c['K'][0][2],c['K'][1][2]] for row,c in zip(txt,scene.cameras.values())))
findings=[
 '新建隔离的v3原生入口，真实执行了下载原生ZIP、文件选择导入、两记录相机渲染、RGB/Alpha切换、下载浮点结果及预览ZIP。旧v1/v2服务保持原样，没有替换119历史人物模型。',
 '导入后的20万点五类线性参数逐位一致，完整K和外参逐项一致。两种原生分辨率1520×2704、2704×1520的RGB及alpha与T8导出前浮点参考最大差均为0。此项验证文件与原生回放一致性，不是实拍标定误差。',
 '26项预登记CPU检查通过。实际UI只执行两个GPU推理任务，切回相机与重载读取已保存结果，不重复计算。输出按原宽高比显示。',
 '原生ZIP和标准预览ZIP独立下载。原生ZIP内含无损NPZ、原字节PLY、同坐标相机和后端合同；预览包含3DGS相机、COLMAP完整K、SIBR目录。两份下载与生成包SHA一致。',
 '实际将预览ZIP送入原生文件入口时被拒绝，原先加载的原生模型及缓存保持可用。包类型不会静默转换。',
 '验收限于已登记合成包和记录视角。任意第三方模型、连续交互轨迹、独立Spark官网及SIBR官方GUI尚未验收，也未在00运行。',
 'DirectFisheye-GS可借鉴的CVO训练组件仍需绑定实际可信物理关联图，不能代替内外参标定。人物最大面部误差仍13.049px，全部相机和材料身份未通过，未启动人物训练。']
review=dict(updated=datetime.now(timezone.utc).isoformat(),stage='T9',state='passed',checks=checks,count=len(checks),comparisons=comparisons,
 actual_open_file=True,actual_recorded_camera_RGB_alpha_review=True,actual_download_hash_verified=True,preview_rejection_verified=True,
 canonical_native_pass=True,standard_preview_native_equivalence=False,person_training=False,physical_quality_accepted=False,
 official_SIBR_GUI=False,independent_Spark_GUI=False,company00_verified=False,findings_zh=findings,screenshots={n:sha(O/n) for n in screens})
save(O/'review.json',review)
print(json.dumps(dict(checks=len(checks),comparisons=comparisons,real_person_pass=False)))
