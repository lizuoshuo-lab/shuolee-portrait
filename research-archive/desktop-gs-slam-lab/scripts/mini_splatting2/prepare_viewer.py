from pathlib import Path
import hashlib, json, shutil, tarfile
import numpy as np

lab = Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab')
stage = Path(r'C:\Users\Administrator\Downloads\05_mini_splatting2_viewer')
root = stage / 'VIEWER'
out = lab / 'outputs/mini-splatting2-computer00-001'
tools = lab / 'tools/mini-splat-viewer'
archive = stage / 'trained-model.tar.gz'
digest = hashlib.sha256(archive.read_bytes()).hexdigest()
assert digest == (stage/'trained-model.tar.gz.sha256').read_text().split()[0]
(root/'models/TRAINED_18000').mkdir(parents=True, exist_ok=True)
with tarfile.open(str(archive)) as tar:
    expected = {'point_cloud/iteration_18000/point_cloud.ply': root/'models/TRAINED_18000/point_cloud.ply',
                'cameras.json':root/'cameras.json', 'cfg_args':root/'models/TRAINED_18000/cfg_args.txt'}
    assert set(tar.getnames()) == set(expected)
    for member, target in expected.items():
        target.write_bytes(tar.extractfile(member).read())
model = root/'models/TRAINED_18000/point_cloud.ply'
with model.open('rb') as f:
    lines = []
    while True:
        line = f.readline().decode('ascii').strip(); lines.append(line)
        if line == 'end_header': break
    assert 'format binary_little_endian 1.0' in lines
    count = int(next(x for x in lines if x.startswith('element vertex ')).split()[-1])
    props = [x.split()[-1] for x in lines if x.startswith('property float ')]
    data = np.frombuffer(f.read(), dtype='<f4').reshape(count,len(props))
assert count == 149413 and np.isfinite(data).all()
assert len([p for p in props if p.startswith('f_rest_')]) == 45
xyz = data[:,[props.index(i) for i in ['x','y','z']]]
cams = sorted(json.loads((root/'cameras.json').read_text()),key=lambda c:c['img_name'])
A = np.zeros((3,3)); b = np.zeros(3)
for c in cams:
    ray = np.array(c['rotation'])[:,2]; proj = np.eye(3)-np.outer(ray,ray)
    A+=proj; b+=proj@np.array(c['position'])
focus = np.linalg.solve(A,b)
meta = {'label':'Mini-Splatting2 · HOVER VERSA 0102','status':'训练完成','iterations':18000,
    'gaussianCount':count,'shDegree':3,'trainViews':51,'testViews':8,'sourceViews':59,
    'model':'models/TRAINED_18000/point_cloud.ply','modelBytes':model.stat().st_size,
    'sha256':hashlib.sha256(model.read_bytes()).hexdigest(),'archiveSha256':digest,
    'focus':focus.tolist(),'bounds05':np.quantile(xyz,.05,axis=0).tolist(),
    'bounds95':np.quantile(xyz,.95,axis=0).tolist(),
    'metrics':{'PSNR':29.40277862548828,'SSIM':.9442502856254578,'LPIPS':.1066034585237503},
    'scope':'静态场景基线训练完成；动态人物仍有残影，尚未达到清晰重建目标。',
    'serverPath':'/home/zhouhaoyi/codex-mini-splatting2-20261016/hover-msv2-fast-001/point_cloud/iteration_18000/point_cloud.ply',
    'viewer':{'engine':'Spark 2.3.1 / Three.js 0.180.0','sh':3,'lod':False,'pruning':False,
              'note':'浏览器实时光栅化与官方 CUDA 输出可能存在细微差异；精确对照见原图与官方渲染。'}}
(root/'model.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
(root/'cameras.json').write_text(json.dumps(cams),encoding='utf-8')
(root/'models/TRAINED_18000/训练完成_模型说明.txt').write_text('已训练完成的 3DGS 模型\n\nMini-Splatting2 / msv2 / fast / 18,000 次迭代\n149,413 个高斯；三阶球谐；51 张训练，8 张留出视图。\n\npoint_cloud.ply 是服务器最终模型原件，未经删点或重新训练。\nSHA256: '+meta['sha256']+'\n\n服务器原件：'+meta['serverPath']+'\n\n这是静态视频基线；移动人物残影仍然存在。\n',encoding='utf-8-sig')
vendor = root/'vendor'; (vendor/'postprocessing').mkdir(parents=True,exist_ok=True)
for source,target in [('spark/dist/spark.module.js','spark.module.js'),('spark/LICENSE','SPARK-LICENSE.txt'),
    ('three180/build/three.module.js','three.module.js'),('three180/build/three.core.js','three.core.js'),
    ('three180/examples/jsm/controls/OrbitControls.js','OrbitControls.js'),
    ('three180/examples/jsm/postprocessing/Pass.js','postprocessing/Pass.js'),('three180/LICENSE','THREE-LICENSE.txt')]:
    shutil.copy2(tools/source,vendor/target)
for name in ['gt','renders']:
    shutil.copytree(out/'delivery'/name,root/'comparison'/name,dirs_exist_ok=True)
shutil.copy2(out/'delivery/index.html',root/'comparison/index.html')
for source in (out/'delivery').glob('*.json'): shutil.copy2(source,root/'comparison'/source.name)
print(json.dumps(meta,ensure_ascii=False,indent=2))
