"""Build a separate offline comparison viewer, preserving the original viewer."""
import hashlib
import json
from pathlib import Path
import shutil
import re
from viewer_capture_frame import apply_capture_frame

LAB=Path(__file__).resolve().parents[2]
BASE=LAB/'outputs/mini-splatting2-computer00-001/viewer'
OUT=LAB/'outputs/mini-splatting2-mip-depth-001'
VIEW=OUT/'viewer'
NAMES={'original':('原始 Mini-Splatting2','original'),
       'control':('基线续训 · 2000步','classic-refine-s0-r1'),
       'mip':('Mip 滤波＋微调','mip-refine-s0'),
       'depth':('深度约束＋微调','depth-refine-s0'),
       'mip-depth':('Mip＋深度约束＋微调','mip-depth-refine-s0')}


def main():
    for key,label,name in [('native-base','原版完整重训 · 18000步','native-baseline-18000-s0'),
                           ('native-motion','运动遮罩重训 · 18000步','native-motion-rgb-18000-s0'),
                           ('native-mip','Mip 完整重训 · 18000步','native-mip-18000-s0'),
                           ('native-motion-mip','运动遮罩＋Mip · 18000步','native-motion-mip-18000-s0'),
                           ('native-motion-mip-depthmask','运动遮罩＋Mip＋补点遮罩','native-motion-mip-depthmask-18000-s0'),
                           ('native-mip-depth003','完整训练＋深度指导 0.003','native-depthguide003-18000-s0'),
                           ('native-mip-depth010','完整训练＋深度指导 0.01','native-depthguide010-18000-s0')]:
        if (OUT/name/'evaluation-summary.json').exists():NAMES[key]=(label,name)
    VIEW.mkdir(exist_ok=True)
    shutil.copytree(BASE/'vendor',VIEW/'vendor',dirs_exist_ok=True)
    for f in ['viewer.css','cameras.json']:shutil.copy2(BASE/f,VIEW/f)
    original=json.loads((BASE/'model.json').read_text(encoding='utf-8'))
    for key,(label,name) in NAMES.items():
        dest=VIEW/'models'/key/'point_cloud.ply';dest.parent.mkdir(parents=True,exist_ok=True)
        is_native=key.startswith('native-')
        source=BASE/original['model'] if name=='original' else OUT/name/'point_cloud.ply'
        if is_native:source=OUT/name/'model/point_cloud/iteration_18000/point_cloud.ply'
        if is_native and 'mip' in key:source=source.with_name('point_cloud_mip3d_baked.ply')
        shutil.copy2(source,dest)
        with source.open('rb') as f:count=int(re.search(rb'element vertex (\d+)',f.read(4096)).group(1))
        meta=dict(original);meta.update(label=label,model=f'models/{key}/point_cloud.ply',
            modelBytes=dest.stat().st_size,sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
            status='原始基线' if key=='original' else '局部实验候选，尚未完成整体任务',
            filter={'preBlurAmount':0 if 'mip' in key else .3,'blurAmount':.1 if 'mip' in key else 0},
            gaussianCount=count,refinement_steps=0 if key=='original' or is_native else 2000,
            comparability_group='native-18000' if is_native else 'fixed-topology-2000',
            scope=('完整18000步重训，包含增密与简化；运动遮罩改变RGB监督，Mip改变三维与二维滤波；重复次数见报告' if is_native else '固定原始149413个高斯后的微调；大幅俯仰与漂浮物未验收'),
            provenance=name)
        meta.pop('serverPath',None);meta.pop('metrics',None)
        if 'depthguide' in name:
            run=json.loads((OUT/name/'manifest.json').read_text())
            meta['depth_guidance']=dict(coefficient=run['depth_weight'],seed=run['seed'],definition='confidence-weighted inverse expected camera z',measured_depth_gt=False)
            meta['scope']='完整18000步重训，保留Mip与运动/补点遮罩，另加训练视角深度指导；该权重目前只有一个种子，几何和大幅俯仰尚未验收'
        (VIEW/f'model-{key}.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    html=(BASE/'index.html').read_text(encoding='utf-8')
    html=html.replace('已训练完成 · Mini-Splatting2 三维查看器','Mini-Splatting2 优化实验对照')
    html=html.replace('TRAINED 3D GAUSSIANS · 01','MINI-SPLATTING2 · LOCAL ABLATIONS')
    html=html.replace('训练完成 <small>18,000 次迭代</small>','实验候选 <small>尚未完成全部验收</small>')
    html=html.replace('电脑00的桌面视频 · 静态场景基线','电脑00工作视频 · 分组实验对照')
    html=html.replace('首次加载约 37 MB，请稍候','首次载入模型，请稍候')
    html=html.replace('<strong>149,413</strong>','<strong id="model-count">149,413</strong>')
    options='<optgroup label="第一组：原模型与2000步微调">'+''.join(f'<option value="{k}">{v[0]}</option>' for k,v in NAMES.items() if not k.startswith('native-'))+'</optgroup>'
    if 'native-base' in NAMES:options+='<optgroup label="第二组：完整18000步重训">'+''.join(f'<option value="{k}">{v[0]}</option>' for k,v in NAMES.items() if k.startswith('native-'))+'</optgroup>'
    html=html.replace('<div class="actions">',f'<label class="sectionlabel" for="model-select">切换模型 · 保持当前观察位置</label><select id="model-select">{options}</select><div class="actions">')
    html=html.replace('comparison/index.html','report.html').replace('原图 / 官方渲染对照 ↗','原图 / 实验效果与结论 ↗')
    a=html.index('<details>');b=html.index('</details>',a)+len('</details>')
    html=html[:a]+'''<details><summary>实验口径与模型信息</summary><p>全部候选从原始18000步模型开始，固定高斯数量，分别微调2000步。保持相同相机、51张训练图与8张留出图。</p><p>Mip 候选的三维滤波已融合进 PLY，并在查看器启用相应的二维面积补偿。浏览器与 CUDA 的排序、精度和投影仍可能有差别，量化结果以报告为准。</p><p>自由浏览的极端距离与俯仰没有对应真实图像；画面改善不能直接证明几何恢复正确。</p><a id="metadata" href="model-original.json" target="_blank">查看当前模型标记与校验值</a></details>'''+html[b:]
    html=html.replace('全部候选从原始18000步模型开始，固定高斯数量，分别微调2000步。保持相同相机、51张训练图与8张留出图。','第一组从原始18000步模型开始，固定高斯数量微调2000步。第二组为原版和运动遮罩的完整18000步重训，包含增密与简化。两组的实验口径分别报告。')
    if 'native-motion' in NAMES:html=html.replace('<details>', '<a class="report-link" href="motion.html">运动残影 · 完整重训对照 ↗</a><details>',1)
    (VIEW/'index.html').write_text(html,encoding='utf-8')
    js=(BASE/'viewer.js').read_text(encoding='utf-8')
    js=js.replace("async function init(){", """async function init(){
 const variants=['original','control','mip','depth','mip-depth'];
 const query=new URLSearchParams(location.search);const variant=variants.includes(query.get('model'))?query.get('model'):'original';
 $('model-select').value=variant;
 $('model-select').onchange=()=>{sessionStorage.setItem('mini-optimization-pose',JSON.stringify({position:camera.position.toArray(),quaternion:camera.quaternion.toArray(),up:camera.up.toArray(),target:controls.target.toArray(),fov:camera.fov,selected,mode:$('mode').textContent}));location.search='?model='+$('model-select').value;};
 $('metadata').href='model-'+variant+'.json';
 """)
    js=js.replace("fetch('model.json')","fetch('model-'+variant+'.json')")
    js=js.replace("const variants=['original','control','mip','depth','mip-depth'];",'const variants='+json.dumps(list(NAMES))+';')
    js=js.replace('focus=new THREE.Vector3(...meta.focus);',"focus=new THREE.Vector3(...meta.focus);$('model-count').textContent=meta.gaussianCount.toLocaleString();")
    js=js.replace('preBlurAmount:.3,blurAmount:0','preBlurAmount:meta.filter.preBlurAmount,blurAmount:meta.filter.blurAmount')
    js=js.replace("overview();mesh=new SplatMesh", """overview();
 const remembered=sessionStorage.getItem('mini-optimization-pose');if(remembered){try{const p=JSON.parse(remembered);camera.position.fromArray(p.position);camera.quaternion.fromArray(p.quaternion);camera.up.fromArray(p.up);camera.fov=p.fov;camera.updateProjectionMatrix();newControls(new THREE.Vector3(...p.target));selected=p.selected??24;$('views').value=String(selected);$('timeline').value=String(selected);$('viewlabel').textContent=label(selected);$('mode').textContent=p.mode||'保留观察位置';}catch(e){console.warn(e);}}
 $('download').href=meta.model+'?download=1';$('download').download='MiniSplatting2_'+variant+'.ply';
 const requestedView=Number(query.get('view'));if(query.has('view')&&Number.isInteger(requestedView)&&requestedView>=0&&requestedView<cameras.length)choose(requestedView);
 mesh=new SplatMesh""")
    js=js.replace('149,413 个高斯 · 18,000 次迭代','${meta.gaussianCount.toLocaleString()} 个高斯 · ${meta.label}')
    js=js.replace("window.viewerState={...window.viewerState,status:'ready'","window.viewerState={...window.viewerState,variant,filter:meta.filter,status:'ready'")
    (VIEW/'viewer.js').write_text(apply_capture_frame(js),encoding='utf-8')
    # Build a portable host on another port/identity; no original binaries changed.
    cs=(BASE/'ViewerHost.cs').read_text(encoding='utf-8')
    cs=cs.replace('MiniSplatting2-HOVER-18000-0751327e','MiniSplatting2-MipDepth-20261018')
    cs=cs.replace('"models","TRAINED_18000"','"models","original"').replace('port=8810;port<=8819','port=8820;port<=8829')
    cs=cs.replace('已训练3DGS · Mini-Splatting2','Mini-Splatting2 优化对照').replace('8810–8819','8820–8829')
    cs=cs.replace('case ".png":mime="image/png";break;', 'case ".png":mime="image/png";break;case ".jpg":mime="image/jpeg";break;case ".mp4":mime="video/mp4";break;')
    (VIEW/'ViewerHost.cs').write_text(cs,encoding='utf-8')
    print(VIEW)


if __name__=='__main__':main()
