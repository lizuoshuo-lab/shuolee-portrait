"""Append the two completed resolution experiments without replacing previous models."""
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
import numpy as np
from plyfile import PlyData

ROOT=Path(__file__).resolve().parents[2]/'outputs/mini-splatting2-mip-depth-001'


def main():
    viewer=ROOT/'viewer';audit=ROOT/'viewer-resolution-registration-001';audit.mkdir(exist_ok=False)
    for name in ['index.html','viewer.js']:shutil.copy2(viewer/name,audit/('before-'+name))
    template=json.loads((viewer/'model-native-mip-depth003.json').read_text())
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in viewer.glob('model-*.json')};assert len(before)==15
    registered=[]
    for arm,size,label in [('low',[757,1347],'分辨率对照 · 757 × 1347 训练'),('full',[1514,2694],'分辨率对照 · 1514 × 2694 训练')]:
        key='native-mip-resolution-'+arm;run='native-resolution-cache-'+arm+'-18000-s0'
        manifest=json.loads((ROOT/run/'manifest.json').read_text());assert manifest['status']=='complete'
        evaluation=json.loads((ROOT/run/'resolution-evaluation-v2/summary.json').read_text());assert evaluation['status']=='complete'
        source=Path(manifest['baked_ply']);destination=viewer/'models'/key/'point_cloud.ply'
        destination.parent.mkdir(exist_ok=False);shutil.copy2(source,destination)
        digest=hashlib.sha256(destination.read_bytes()).hexdigest();assert digest==manifest['baked_ply_sha256']
        vertices=PlyData.read(str(source))['vertex'];xyz=np.stack([vertices['x'],vertices['y'],vertices['z']],-1)
        meta=copy.deepcopy(template);meta.pop('archiveSha256',None)
        meta.update(label=label,status='完整18,000步实验模型 · 单种子，尚未验收',gaussianCount=len(vertices),
            model=f'models/{key}/point_cloud.ply',modelBytes=destination.stat().st_size,sha256=digest,
            bounds05=np.percentile(xyz,5,axis=0).tolist(),bounds95=np.percentile(xyz,95,axis=0).tolist(),
            provenance=run,scope='相同原视频、相机、稀疏点、遮罩与0.003平均深度约束；只比较整套流程对训练分辨率的响应。',
            resolution_ablation=dict(training_size=size,seed=0,iterations=18000,depth_weight=.003,
                original_joint_backward=True,gpu_reference_cache_bound=4,accepted=False),
            report='native-resolution.html',comparability_group='native-resolution-cache-18000-seed0')
        (viewer/('model-'+key+'.json')).write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
        registered.append(dict(key=key,label=label,sha256=digest,gaussians=len(vertices)))
    html=(viewer/'index.html').read_text(encoding='utf-8')
    group='<optgroup label="第四组：原视频分辨率对照 · 单种子 · 尚未验收">'+''.join('<option value="'+r['key']+'">'+r['label']+'</option>' for r in registered)+'</optgroup>'
    match=re.search(r'(<select id="model-select">)(.*?)(</select>)',html,re.S);assert match
    html=html[:match.end(2)]+group+html[match.end(2):];(viewer/'index.html').write_text(html,encoding='utf-8')
    js=(viewer/'viewer.js').read_text(encoding='utf-8');match=re.search(r'const variants=(\[.*?\]);',js);assert match
    keys=json.loads(match[1]);assert len(keys)==15;keys.extend(r['key'] for r in registered)
    js=js[:match.start(1)]+json.dumps(keys)+js[match.end(1):];(viewer/'viewer.js').write_text(js,encoding='utf-8')
    assert all(hashlib.sha256((viewer/name).read_bytes()).hexdigest()==sha for name,sha in before.items())
    state=dict(status='registered_pending_browser_review',created=time.time(),models=registered,total_models=17,
        previous_metadata_unchanged=True,previous_model_files_untouched=True)
    (audit/'summary.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(state,ensure_ascii=False))


if __name__=='__main__':main()
