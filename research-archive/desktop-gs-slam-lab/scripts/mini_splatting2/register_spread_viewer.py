"""Append three experimental models; preserve all previous models and pages."""
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
    viewer=ROOT/'viewer';audit=ROOT/'viewer-spread-registration-001';audit.mkdir(exist_ok=False)
    for name in ['index.html','viewer.js']:shutil.copy2(viewer/name,audit/('before-'+name))
    template=json.loads((viewer/'model-native-mip-depth003.json').read_text())
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in viewer.glob('model-*.json')}
    assert len(before)==12
    configs=[('000',0.,'深度分散 · 零权重对照'),('003',.003,'深度分散 · 0.003 候选'),('010',.01,'深度分散 · 0.010 候选')]
    registered=[]
    for suffix,weight,label in configs:
        key='native-mip-spread'+suffix;run='native-depthspread'+suffix+'-18000-s0'
        m=json.loads((ROOT/run/'manifest.json').read_text());assert m['status']=='complete'
        source=ROOT/run/'model/point_cloud/iteration_18000/point_cloud_mip3d_baked.ply'
        destination=viewer/'models'/key/'point_cloud.ply';destination.parent.mkdir(exist_ok=False)
        shutil.copy2(source,destination);sha=hashlib.sha256(destination.read_bytes()).hexdigest();assert sha==m['baked_ply_sha256']
        v=PlyData.read(str(source))['vertex'];xyz=np.stack([v['x'],v['y'],v['z']],-1)
        meta=copy.deepcopy(template);meta.pop('archiveSha256',None)
        meta.update(label=label,status='实验对照：尚未证明人物更清晰',gaussianCount=len(v),
            model=f'models/{key}/point_cloud.ply',modelBytes=destination.stat().st_size,sha256=sha,
            bounds05=np.percentile(xyz,5,axis=0).tolist(),bounds95=np.percentile(xyz,95,axis=0).tolist(),
            provenance=run,scope='单种子完整18000步训练，固定Mip与0.003平均深度指导；仅改变高斯中心深度分散权重。人物清晰度与真实几何均未验收。',
            spread_guidance=dict(coefficient=weight,seed=0,definition='confidence-weighted normalized Gaussian-center camera-z variance',
                accepted_for_human_clarity=False,measured_depth_gt=False),
            report='native-spread.html',comparability_group='native-spread-18000-seed0')
        (viewer/('model-'+key+'.json')).write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
        registered.append(dict(key=key,label=label,sha256=sha,gaussians=len(v)))
    html=(viewer/'index.html').read_text(encoding='utf-8')
    group='<optgroup label="第三组：深度分散实验 · 单种子 · 尚未验收">'+''.join('<option value="'+r['key']+'">'+r['label']+'</option>' for r in registered)+'</optgroup>'
    select=re.search(r'(<select id="model-select">)(.*?)(</select>)',html,re.S);assert select
    html=html[:select.end(2)]+group+html[select.end(2):]
    html=html.replace('两组的实验口径分别报告。','第三组固定平均深度指导，仅比较深度分散约束；头肩细节尚未改善。各组的实验口径分别报告。')
    (viewer/'index.html').write_text(html,encoding='utf-8')
    js=(viewer/'viewer.js').read_text(encoding='utf-8');match=re.search(r'const variants=(\[.*?\]);',js);assert match
    keys=json.loads(match[1]);assert len(keys)==12
    keys.extend(r['key'] for r in registered)
    js=js[:match.start(1)]+json.dumps(keys)+js[match.end(1):]
    (viewer/'viewer.js').write_text(js,encoding='utf-8')
    assert all(hashlib.sha256((viewer/n).read_bytes()).hexdigest()==h for n,h in before.items())
    result=dict(status='registered_pending_browser_review',created=time.time(),models=registered,
        total_models=15,previous_metadata_unchanged=True,previous_model_files_untouched=True)
    (audit/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
