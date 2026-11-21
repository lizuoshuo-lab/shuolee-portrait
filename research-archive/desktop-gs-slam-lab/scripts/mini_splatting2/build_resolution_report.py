"""Compare paired resolution runs on identical real photos; no automatic acceptance."""
import json
import hashlib
from pathlib import Path
import shutil
import statistics
import time
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]/'outputs/mini-splatting2-mip-depth-001'


def main():
    for arm in ['low','full']:
        name=f'native-resolution-{arm}-18000-s0'
        if not (ROOT/name/'resolution-evaluation-v2/summary.json').exists():
            with (ROOT/name/'evaluate-resolution-v2.log').open('x') as log:
                subprocess.run([sys.executable,str(Path(__file__).with_name('evaluate_resolution_stage.py')),name],stdout=log,stderr=subprocess.STDOUT,check=True)
    models={arm:json.loads((ROOT/f'native-resolution-{arm}-18000-s0/resolution-evaluation-v2/summary.json').read_text()) for arm in ['low','full']}
    assert all(m['status']=='complete' for m in models.values())
    assert models['low']['references']==models['full']['references']
    data=dict(status='complete_pending_visual_review',created=time.time(),models=models,accepted=False,
        frames=[r['frame'] for r in models['low']['rows'] if r['target']=='full'],compared_to_low={},
        builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        evaluator_correction='native-resolution-evaluator-correction-001/after.json; legacy-kernel v1 metrics excluded')
    for target in ['full','low']:
        delta={region:{key:models['full']['aggregate'][target][region][key]-models['low']['aggregate'][target][region][key]
            for key in models['low']['aggregate'][target][region]} for region in ['full','person','head']}
        a=[r for r in models['low']['rows'] if r['target']==target];b=[r for r in models['full']['rows'] if r['target']==target]
        assert [r['frame'] for r in a]==[r['frame'] for r in b]
        per=[dict(frame=x['frame'],head_psnr=y['metrics']['head']['psnr']-x['metrics']['head']['psnr'],
            head_gradient=y['metrics']['head']['normalized_gradient_error']-x['metrics']['head']['normalized_gradient_error']) for x,y in zip(a,b)]
        data['compared_to_low'][target]=dict(delta=delta,per_frame=per,
            head_psnr_positive_views=sum(r['head_psnr']>0 for r in per),
            head_gradient_better_views=sum(r['head_gradient']<0 for r in per))
    out=ROOT/'native-resolution-review-001';out.mkdir(exist_ok=False)
    data['protocol']=json.loads((ROOT/'native-resolution-protocol-001.json').read_text())
    (out/'summary.json').write_text(json.dumps(data,indent=2))
    view=ROOT/'viewer';assets=view/'native-resolution-assets';assets.mkdir(exist_ok=False)
    for arm in ['low','full']:
        source=ROOT/f'native-resolution-{arm}-18000-s0/resolution-evaluation-v2'
        for target in ['full','low']:
            dest=assets/arm/target;dest.mkdir(parents=True)
            rows=[r for r in models[arm]['rows'] if r['target']==target]
            for row in rows:
                frame=row['frame'];p=source/target
                gt=np.asarray(Image.open(p/(frame+'-gt.png')).convert('RGB'),dtype='float32')/255
                pred=np.asarray(Image.open(p/(frame+'-render.png')).convert('RGB'),dtype='float32')/255
                for image in p.glob(frame+'-*.png'):shutil.copy2(image,dest/image.name)
                # The two arms must compare to byte-identical references.
                other=ROOT/'native-resolution-low-18000-s0/resolution-evaluation-v2'/target/(frame+'-gt.png')
                assert hashlib.sha256((p/(frame+'-gt.png')).read_bytes()).digest()==hashlib.sha256(other.read_bytes()).digest()
                error=np.abs(pred-gt).mean(-1)
                rgb=Image.fromarray((plt.get_cmap('magma')(np.clip(error/.1,0,1))[...,:3]*255).round().astype('uint8'))
                rgb.save(dest/(frame+'-error.png'))
                for region in ['head','person']:rgb.crop(tuple(row['regions'][region])).save(dest/(frame+'-'+region+'-error.png'))
    paths=ROOT/'native-resolution-paths-001'
    if paths.exists():
        assert json.loads((paths/'manifest.json').read_text())['status']=='complete'
        shutil.copytree(paths,assets/'paths')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,3.6),constrained_layout=True)
    for target,color,label in [('full','#0072B2','Common1514 x2694 target'),('low','#D55E00','Common757 x1347 target')]:
        per=data['compared_to_low'][target]['per_frame']
        for ax,metric in zip(axes,['head_psnr','head_gradient']):
            ax.plot(range(8),[r[metric] for r in per],label=label,marker='o',markersize=4,color=color,linewidth=1.4)
    for ax,title in zip(axes,['Head/shoulder PSNR change (higher is better)','Gradient fidelity error change (lower is better)']):
        ax.axhline(0,color='#56686b',linewidth=.8);ax.set_xticks(range(8),data['frames'],rotation=35,ha='right');ax.set_title(title);ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('Change / dB');axes[1].set_ylabel('Change / normalized error');axes[0].legend(frameon=False)
    fig.savefig(assets/'head-errors.png',dpi=180);plt.close(fig)
    (assets/'summary.json').write_text(json.dumps(data,indent=2))
    template=Path(__file__).with_name('native_resolution_report.html').read_text(encoding='utf-8')
    target=view/'native-resolution.html';assert not target.exists()
    target.write_text(template.replace('__DATA__',json.dumps(data).replace('</','<\\/')),encoding='utf-8')
    index=view/'index.html';html=index.read_text(encoding='utf-8')
    if 'href="native-resolution.html"' not in html:
        shutil.copy2(index,out/'index-before-resolution.html')
        html=html.replace('<a class="report-link"', '<a class="report-link" href="native-resolution.html">最新对照：训练分辨率与人物细节 ↗</a><a class="report-link"',1)
        index.write_text(html,encoding='utf-8')
    print(json.dumps(data['compared_to_low']),flush=True)


if __name__=='__main__':main()
