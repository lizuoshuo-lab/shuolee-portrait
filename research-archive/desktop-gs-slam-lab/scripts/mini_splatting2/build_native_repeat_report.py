"""Publish matched-seed appearance and ray-depth diagnostics as a new snapshot."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

LAB=Path(__file__).resolve().parents[2];ROOT=LAB/'outputs/mini-splatting2-mip-depth-001'
VIEW=ROOT/'viewer';ASSETS=VIEW/'native-repeat-assets'


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    assert read(ROOT/'depth-repeat-acceptance-queue.json')['status']=='complete'
    assert not (VIEW/'native-repeats.html').exists()
    matched=read(ROOT/'native-depth-matched-repeats.json')
    head={weight:read(ROOT/f'head-detail-depth{weight}-001.json') for weight in ['003','010']}
    spread=read(ROOT/'depth-spread-evaluation-001/summary.json')
    epipolar=read(ROOT/'head-epipolar-audit-001/summary.json')
    earlier=read(ROOT/'additional-frame-evaluation-001/summary.json')['models']
    second=read(ROOT/'additional-frame-depth-evaluation-002/summary.json')['models']
    latest=read(ROOT/'additional-frame-depth-repeats-003/summary.json')['models']
    geom_old=read(ROOT/'native-depth-geometry-001/summary.json')['models']
    geom_new=read(ROOT/'native-depth-geometry-repeats-002/summary.json')['models']
    runs={f'{key}{seed}':f'{prefix}-18000-s{seed}' for key,prefix in
        [('combined','native-motion-mip-depthmask'),('depth003','native-depthguide003'),('depth010','native-depthguide010')]
        for seed in range(3)}
    ASSETS.mkdir(exist_ok=False)
    assignment=next(n for n in ast.parse(Path(__file__).with_name('local_refinement.py').read_text()).body
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SUBJECT_BOX' for t in n.targets))
    boxes=ast.literal_eval(assignment.value);results={};extra={};geometry={};models={}
    for key,name in runs.items():
        results[key]=read(ROOT/name/'evaluation-summary.json');manifest=read(ROOT/name/'manifest.json')
        models[key]=dict(name=name,sha256=manifest.get('baked_ply_sha256'),gaussians=results[key]['gaussians'])
        geometry[key]=(geom_old if name in geom_old else geom_new)[name]
        if key=='combined0':extra[key]=earlier['depthmask'];source=ROOT/'additional-frame-evaluation-001/depthmask'
        elif key in ['combined1','combined2']:
            old_key='combo'+key[-1];extra[key]=second[old_key];source=ROOT/'additional-frame-depth-evaluation-002'/old_key
        elif key[-1]=='0':
            old_key=key[:-1];extra[key]=second[old_key];source=ROOT/'additional-frame-depth-evaluation-002'/old_key
        else:extra[key]=latest[name];source=ROOT/'additional-frame-depth-repeats-003'/name
        dest=ASSETS/key;dest.mkdir()
        for index,row in enumerate(results[key]['native_rows']):
            path=ROOT/name/'model/test/ours_18000/renders'/f'{index:05d}.png'
            image=Image.open(path).convert('RGB');w,h=image.size;b=boxes[int(row['frame'])]
            shutil.copy2(path,dest/f'{index:05d}.png')
            person=image.crop(tuple(round(v*s) for v,s in zip(b,[w,h,w,h])));person.save(dest/f'{index:05d}-person.png')
            person.crop((0,0,person.width,round(person.height/3))).save(dest/f'{index:05d}-head.png')
        extra_dest=ASSETS/'extra'/key;extra_dest.mkdir(parents=True)
        for p in source.glob('*-s1.png'):shutil.copy2(p,extra_dest/p.name)
    shutil.copytree(VIEW/'native-depth-assets/gt',ASSETS/'gt')
    shutil.copytree(ROOT/'additional-frame-evaluation-001/gt',ASSETS/'extra/gt')
    for frame_dir in (ROOT/'depth-spread-evaluation-001').iterdir():
        if frame_dir.is_dir():
            dest=ASSETS/'spread'/frame_dir.name;dest.mkdir(parents=True)
            for p in frame_dir.glob('*.png'):shutil.copy2(p,dest/p.name)
            # Display-only rescale after observing the diagnostic range. Frozen
            # measurements and the originally predeclared0.15 images stay intact.
            for p in frame_dir.glob('*-maps.npz'):
                with np.load(p) as arrays:
                    rgb=(plt.get_cmap('magma')(np.clip(arrays['relative_std']/.5,0,1))[...,:3]*255).round().astype('uint8')
                    rgb[~arrays['common_mask']]=[180,180,180]
                Image.fromarray(rgb).save(dest/(p.name.replace('-maps.npz','-spread.png')))
    extra_stats={}
    for weight in ['003','010']:
        extra_stats[weight]={metric:dict(values=[extra[f'depth{weight}{seed}']['aggregate']['1'][metric]-extra[f'combined{seed}']['aggregate']['1'][metric] for seed in range(3)])
            for metric in ['full_psnr','crop_psnr','lpips_alex']}
        for entry in extra_stats[weight].values():entry.update(mean=statistics.mean(entry['values']),sample_std=statistics.stdev(entry['values']))
    data=dict(status='complete_matched_development_audit',matched=matched,head=head,results=results,
        extra=extra,extra_statistics=extra_stats,geometry=geometry,spread=spread,epipolar=epipolar,models=models,
        scope='One work video; matched3seeds; development views already inspected; no independent geometry or extrapolation GT',
        model_promotion='none: head/shoulder clarity is evaluated separately from full-image gains',
        all_existing_models_preserved=True)
    (ASSETS/'summary.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'native-depth-repeat-review.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),constrained_layout=True)
    for index,(metric,title) in enumerate([('full_psnr','Full image'),('crop_psnr','Person region'),('head','Head / shoulders')]):
        ax=axes[index]
        for weight,color,label in [('003','#0072B2','Depth0.003'),('010','#D55E00','Depth0.010')]:
            values=head[weight]['aggregate']['difference']['psnr'] if metric=='head' else matched['models']['depth'+weight]['paired_difference'][metric]
            y=[row['difference']['psnr'] for row in head[weight]['by_seed']] if metric=='head' else values['values']
            ax.plot(range(3),y,marker='o',color=color,label=label,linewidth=1.5)
        ax.axhline(0,color='#576671',linewidth=.8);ax.set_xticks(range(3));ax.set_xlabel('Matched random seed');ax.set_title(title)
        ax.set_ylabel('PSNR change / dB');ax.grid(axis='y',alpha=.15)
    axes[0].legend(frameon=False,fontsize=9);fig.savefig(ASSETS/'matched-seed-gains.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9, .75),constrained_layout=True)
    from matplotlib.colors import Normalize
    from matplotlib.colorbar import ColorbarBase
    ColorbarBase(ax,cmap='magma',norm=Normalize(0,.5),orientation='horizontal',label='Relative depth spread (dimensionless; >=0.50 clipped for display)')
    fig.savefig(ASSETS/'spread-legend.png');plt.close(fig)
    template=Path(__file__).with_name('native_repeat_report.html').read_text(encoding='utf-8')
    (VIEW/'native-repeats.html').write_text(template.replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('</','<\\/')),encoding='utf-8')
    for filename in ['index.html','native-depth.html']:
        path=VIEW/filename;html=path.read_text(encoding='utf-8')
        link='<a class="report-link" href="native-repeats.html">三次匹配重复：人物清晰度与深度分散 ↗</a>'
        if 'href="native-repeats.html"' not in html:
            html=html.replace('<details>',link+'<details>',1) if filename=='index.html' else html.replace('</h1>','</h1><p>'+link+'</p>',1)
            path.write_text(html,encoding='utf-8')
    print(json.dumps(dict(status=data['status'],report=str(VIEW/'native-repeats.html'),additional24=extra_stats),ensure_ascii=False))


if __name__=='__main__':main()
