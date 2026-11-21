"""Build a real-image-first result page only after the complete variance audit."""
import ast
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

SCRIPTS=Path(__file__).parent
ROOT=SCRIPTS.parents[1]/'outputs/mini-splatting2-mip-depth-001'
VIEW=ROOT/'viewer';ASSETS=VIEW/'native-spread-assets'


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    assert read(ROOT/'native-spread-acceptance-queue.json')['status']=='complete'
    data=read(ROOT/'native-spread-candidate-review.json')
    assert data['status']=='complete' and data['accepted'] is False
    paths=ROOT/'native-spread-paths-001';assert read(paths/'manifest.json')['status']=='complete'
    assert not (VIEW/'native-spread.html').exists();ASSETS.mkdir(exist_ok=False)
    assignment=next(n for n in ast.parse((SCRIPTS/'local_refinement.py').read_text()).body
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SUBJECT_BOX' for t in n.targets))
    boxes=ast.literal_eval(assignment.value)
    frames=data['models']['control0']['head_rows']
    data['frames']=[r['frame'] for r in frames]
    data['replay_audit']=read(ROOT/'native-spread-zero-control-audit.json')
    data['observed_groups']=read(ROOT/'native-spread-observed-groups-001/summary.json')
    assert data['observed_groups']['status']=='complete'
    extra=read(ROOT/'additional-frame-spread-004/summary.json')
    data['extra_frames']=[dict(frame=r['frame'],source_index=r['source_index'])
        for r in extra['models']['control0']['rows'] if r['divisor']==1]
    for key in ['gt',*data['models']]:
        dest=ASSETS/key;dest.mkdir()
        name=data['models']['control0' if key=='gt' else key]['run']
        for i,row in enumerate(frames):
            filename=f'{i:05d}.png'
            source=ROOT/name/'model/test/ours_18000'/('gt' if key=='gt' else 'renders')/filename
            image=Image.open(source).convert('RGB');w,h=image.size
            box=tuple(round(a*b) for a,b in zip(boxes[int(row['frame'])],[w,h,w,h]))
            shutil.copy2(source,dest/filename)
            image.crop(box).save(dest/f'{i:05d}-person.png')
            image.crop(tuple(row['box'])).save(dest/f'{i:05d}-head.png')
            if key!='gt':
                gt_path=ROOT/data['models']['control0']['run']/'model/test/ours_18000/gt'/filename
                gt=np.asarray(Image.open(gt_path).convert('RGB'),dtype='float32')/255
                error=np.mean(np.abs(np.asarray(image,dtype='float32')/255-gt),axis=-1)
                colored=Image.fromarray((plt.get_cmap('magma')(np.clip(error/.1,0,1))[...,:3]*255).round().astype('uint8'))
                colored.save(dest/f'{i:05d}-error.png')
                colored.crop(box).save(dest/f'{i:05d}-person-error.png')
                colored.crop(tuple(row['box'])).save(dest/f'{i:05d}-head-error.png')
    for key in ['gt',*data['models']]:
        source=ROOT/'additional-frame-spread-004'/key
        dest=ASSETS/'extra'/key;dest.mkdir(parents=True)
        for image in source.iterdir():
            if image.suffix.lower() in ['.png','.jpg'] and (key=='gt' or image.name.endswith('-s1.png')):
                shutil.copy2(image,dest/image.name)
    for frame in data['frames']:
        source=ROOT/'depth-spread-candidates-002'/frame;dest=ASSETS/'spread'/frame;dest.mkdir(parents=True)
        for key in ['control0','spread0030','spread0100']:
            shutil.copy2(source/f'{key}-spread.png',dest/f'{key}-spread.png')
    dest=ASSETS/'paths';dest.mkdir()
    for name in ['elevation.mp4','distance.mp4','elevation-00.png','distance-00.png','manifest.json']:
        shutil.copy2(paths/name,dest/name)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'savefig.dpi':180})
    fig,axes=plt.subplots(1,2,figsize=(11,3.5),constrained_layout=True)
    for ax,metric,title in [(axes[0],'psnr','Head / shoulder PSNR change (higher is better)'),
                            (axes[1],'normalized_gradient_error','Gradient fidelity error change (lower is better)')]:
        for key,color,label in [('spread0030','#0072B2','Spread 0.003'),('spread0100','#D55E00','Spread 0.010')]:
            values=[a['head'][metric]-b['head'][metric] for a,b in zip(data['models'][key]['head_rows'],frames)]
            ax.plot(range(8),values,color=color,marker='o',markersize=4,linewidth=1.4,label=label)
        ax.axhline(0,color='#5d686e',linewidth=.8);ax.set_xticks(range(8),data['frames'],rotation=35,ha='right')
        ax.set_xlabel('Development frame');ax.set_title(title);ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('Change / dB');axes[1].set_ylabel('Change / normalized error');axes[0].legend(frameon=False)
    fig.savefig(ASSETS/'head-frame-errors.png');plt.close(fig)
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import Normalize
    for name,maximum,label in [('spread',.5,'Relative center-depth spread; gray = outside common opacity mask'),
                               ('error',.1,'Mean absolute RGB error; values above 0.10 use the maximum color')]:
        fig,ax=plt.subplots(figsize=(9,.75),constrained_layout=True)
        ColorbarBase(ax,cmap='magma',norm=Normalize(0,maximum),orientation='horizontal',label=label)
        fig.savefig(ASSETS/(name+'-legend.png'));plt.close(fig)
    (ASSETS/'summary.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    template=(SCRIPTS/'native_spread_report.html').read_text(encoding='utf-8')
    (VIEW/'native-spread.html').write_text(template.replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('</','<\\/')),encoding='utf-8')
    for filename in ['index.html','native-repeats.html']:
        target=VIEW/filename;html=target.read_text(encoding='utf-8')
        if 'href="native-spread.html"' not in html:
            shutil.copy2(target,ROOT/(filename.replace('.html','')+'-before-spread.html'))
            link='<a class="report-link" href="native-spread.html">深度分散约束：真实头肩细节与相机路径对照 ↗</a>'
            html=html.replace('<details>',link+'<details>',1) if filename=='index.html' else html.replace('</h1>','</h1><p>'+link+'</p>',1)
            target.write_text(html,encoding='utf-8')
    print(json.dumps(dict(report=str(VIEW/'native-spread.html'),status='published_development_evidence',accepted=False)))


if __name__=='__main__':main()
