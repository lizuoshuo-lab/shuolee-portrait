"""Present complete depth experiments with real images and explicit evidence limits."""
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

LAB=Path(__file__).resolve().parents[2]
ROOT=LAB/'outputs/mini-splatting2-mip-depth-001';VIEW=ROOT/'viewer';ASSETS=VIEW/'native-depth-assets'
RUNS={'combo0':'native-motion-mip-depthmask-18000-s0',
      'depth003':'native-depthguide003-18000-s0','depth010':'native-depthguide010-18000-s0',
      'combo1':'native-motion-mip-depthmask-18000-s1','combo2':'native-motion-mip-depthmask-18000-s2'}


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def write(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    assert read(ROOT/'native-depth-stage-queue.json')['status']=='complete'
    assert read(ROOT/'native-depth-evaluation-queue.json')['status']=='complete'
    results={key:read(ROOT/name/'evaluation-summary.json') for key,name in RUNS.items()}
    geom=read(ROOT/'native-depth-geometry-001/summary.json')
    extra=read(ROOT/'additional-frame-depth-evaluation-002/summary.json')
    previous=read(ROOT/'additional-frame-evaluation-001/summary.json')
    diffusion=read(ROOT/'difix-training-support-gating-001/summary.json')
    ASSETS.mkdir(exist_ok=True)
    paths=ROOT/'native-depth-paths-001'
    assert read(paths/'manifest.json')['status']=='complete'
    shutil.copytree(paths,ASSETS/'paths',dirs_exist_ok=True)
    box_assign=next(n for n in ast.parse(Path(__file__).with_name('local_refinement.py').read_text()).body
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SUBJECT_BOX' for t in n.targets))
    boxes=ast.literal_eval(box_assign.value)
    for key,name in RUNS.items():
        dest=ASSETS/key;dest.mkdir(exist_ok=True)
        for j,row in enumerate(results[key]['native_rows']):
            source=ROOT/name/'model/test/ours_18000/renders'/f'{j:05d}.png'
            shutil.copy2(source,dest/f'{j:05d}.png')
            image=Image.open(source);w,h=image.size;b=boxes[int(row['frame'])]
            crop=tuple(round(v*s) for v,s in zip(b,[w,h,w,h]))
            person=image.crop(crop);person.save(dest/f'{j:05d}-person.png')
            person.crop((0,0,person.width,round(person.height/3))).save(dest/f'{j:05d}-head.png')
        for p in (ROOT/name).glob('elevation-*.png'):shutil.copy2(p,dest/p.name)
        for p in (ROOT/name).glob('distance-*.png'):shutil.copy2(p,dest/p.name)
    gt_dir=ASSETS/'gt';gt_dir.mkdir(exist_ok=True)
    for j,row in enumerate(results['combo0']['native_rows']):
        source=ROOT/RUNS['combo0']/'model/test/ours_18000/gt'/f'{j:05d}.png'
        shutil.copy2(source,gt_dir/f'{j:05d}.png')
        image=Image.open(source);w,h=image.size;b=boxes[int(row['frame'])]
        person=image.crop(tuple(round(v*s) for v,s in zip(b,[w,h,w,h])));person.save(gt_dir/f'{j:05d}-person.png')
        person.crop((0,0,person.width,round(person.height/3))).save(gt_dir/f'{j:05d}-head.png')
    extra_assets=ASSETS/'extra';extra_assets.mkdir(exist_ok=True)
    for key in ['gt',*RUNS]:
        source=ROOT/'additional-frame-evaluation-001'/('depthmask' if key=='combo0' else 'gt') if key in ['gt','combo0'] else ROOT/'additional-frame-depth-evaluation-002'/key
        shutil.copytree(source,extra_assets/key,dirs_exist_ok=True)
    dif_assets=ASSETS/'diffusion';dif_assets.mkdir(exist_ok=True)
    for p in (ROOT/'difix-training-support-gating-001').glob('*.png'):shutil.copy2(p,dif_assets/p.name)
    for p in (ROOT/'difix-appearance-test-r1').glob('*.png'):shutil.copy2(p,dif_assets/p.name)
    baseline={metric:dict(mean=statistics.mean(results[k]['native_mean'][metric] for k in ['combo0','combo1','combo2']),
        sample_std=statistics.stdev(results[k]['native_mean'][metric] for k in ['combo0','combo1','combo2']),
        seeds={k:results[k]['native_mean'][metric] for k in ['combo0','combo1','combo2']})
        for metric in ['full_psnr','crop_psnr','ssim','lpips_alex']}
    ext={key:previous['models']['depthmask'] if key=='combo0' else extra['models'][key] for key in RUNS}
    geometry={key:geom['models'][name] for key,name in RUNS.items()}
    overview=dict(status='two_guided_weights_and_three_baseline_seeds_evaluated',runs=RUNS,results=results,
        baseline_repeat_statistics=baseline,additional_frames=ext,geometry=geometry,diffusion=diffusion,
        guided_repeat_count=1,baseline_repeat_count=3,independent_final_test=False,
        caveats=['All results are development evidence from one video. Twenty-four extra frames were already inspected before these new variants.',
            'Depth priors and SfM are not independent measured depth. Contradiction counts are not confirmed floaters.',
            'Diffusion gate is a two-dimensional follow-up on the earlier fixed-topology model, not the new native-depth candidates.',
            'No generated pixels entered training. Wide pitch has no photographic GT and visibly remains distorted.'])
    write(ROOT/'native-depth-summary.json',overview);write(ASSETS/'summary.json',overview)
    for source,name in [(ROOT/'native-depth-runtime/validation.json','gradient-validation.json'),
                        (ROOT/'native-depth-geometry-001/protocol.json','geometry-protocol.json'),
                        (ROOT/'additional-frame-depth-evaluation-002/frozen-protocol.json','extra-protocol.json'),
                        (ROOT/'difix-training-support-gating-001/masks-frozen-before-scoring.json','diffusion-gate-protocol.json')]:
        shutil.copy2(source,ASSETS/name)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.6),constrained_layout=True)
    for ax,metric,title in zip(axes,['full','subject_crop'],['Full image','Fixed person region']):
        for key,color,label in [('depth003','#0072B2','Depth weight 0.003'),('depth010','#D55E00','Depth weight 0.010')]:
            delta=[a[metric]['psnr']-b[metric]['psnr'] for a,b in zip(results[key]['native_rows'],results['combo0']['native_rows'])]
            ax.plot(range(8),delta,marker='o',linewidth=1.6,markersize=4,color=color,label=label)
        ax.axhline(0,color='#6e7780',linewidth=.8);ax.set_xticks(range(8),[r['frame'].lstrip('0') or '0' for r in results['combo0']['native_rows']])
        ax.set_xlabel('Development frame');ax.set_ylabel('PSNR change vs matched seed 0 / dB');ax.set_title(title);ax.grid(axis='y',alpha=.18)
    axes[0].legend(frameon=False,fontsize=9);fig.savefig(ASSETS/'paired-frame-error.png');plt.close(fig)
    template=Path(__file__).with_name('native_depth_report.html').read_text(encoding='utf-8')
    template=template.replace('<option value="person">人物区域</option>','<option value="head">头肩局部</option><option value="person">人物区域</option>')
    template=template.replace("part=$('region').value==='person'?'-person':''", "part=$('region').value==='person'?'-person':$('region').value==='head'?'-head':''")
    template=template.replace('先选视角，再切换人物区域。','先选视角，再切换完整照片、人物区域或头肩局部。头肩局部固定取人物矩形上方三分之一，不做超分辨率。')
    video='''<h3>连续观察同一条相机路径</h3><p>每段60帧，三列分别是原组合、深度0.003和深度0.01；三者使用完全相同的相机轨迹。以下相机轨迹超出拍摄范围，没有对应真实视频。</p><div class="controls"><label>连续路径 <select id="video-kind"><option value="elevation">俯仰变化</option><option value="distance">距离变化</option></select></label></div><video id="path-video" style="width:100%;height:auto;max-width:1140px;background:#172d3a;border-radius:7px" controls preload="metadata" poster="native-depth-assets/paths/elevation-15.png" src="native-depth-assets/paths/elevation.mp4" aria-label="三个模型在相同外推相机路径上的连续渲染"></video>'''
    template=template.replace('<div class="controls"><label>相机路径',video+'<div class="controls"><label>相机路径',1)
    template=template.replace('</script>',"$('video-kind').onchange=()=>{const kind=$('video-kind').value;$('path-video').src=root+'paths/'+kind+'.mp4';$('path-video').poster=root+'paths/'+kind+'-15.png';};</script>")
    template=template.replace('保留前景高斯计数只表示','被标记的前景高斯计数只表示')
    template=template.replace('<p>本轮没有把生成图像用于高斯训练，','<p>门控后只修改了约0.4%的像素，PSNR改善约0.002 dB，新增收益可以忽略。它减少了无约束修正的错误改动，目前不作为主线训练模块。</p><p>本轮没有把生成图像用于高斯训练，')
    template=template.replace('__DATA__',json.dumps(overview,ensure_ascii=False).replace('</','<\/'))
    (VIEW/'native-depth.html').write_text(template,encoding='utf-8')
    for name in ['index.html','native-mip.html','report.html','motion.html']:
        p=VIEW/name;html=p.read_text(encoding='utf-8')
        link='<a class="report-link" href="native-depth.html">深度指导：原图、几何与扩散对照 ↗</a>'
        if 'href="native-depth.html"' not in html:
            html=html.replace('<details>',link+'<details>',1) if name=='index.html' else html.replace('</h1>','</h1><p>'+link+'</p>',1)
            p.write_text(html,encoding='utf-8')
    print(json.dumps(dict(status=overview['status'],report=str(VIEW/'native-depth.html'),baseline=baseline),ensure_ascii=False))


if __name__=='__main__':main()
