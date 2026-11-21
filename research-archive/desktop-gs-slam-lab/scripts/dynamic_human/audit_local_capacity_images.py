"""Verify actual PNGs, training/heldout scope, and create unenhanced comparisons."""
from pathlib import Path
import json
import statistics
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from audit_street_joint_images import check_quantization,sha

LAB=Path(__file__).resolve().parents[2]
OUT=LAB/'outputs/dynamic-human-street-local-capacity-020'
SOURCE=LAB/'outputs/dynamic-human-street-pose-convergence-016'
SCENE=LAB/'data/dynamic-human/argoverse2/processed/training/000'
BRANCHES=('single_view','adjacent_pair')

def pixels(path):return np.asarray(Image.open(path).convert('RGB'))
def gt_for(f,c,size):
    return np.asarray(Image.open(SCENE/'images'/f'{f:03d}_{c}.jpg').convert('RGB').resize(size,Image.Resampling.BILINEAR))

def main():
    rois=json.loads((SOURCE/'person-rois.json').read_text())['regions']
    lookup={(r['frame'],r['camera'],r['instance']):r for r in rois};assert len(lookup)==435
    train_rois=json.loads((SOURCE/'training-rois.json').read_text())['regions']
    assert len(train_rois)==101
    summaries=[]
    for branch in BRANCHES:
        run=OUT/branch;destination=run/'independent-image-audit.json'
        assert (run/'completion.json').exists() and not destination.exists()
        m=json.loads((run/'evaluation/final/metrics.json').read_text())
        assert len(m['views'])==105 and {(r['frame'],r['camera']) for r in m['views']}=={(f,c) for f in range(10,151,10) for c in range(7)}
        scores={(r['frame'],r['camera'],r['instance']):r for r in m['person_box_regions']}
        assert len(scores)==435 and scores.keys()==lookup.keys()
        images=[];regions=[]
        for view in m['views']:
            f,c=view['frame'],view['camera'];stem=f'frame{f:03d}-camera{c}'
            file=run/'evaluation/final'/(stem+'-prediction.png');pred=pixels(file);target=gt_for(f,c,(pred.shape[1],pred.shape[0]))
            assert np.array_equal(target,pixels(run/'ground-truth'/(stem+'.png')))
            psnr,diff=check_quantization(target,pred,view['psnr_db'])
            images.append(dict(frame=f,camera=c,sha256=sha(file),png_psnr_db=psnr,quantization_difference_db=diff))
            for key,roi in lookup.items():
                if key[:2]!=(f,c):continue
                row=scores[key];assert row['box']==roi['box']
                x0,y0,x1,y1=roi['box'];psnr,diff=check_quantization(target[y0:y1,x0:x1],pred[y0:y1,x0:x1],row['psnr_db'])
                if key[2]==79:
                    assert np.array_equal(pred[y0:y1,x0:x1],pixels(run/'evaluation/final'/(stem+'-person79.png')))
                regions.append(dict(frame=f,camera=c,instance=key[2],float_psnr_db=row['psnr_db'],png_psnr_db=psnr,ssim=row['ssim'],quantization_difference_db=diff))
        training=[]
        for label in ('before','after'):
            metric=json.loads((run/'training-evaluation'/label/'metrics.json').read_text())
            assert metric['count']==len(metric['rows'])==101
            assert {(r['frame'],r['camera']) for r in metric['rows']}=={(r['frame'],r['camera']) for r in train_rois}
            for f,c in [(129,4),(131,4),(149,6),(151,6)]:
                row=next(r for r in metric['rows'] if (r['frame'],r['camera'])==(f,c));stem=f'frame{f:03d}-camera{c}'
                pred=pixels(run/'training-evaluation'/label/(stem+'-prediction.png'));target=gt_for(f,c,(pred.shape[1],pred.shape[0]))
                assert np.array_equal(target,pixels(run/'training-ground-truth'/(stem+'.png')))
                x0,y0,x1,y1=row['box'];check_quantization(target[y0:y1,x0:x1],pred[y0:y1,x0:x1],row['psnr_db'])
            selected=[r for r in metric['rows'] if (r['frame'],r['camera']) in [(129,4),(131,4)]]
            training.append(dict(label=label,all101_mean_psnr_db=metric['mean_person_psnr_db'],selected_views=selected,saved_images_checked=4))
        snapshots=[]
        for iteration in (0,100,500,1000):
            directory=run/'snapshots'/f'update{iteration:04d}'
            rows=json.loads((directory/'metrics.json').read_text())['rows'];assert len(rows)==2
            for r in rows:
                f,c=r['frame'],r['camera'];stem=f'frame{f:03d}-camera{c}'
                assert (f,c) in [(129,4),(131,4)]
                pred=pixels(directory/(stem+'-prediction.png'));target=gt_for(f,c,(pred.shape[1],pred.shape[0]))
                assert np.array_equal(target,pixels(run/'snapshots/ground-truth'/(stem+'.png')))
                x0,y0,x1,y1=r['box'];psnr,diff=check_quantization(target[y0:y1,x0:x1],pred[y0:y1,x0:x1],r['psnr_db'])
                snapshots.append(dict(iteration=iteration,frame=f,camera=c,png_psnr_db=psnr,reported_psnr_db=r['psnr_db']))
        target=[r for r in regions if r['instance']==79];assert len(target)==13
        report=dict(status='all105heldout435regions_and_training_snapshots_verified',branch=branch,
            images=images,regions=regions,training=training,snapshots=snapshots,quality_accepted=False,
            summary=dict(heldout_target13_mean_psnr_db=statistics.mean(r['float_psnr_db'] for r in target),
                heldout_target13_mean_ssim=statistics.mean(r['ssim'] for r in target)),
            boundaries=['PNG PSNR independently recomputed with quantization bounds; SSIM remains reported float evaluator score.',
                'Only4saved training full images per before/after stage and8training snapshot images per branch independently checked.',
                'One/two training-view fitting diagnostic, not accepted scene reconstruction.'])
        destination.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
        summaries.append(dict(branch=branch,summary=report['summary'],training=training))
    directory=OUT/'paired-audit';directory.mkdir(exist_ok=True)
    assert not (directory/'figures.json').exists()
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19);figures=[]
    requested=[('training',129,4),('training',131,4)]+[('heldout',f,c) for f,c,i in sorted(lookup) if i==79]
    for kind,f,c in requested:
        stem=f'frame{f:03d}-camera{c}'
        source_sub='training-evaluation/after' if kind=='training' else 'evaluation/final'
        gt_sub='training-ground-truth' if kind=='training' else 'ground-truth'
        target=Image.open(SOURCE/gt_sub/(stem+'-person79.png')).convert('RGB')
        panels=[target,Image.open(SOURCE/source_sub/(stem+'-person79.png')).convert('RGB')]
        sub='training-evaluation/after' if kind=='training' else 'evaluation/final'
        panels += [Image.open(OUT/b/sub/(stem+'-person79.png')).convert('RGB') for b in BRANCHES]
        scale=min(3.,280/target.width,650/target.height);size=tuple(round(x*scale) for x in target.size)
        canvas=Image.new('RGB',(1200,size[1]+88),'white');draw=ImageDraw.Draw(canvas)
        for col,(label,panel) in enumerate(zip(['原始观测','共同起点','仅拟合训练帧129','拟合训练帧129与131'],panels)):
            assert panel.size==target.size
            draw.text((col*300+8,8),label,font=font,fill='#182d39')
            canvas.paste(panel.resize(size,Image.Resampling.NEAREST),(col*300+(300-size[0])//2,38))
        role='已有训练时刻' if kind=='training' else '留出时刻'
        draw.text((10,size[1]+48),f'帧 {f} / 相机 {c} · {role} · 后两组各1000次更新；未锐化',font=font,fill='#182d39')
        path=directory/(stem+'-person79-comparison.png');canvas.save(path)
        figures.append(dict(file=path.name,frame=f,camera=c,kind=kind,sha256=sha(path)))
    (directory/'figures.json').write_text(json.dumps(dict(figures=figures,visual_review_pending=True),indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status='both_image_audits_and15figures_complete',branches=summaries),indent=2))

if __name__=='__main__':main()
